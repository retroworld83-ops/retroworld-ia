from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

SERVICE_NAME = os.getenv("SERVICE_NAME", "retroworld-ia")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

KB_DIR = os.getenv("KB_DIR", BASE_DIR)

# Render fournit PORT en env. On garde 10000 par défaut (Render Docker)
PORT = int(os.getenv("PORT", "10000"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "800"))

ADMIN_DASHBOARD_TOKEN = os.getenv("ADMIN_DASHBOARD_TOKEN", "").strip()
USER_HISTORY_TOKEN = os.getenv("USER_HISTORY_TOKEN", "").strip()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(SERVICE_NAME)

app = Flask(__name__, static_folder="static")
CORS(app)

# ---------------------------------------------------------
# OpenAI client (lazy)
# ---------------------------------------------------------

_openai_client = None
try:
    if OPENAI_API_KEY:
        from openai import OpenAI  # type: ignore

        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logger.warning("OpenAI client init failed: %s", e)
    _openai_client = None


# ---------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------


def _safe_read_json(path: str, default: Any) -> Any:
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _safe_write_json(path: str, data: Any) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.warning("write json failed (%s): %s", path, e)
        return False


# ---------------------------------------------------------
# user index (user_id -> conversation_id)
# ---------------------------------------------------------

USER_INDEX_PATH = os.path.join(DATA_DIR, "user_index.json")


def get_user_index() -> Dict[str, str]:
    return _safe_read_json(USER_INDEX_PATH, {}) or {}


def set_user_conversation(user_id: str, conversation_id: str) -> None:
    idx = get_user_index()
    idx[str(user_id)] = str(conversation_id)
    _safe_write_json(USER_INDEX_PATH, idx)


def get_user_conversation(user_id: str) -> Optional[str]:
    idx = get_user_index()
    return idx.get(str(user_id))


# ---------------------------------------------------------
# KB (knowledge base) cache
# ---------------------------------------------------------


@dataclass
class _KBCacheEntry:
    ts: float
    kb: Dict[str, Any]


_KB_CACHE: Dict[str, _KBCacheEntry] = {}
_KB_CACHE_TTL_SEC = 30.0


def load_kb(brand: str) -> Dict[str, Any]:
    brand = (brand or "").lower().strip()
    if brand not in ("retroworld", "runningman"):
        brand = "retroworld"

    now = time.time()
    entry = _KB_CACHE.get(brand)
    if entry and (now - entry.ts) < _KB_CACHE_TTL_SEC:
        return entry.kb

    filename = f"kb_{brand}.json"
    path = os.path.join(KB_DIR, filename)
    kb = _safe_read_json(path, {"brand": brand, "items": []})
    if not isinstance(kb, dict):
        kb = {"brand": brand, "items": []}

    _KB_CACHE[brand] = _KBCacheEntry(ts=now, kb=kb)
    return kb


def save_kb(brand: str, kb: Dict[str, Any]) -> bool:
    brand = (brand or "").lower().strip()
    if brand not in ("retroworld", "runningman"):
        brand = "retroworld"
    filename = f"kb_{brand}.json"
    path = os.path.join(KB_DIR, filename)
    ok = _safe_write_json(path, kb)
    if ok:
        _KB_CACHE[brand] = _KBCacheEntry(ts=time.time(), kb=kb)
    return ok


# ---------------------------------------------------------
# OpenAI wrapper + basic NLP
# ---------------------------------------------------------


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def call_openai_chat(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    if not _openai_client:
        return (
            "Le service IA n’est pas configuré (OPENAI_API_KEY manquant).",
            {"error": "openai_not_configured"},
        )

    resp = _openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
    )
    text = (resp.choices[0].message.content or "").strip()

    usage: Dict[str, Any] = {}
    try:
        usage = resp.usage.model_dump()  # type: ignore
    except Exception:
        try:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
        except Exception:
            usage = {}

    return text, usage


def detect_brand_from_text(text: str) -> str:
    t = _norm(text)
    if "runningman" in t or "running man" in t or "action game" in t:
        return "runningman"
    return "retroworld"


def _is_reservation_intent(text: str) -> bool:
    t = _norm(text)
    keys = [
        "réserver",
        "reservation",
        "réservation",
        "dispo",
        "disponible",
        "créneau",
        "horaire",
        "aujourd",
        "demain",
        "ce week",
        "samedi",
        "dimanche",
    ]
    return any(k in t for k in keys)


# ---------------------------------------------------------
# Conversation storage
# ---------------------------------------------------------

CONV_DIR = os.path.join(DATA_DIR, "conversations")
os.makedirs(CONV_DIR, exist_ok=True)


def _conversation_path(conversation_id: str) -> str:
    conversation_id = re.sub(r"[^a-zA-Z0-9_\-]", "", conversation_id or "")
    if not conversation_id:
        conversation_id = "conv_" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return os.path.join(CONV_DIR, f"{conversation_id}.json")


def load_conversation(conversation_id: str) -> List[Dict[str, Any]]:
    path = _conversation_path(conversation_id)
    data = _safe_read_json(path, [])
    if not isinstance(data, list):
        return []
    return data


def append_conversation_record(
    conversation_id: str,
    brand: str,
    user_messages: List[Dict[str, Any]],
    assistant_reply: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = _conversation_path(conversation_id)
    records = _safe_read_json(path, [])
    if not isinstance(records, list):
        records = []
    records.append(
        {
            "ts": datetime.utcnow().isoformat() + "Z",
            "brand": brand,
            "user_messages": user_messages,
            "assistant_reply": assistant_reply,
            "extra": extra or {},
        }
    )
    _safe_write_json(path, records)


def list_conversations() -> List[str]:
    try:
        files = [f for f in os.listdir(CONV_DIR) if f.endswith(".json")]
        files.sort(reverse=True)
        return [os.path.splitext(f)[0] for f in files]
    except Exception:
        return []


def prune_history_for_prompt(
    history: List[Dict[str, Any]], max_turns: int = 18
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not isinstance(history, list):
        return out
    turns = 0
    for rec in reversed(history):
        if turns >= max_turns:
            break
        if not isinstance(rec, dict):
            continue
        user_msgs = rec.get("user_messages") or []
        if isinstance(user_msgs, list):
            for um in user_msgs:
                if isinstance(um, dict) and um.get("role") == "user":
                    out.append({"role": "user", "content": str(um.get("content") or "")})
        ar = rec.get("assistant_reply") or ""
        out.append({"role": "assistant", "content": str(ar)})
        turns += 1
    out.reverse()
    return out


def classify_conversation_brands(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    brands_seen = set()
    last_effective = None
    for rec in records:
        extra = rec.get("extra") or {}
        meta = extra.get("metadata") or {}
        be = extra.get("brand_effective") or meta.get("brand_effective") or rec.get("brand")
        if be in ("runningman", "retroworld"):
            brands_seen.add(be)
            last_effective = be
    if not brands_seen:
        return {"brand_final": "unknown", "brands_seen": []}
    if len(brands_seen) == 1:
        return {"brand_final": list(brands_seen)[0], "brands_seen": sorted(brands_seen)}
    return {"brand_final": last_effective or "unknown", "brands_seen": sorted(brands_seen)}


# ---------------------------------------------------------
# Core chat pipeline
# ---------------------------------------------------------


def process_chat(
    brand_entry: str,
    messages: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    brand_entry = (brand_entry or "auto").lower()

    if brand_entry == "auto":
        last_user = ""
        for m in reversed(messages or []):
            if isinstance(m, dict) and m.get("role") == "user":
                last_user = str(m.get("content") or "")
                break
        brand_effective = detect_brand_from_text(last_user)
    else:
        brand_effective = brand_entry

    kb = load_kb(brand_effective)
    kb_items = kb.get("items") or []
    if not isinstance(kb_items, list):
        kb_items = []

    system = (
        "Vous êtes un assistant pour deux marques: Retroworld (VR/Quiz/Salle enfant) et Runningman Games (action game).\n"
        "Répondez de façon claire, précise, utile, sans inventer.\n"
        "Si l’utilisateur veut réserver: répondre 'disponible' puis proposer la suite.\n"
        "Respectez la marque et ses règles.\n"
    )

    if kb_items:
        system += "\nConnaissances (KB) :\n"
        for it in kb_items[:140]:
            if isinstance(it, str):
                system += f"- {it}\n"
            elif isinstance(it, dict):
                title = it.get("title") or it.get("name") or ""
                content = it.get("content") or it.get("text") or it.get("value") or ""
                s = _norm(f"{title} {content}")
                if s:
                    system += f"- {title}: {content}\n" if title else f"- {content}\n"

    openai_messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

    conversation_id = str(metadata.get("conversation_id") or "").strip()
    if conversation_id:
        hist = load_conversation(conversation_id)
        openai_messages.extend(prune_history_for_prompt(hist, max_turns=12))

    clipped: List[Dict[str, str]] = []
    for m in (messages or [])[-20:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and content is not None:
            clipped.append({"role": role, "content": str(content)})
    openai_messages.extend(clipped)

    last_user_text = ""
    for m in reversed(clipped):
        if m["role"] == "user":
            last_user_text = m["content"]
            break

    guard_hits: List[str] = []
    if _is_reservation_intent(last_user_text):
        guard_hits.append("reservation_intent")

    reply_text, usage = call_openai_chat(openai_messages)

    return {
        "reply": reply_text,
        "brand_entry": brand_entry,
        "brand_effective": brand_effective,
        "openai_usage": usage,
        "guard_hits": guard_hits,
        "metadata": metadata,
    }


# ---------------------------------------------------------
# AUTH HELPERS
# ---------------------------------------------------------


def _require_admin_token(req) -> bool:
    tok = (req.args.get("token") or "").strip()
    if not tok:
        tok = (req.headers.get("X-Admin-Token") or "").strip()
    return bool(tok) and bool(ADMIN_DASHBOARD_TOKEN) and tok == ADMIN_DASHBOARD_TOKEN


def _require_user_token(req) -> bool:
    tok = (req.args.get("token") or "").strip()
    if not tok:
        tok = (req.headers.get("X-User-Token") or "").strip()
    return bool(tok) and bool(USER_HISTORY_TOKEN) and tok == USER_HISTORY_TOKEN


def _coerce_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Accept multiple client payload formats and normalize to OpenAI-style messages."""
    raw = payload.get("messages")
    if isinstance(raw, list):
        out: List[Dict[str, str]] = []
        for m in raw:
            if not isinstance(m, dict):
                continue
            role = (m.get("role") or "").strip()
            content = m.get("content")
            if role in ("user", "assistant") and content is not None:
                out.append({"role": role, "content": str(content)})
        return out

    out: List[Dict[str, str]] = []
    history = payload.get("history") or []
    if isinstance(history, list):
        for item in history:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                u, a = item
                if u:
                    out.append({"role": "user", "content": str(u)})
                if a:
                    out.append({"role": "assistant", "content": str(a)})
                continue

            if isinstance(item, dict):
                role = (item.get("role") or "").strip()
                content = item.get("content")
                if role in ("user", "assistant") and content is not None:
                    out.append({"role": role, "content": str(content)})

    msg = payload.get("message")
    if msg:
        out.append({"role": "user", "content": str(msg)})

    return out


def _coerce_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = payload.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------


@app.route("/chat", methods=["POST"])
def chat_compat():
    """Backwards-compatible endpoint for clients that POST to /chat."""
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    messages = _coerce_messages(payload)
    if not messages:
        return jsonify({"error": "invalid_messages"}), 400

    metadata = _coerce_metadata(payload)

    out = process_chat(brand_entry="auto", messages=messages, metadata=metadata)

    # ✅ Compat: certains fronts lisent reply, d'autres answer
    reply = out.get("reply", "")
    return jsonify(
        {
            "reply": reply,
            "answer": reply,
            "brand_effective": out.get("brand_effective"),
            "brand_entry": out.get("brand_entry"),
        }
    ), 200


@app.route("/", methods=["GET"])
def root():
    return (
        jsonify(
            {
                "service": SERVICE_NAME,
                "status": "ok",
                "time_utc": datetime.utcnow().isoformat() + "Z",
            }
        ),
        200,
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/chat/<brand>", methods=["POST"])
def chat_route(brand: str):
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    messages = _coerce_messages(payload)
    if not messages:
        return jsonify({"error": "invalid_messages"}), 400

    metadata = _coerce_metadata(payload)

    conversation_id = str(payload.get("conversation_id") or "").strip()
    user_id = str(payload.get("user_id") or "").strip()

    if user_id and not conversation_id:
        conversation_id = get_user_conversation(user_id) or ""

    if not conversation_id:
        conversation_id = "conv_" + datetime.utcnow().strftime("%Y%m%d%H%M%S")

    if user_id:
        set_user_conversation(user_id, conversation_id)

    metadata = dict(metadata)
    metadata.setdefault("conversation_id", conversation_id)

    out = process_chat(brand_entry=brand, messages=messages, metadata=metadata)
    reply_text = out.get("reply", "")

    append_conversation_record(
        conversation_id=conversation_id,
        brand=brand,
        user_messages=messages,
        assistant_reply=reply_text,
        extra={
            "metadata": metadata,
            "brand_entry": out.get("brand_entry"),
            "brand_effective": out.get("brand_effective"),
            "openai_usage": out.get("openai_usage"),
            "guard_hits": out.get("guard_hits"),
        },
    )

    # ✅ Compat reply/answer
    return (
        jsonify(
            {
                "reply": reply_text,
                "answer": reply_text,
                "brand_effective": out.get("brand_effective"),
                "conversation_id": conversation_id,
            }
        ),
        200,
    )


@app.route("/user/api/history", methods=["GET"])
def user_api_history():
    if USER_HISTORY_TOKEN and not _require_user_token(request):
        return jsonify({"error": "forbidden"}), 403

    conversation_id = (request.args.get("conversation_id") or "").strip()
    user_id = (request.args.get("user_id") or "").strip()

    if not conversation_id and user_id:
        conversation_id = get_user_conversation(user_id) or ""

    if not conversation_id:
        return jsonify({"error": "missing_conversation_id"}), 400

    records = load_conversation(conversation_id)
    return jsonify({"conversation_id": conversation_id, "records": records}), 200


@app.route("/kb/<brand>", methods=["GET"])
def kb_get(brand: str):
    kb = load_kb(brand)
    return jsonify(kb), 200


@app.route("/kb/<brand>", methods=["POST"])
def kb_upsert(brand: str):
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    kb = load_kb(brand)
    items = kb.get("items") or []
    if not isinstance(items, list):
        items = []

    new_items = payload.get("items")
    if isinstance(new_items, list):
        items = new_items

    kb["items"] = items
    ok = save_kb(brand, kb)
    return jsonify({"ok": ok, "brand": brand, "count": len(items)}), 200


@app.route("/qweekle/webhook", methods=["POST"])
def qweekle_webhook():
    path = os.path.join(DATA_DIR, "qweekle_webhook.log")
    try:
        payload = request.get_data(as_text=True) or ""
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.utcnow().isoformat()}Z]\n")
            f.write(payload)
            f.write("\n")
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/admin/api/conversations", methods=["GET"])
def admin_api_conversations():
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    convs = list_conversations()
    return jsonify({"items": convs}), 200


@app.route("/admin/api/conversation/<conversation_id>", methods=["GET"])
def admin_api_conversation_detail(conversation_id: str):
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    records = load_conversation(conversation_id)
    brands = classify_conversation_brands(records)
    return jsonify({"conversation_id": conversation_id, "records": records, "brands": brands}), 200


@app.route("/admin/api/test", methods=["GET"])
def admin_api_test():
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403
    return jsonify({"ok": True, "service": SERVICE_NAME}), 200


@app.route("/admin/api/diag", methods=["GET"])
def admin_api_diag():
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    return jsonify(
        {
            "service": SERVICE_NAME,
            "port": PORT,
            "openai_configured": bool(OPENAI_API_KEY),
            "model": OPENAI_MODEL,
            "kb_dir": KB_DIR,
            "data_dir": DATA_DIR,
            "has_admin_token": bool(ADMIN_DASHBOARD_TOKEN),
            "has_user_history_token": bool(USER_HISTORY_TOKEN),
        }
    ), 200


# ---------------------------------------------------------
# Static pages
# ---------------------------------------------------------


@app.route("/admin", methods=["GET"])
def admin_page():
    return send_from_directory(app.static_folder, "admin.html")


@app.route("/admin/conversations", methods=["GET"])
def admin_conversations_page():
    return send_from_directory(app.static_folder, "admin.html")


@app.route("/<path:path>", methods=["GET"])
def static_proxy(path: str):
    return send_from_directory(app.static_folder, path)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting %s on port %s", SERVICE_NAME, PORT)
    app.run(host="0.0.0.0", port=PORT)
