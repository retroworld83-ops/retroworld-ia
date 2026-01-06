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
PORT = int(os.getenv("PORT", "10000"))

# Tokens
ADMIN_DASHBOARD_TOKEN = (os.getenv("ADMIN_DASHBOARD_TOKEN") or "").strip()
USER_HISTORY_TOKEN = (os.getenv("USER_HISTORY_TOKEN") or "").strip()

# KB paths
KB_RETROWORLD_PATH = os.getenv("KB_RETROWORLD_PATH", "kb_retroworld.json")
KB_RUNNINGMAN_PATH = os.getenv("KB_RUNNINGMAN_PATH", "kb_runningman.json")

# Logs
CONVERSATIONS_LOG_DIR = os.getenv("CONVERSATIONS_LOG_DIR", "conversations")
QWEEKLE_LOG_DIR = os.getenv("QWEEKLE_LOG_DIR", "qweekle_logs")
QWEEKLE_WEBHOOK_SECRET = (os.getenv("QWEEKLE_WEBHOOK_SECRET") or "").strip()
QWEEKLE_SOURCE_NAME = os.getenv("QWEEKLE_SOURCE_NAME", "qweekle")

# OpenAI
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "600"))

# Behavior
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "16"))
LOG_PREVIEW_MAX_CHARS = int(os.getenv("LOG_PREVIEW_MAX_CHARS", "220"))

# ---------------------------------------------------------
# APP INIT
# ---------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)

app = Flask(__name__, static_folder="static")
CORS(app)

os.makedirs(CONVERSATIONS_LOG_DIR, exist_ok=True)
os.makedirs(QWEEKLE_LOG_DIR, exist_ok=True)

# ---------------------------------------------------------
# SMALL UTILS
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
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------
# USER -> conversation mapping
# ---------------------------------------------------------

USER_INDEX_PATH = os.getenv("USER_INDEX_PATH", "user_index.json")


def get_user_index() -> Dict[str, str]:
    return _safe_read_json(USER_INDEX_PATH, {})


def set_user_conversation(user_id: str, conversation_id: str) -> None:
    idx = get_user_index()
    idx[str(user_id)] = str(conversation_id)
    _safe_write_json(USER_INDEX_PATH, idx)


def get_user_conversation(user_id: str) -> Optional[str]:
    idx = get_user_index()
    return idx.get(str(user_id))


# ---------------------------------------------------------
# KB (cached)
# ---------------------------------------------------------


@dataclass
class _KBCacheEntry:
    path: str
    mtime: float
    data: Dict[str, Any]


_kb_cache: Dict[str, _KBCacheEntry] = {}


def load_kb(brand: str) -> Dict[str, Any]:
    brand = (brand or "retroworld").lower()
    path = KB_RETROWORLD_PATH if brand == "retroworld" else KB_RUNNINGMAN_PATH

    try:
        mtime = os.path.getmtime(path)
    except Exception:
        return {"brand": brand, "items": []}

    cached = _kb_cache.get(brand)
    if cached and cached.path == path and cached.mtime == mtime:
        return cached.data

    data = _safe_read_json(path, {"brand": brand, "items": []})
    if not isinstance(data, dict):
        data = {"brand": brand, "items": []}

    _kb_cache[brand] = _KBCacheEntry(path=path, mtime=mtime, data=data)
    return data


def save_kb(brand: str, kb: Dict[str, Any]) -> bool:
    brand = (brand or "retroworld").lower()
    path = KB_RETROWORLD_PATH if brand == "retroworld" else KB_RUNNINGMAN_PATH
    ok = _safe_write_json(path, kb)
    if ok:
        _kb_cache.pop(brand, None)
    return ok


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# ---------------------------------------------------------
# OPENAI CALL
# ---------------------------------------------------------

_openai_client = None
_openai_init_error = None

try:
    from openai import OpenAI  # type: ignore

    if OPENAI_API_KEY:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        _openai_init_error = "OPENAI_API_KEY vide ou non défini"
except Exception as e:
    _openai_client = None
    _openai_init_error = f"Import/initialisation OpenAI impossible: {type(e).__name__}: {e}"
    logger.exception("OpenAI init failed")


def call_openai_chat(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (assistant_text, usage_dict).
    """
    if not _openai_client:
        reason = _openai_init_error or "Client OpenAI non initialisé"
        return (f"Le service IA n’est pas configuré. Détail: {reason}", {"skipped": True, "reason": reason})

    resp = _openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
    )
    text = (resp.choices[0].message.content or "").strip()

    usage = {}
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


# ---------------------------------------------------------
# BRAND DETECTION
# ---------------------------------------------------------


def detect_brand_from_text(text: str) -> str:
    t = (text or "").lower()
    if "runningman" in t or "running man" in t or "action game" in t:
        return "runningman"
    return "retroworld"


def _is_reservation_intent(text: str) -> bool:
    t = (text or "").lower()
    keywords = ["réserver", "reservation", "réservation", "disponible", "disponibilité", "horaire", "créneau"]
    return any(k in t for k in keywords)


# ---------------------------------------------------------
# CONVERSATION LOGGING
# ---------------------------------------------------------


def _conversation_path(conversation_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", conversation_id)
    return os.path.join(CONVERSATIONS_LOG_DIR, f"{safe}.jsonl")


def append_conversation_log(
    conversation_id: str,
    brand: str,
    channel: str,
    user_messages: List[Dict[str, Any]],
    assistant_reply: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = _conversation_path(conversation_id)
    rec = {
        "timestamp": time.time(),
        "datetime": datetime.utcnow().isoformat() + "Z",
        "conversation_id": conversation_id,
        "brand": brand,
        "channel": channel,
        "user_messages": user_messages,
        "assistant_reply": assistant_reply,
        "extra": extra or {},
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_conversation_records(conversation_id: str) -> List[Dict[str, Any]]:
    path = _conversation_path(conversation_id)
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def reconstruct_history_from_logs(conversation_id: str) -> List[Dict[str, str]]:
    records = load_conversation_records(conversation_id)
    history: List[Dict[str, str]] = []
    for rec in records:
        rec_msgs = rec.get("user_messages") or []
        for m in rec_msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and content is not None:
                history.append({"role": role, "content": str(content)})
        assistant_text = rec.get("assistant_reply")
        if assistant_text:
            history.append({"role": "assistant", "content": str(assistant_text)})
    return history


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
        b = next(iter(brands_seen))
        return {"brand_final": b, "brands_seen": [b]}
    brand_final = last_effective or "mixed"
    if brand_final not in ("runningman", "retroworld"):
        brand_final = "mixed"
    return {"brand_final": brand_final, "brands_seen": sorted(list(brands_seen))}


# ---------------------------------------------------------
# CORE CHAT ENGINE
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
        brand_effective = "runningman" if brand_entry == "runningman" else "retroworld"

    kb = load_kb(brand_effective)
    kb_items = kb.get("items") if isinstance(kb, dict) else []
    if not isinstance(kb_items, list):
        kb_items = []

    system = (
        "Vous êtes un assistant conversationnel professionnel.\n"
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
                    system += f"- {s}\n"

    openai_messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

    clipped = []
    for m in messages[-MAX_HISTORY_MESSAGES:]:
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

    guard_hits = []
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


# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------


@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": SERVICE_NAME, "status": "ok", "time_utc": datetime.utcnow().isoformat() + "Z"}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/chat/<brand>", methods=["POST"])
def chat_route(brand: str):
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        return jsonify({"error": "invalid_messages"}), 400

    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    conversation_id = str(payload.get("conversation_id") or "").strip()
    user_id = str(payload.get("user_id") or "").strip()

    if not conversation_id and user_id:
        conversation_id = get_user_conversation(user_id) or ""

    if not conversation_id:
        conversation_id = f"conv_{int(time.time())}_{abs(hash(str(user_id) or str(request.remote_addr) or 'x'))%100000}"

    if user_id:
        set_user_conversation(user_id, conversation_id)

    out = process_chat(brand_entry=brand, messages=messages, metadata=metadata)
    reply_text = out.get("reply") or ""

    channel = str(metadata.get("source") or "web")
    append_conversation_log(
        conversation_id=conversation_id,
        brand=out.get("brand_effective") or (brand or "auto"),
        channel=channel,
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

    return jsonify({"reply": reply_text, "brand_effective": out.get("brand_effective"), "conversation_id": conversation_id}), 200


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

    history = reconstruct_history_from_logs(conversation_id)
    return jsonify({"conversation_id": conversation_id, "messages": history}), 200


@app.route("/kb/upsert/<brand>", methods=["POST"])
def kb_upsert(brand: str):
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400
    if not isinstance(body, dict):
        return jsonify({"error": "invalid_payload"}), 400

    kb = load_kb(brand)
    items = kb.get("items")
    if not isinstance(items, list):
        items = []
        kb["items"] = items

    new_items = body.get("items")
    if isinstance(new_items, list):
        kb["items"] = new_items
    else:
        it = body.get("item")
        if it is not None:
            kb["items"].append(it)

    ok = save_kb(brand, kb)
    return jsonify({"status": "ok" if ok else "error", "brand": brand}), (200 if ok else 500)


@app.route("/webhooks/qweekle", methods=["POST"])
def qweekle_webhook():
    if QWEEKLE_WEBHOOK_SECRET:
        incoming_secret = request.headers.get("X-Qweekle-Secret") or ""
        if incoming_secret != QWEEKLE_WEBHOOK_SECRET:
            logger.warning("Qweekle webhook rejected (invalid secret)")
            return jsonify({"error": "forbidden"}), 403

    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    event_type = payload.get("event_type") or payload.get("type") or "unknown"
    logger.info("Webhook Qweekle received: %s", event_type)

    fname = f"{event_type or 'unknown'}.jsonl"
    path = os.path.join(QWEEKLE_LOG_DIR, fname)
    record = {"timestamp": time.time(), "event_type": event_type, "payload": payload, "source": QWEEKLE_SOURCE_NAME}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return jsonify({"status": "ok", "event_type": event_type}), 200


# ---------------------------------------------------------
# ADMIN API
# ---------------------------------------------------------

@app.route("/admin/api/conversations", methods=["GET"])
def admin_api_conversations():
    if not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    convs: List[Dict[str, Any]] = []
    if not os.path.isdir(CONVERSATIONS_LOG_DIR):
        return jsonify(convs), 200

    for fname in os.listdir(CONVERSATIONS_LOG_DIR):
        if not fname.endswith(".jsonl"):
            continue

        conversation_id = fname[:-5]
        records = load_conversation_records(conversation_id)
        if not records:
            continue

        last = records[-1]
        ts = float(last.get("timestamp") or 0.0)
        channel = last.get("channel") or "web"
        extra = last.get("extra") or {}
        meta = extra.get("metadata") or {}
        source = extra.get("source") or meta.get("source") or "unknown"

        brand_info = classify_conversation_brands(records)
        brand_final = brand_info.get("brand_final")

        preview = ""
        for rec in reversed(records):
            umsgs = rec.get("user_messages") or []
            for m in reversed(umsgs):
                if isinstance(m, dict) and m.get("role") == "user":
                    preview = str(m.get("content") or "")
                    break
            if preview:
                break

        preview = _norm(preview)
        if len(preview) > LOG_PREVIEW_MAX_CHARS:
            preview = preview[:LOG_PREVIEW_MAX_CHARS].rstrip() + "…"

        convs.append({
            "id": conversation_id,
            "timestamp": ts,
            "datetime": last.get("datetime"),
            "brand": brand_final,
            "channel": channel,
            "source": source,
            "preview": preview,
        })

    convs.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return jsonify(convs), 200


@app.route("/admin/api/conversation/<conversation_id>", methods=["GET"])
def admin_api_conversation_detail(conversation_id: str):
    if not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    records = load_conversation_records(conversation_id)
    if not records:
        return jsonify({"error": "not_found"}), 404

    brand_info = classify_conversation_brands(records)
    history = reconstruct_history_from_logs(conversation_id)

    return jsonify({
        "conversation_id": conversation_id,
        "brand_final": brand_info.get("brand_final"),
        "brands_seen": brand_info.get("brands_seen"),
        "messages": history,
        "records": records,
    }), 200


@app.route("/admin/api/test", methods=["POST"])
def admin_api_test():
    if not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400
    if not isinstance(body, dict):
        return jsonify({"error": "invalid_payload"}), 400

    brand = str(body.get("brand") or "auto").lower()

    questions = body.get("questions")
    if not isinstance(questions, list):
        # compat: accept payload multiline
        payload = body.get("payload")
        questions = []
        if isinstance(payload, str):
            for line in payload.splitlines():
                ln = line.strip()
                if ln:
                    questions.append(ln)

    if not isinstance(questions, list) or not questions:
        return jsonify({"error": "no_questions"}), 400

    # dedupe
    seen = set()
    clean = []
    for q in questions:
        if isinstance(q, str):
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                clean.append(q)

    results = []
    for q in clean:
        msgs = [{"role": "user", "content": q}]
        out = process_chat(brand_entry=brand, messages=msgs, metadata={"source": "admin_test"})
        results.append({"q": q, "brand_effective": out.get("brand_effective"), "answer": out.get("reply")})

    return jsonify({"status": "ok", "count": len(results), "results": results}), 200


@app.route("/admin/api/diag", methods=["GET"])
def admin_api_diag():
    if not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    # OpenAI (sans exposer la clé)
    openai_key_present = bool(OPENAI_API_KEY)
    openai_key_len = len(OPENAI_API_KEY) if OPENAI_API_KEY else 0
    openai_client_ready = bool(_openai_client)
    openai_init_error = _openai_init_error

    # KB
    kb_rw_exists = os.path.exists(KB_RETROWORLD_PATH)
    kb_rm_exists = os.path.exists(KB_RUNNINGMAN_PATH)

    kb_rw_ok = False
    kb_rm_ok = False
    kb_rw_count = 0
    kb_rm_count = 0

    try:
        kb_rw = load_kb("retroworld")
        items = kb_rw.get("items") if isinstance(kb_rw, dict) else []
        kb_rw_count = len(items) if isinstance(items, list) else 0
        kb_rw_ok = True
    except Exception:
        kb_rw_ok = False

    try:
        kb_rm = load_kb("runningman")
        items = kb_rm.get("items") if isinstance(kb_rm, dict) else []
        kb_rm_count = len(items) if isinstance(items, list) else 0
        kb_rm_ok = True
    except Exception:
        kb_rm_ok = False

    # Logs
    conv_dir_exists = os.path.isdir(CONVERSATIONS_LOG_DIR)
    conv_files = []
    try:
        if conv_dir_exists:
            conv_files = [f for f in os.listdir(CONVERSATIONS_LOG_DIR) if f.endswith(".jsonl")]
    except Exception:
        conv_files = []

    qweekle_dir_exists = os.path.isdir(QWEEKLE_LOG_DIR)
    qweekle_files = []
    try:
        if qweekle_dir_exists:
            qweekle_files = [f for f in os.listdir(QWEEKLE_LOG_DIR) if f.endswith(".jsonl")]
    except Exception:
        qweekle_files = []

    return jsonify({
        "service": SERVICE_NAME,
        "time_utc": datetime.utcnow().isoformat() + "Z",

        "openai": {
            "key_present": openai_key_present,
            "key_length": openai_key_len,
            "client_ready": openai_client_ready,
            "init_error": openai_init_error,
            "model": OPENAI_MODEL,
            "temperature": OPENAI_TEMPERATURE,
            "max_tokens": OPENAI_MAX_TOKENS,
        },

        "kb": {
            "retroworld": {
                "path": KB_RETROWORLD_PATH,
                "exists": kb_rw_exists,
                "load_ok": kb_rw_ok,
                "items_count": kb_rw_count,
            },
            "runningman": {
                "path": KB_RUNNINGMAN_PATH,
                "exists": kb_rm_exists,
                "load_ok": kb_rm_ok,
                "items_count": kb_rm_count,
            },
        },

        "logs": {
            "conversations_dir": CONVERSATIONS_LOG_DIR,
            "conversations_dir_exists": conv_dir_exists,
            "conversations_files_count": len(conv_files),
            "qweekle_dir": QWEEKLE_LOG_DIR,
            "qweekle_dir_exists": qweekle_dir_exists,
            "qweekle_files_count": len(qweekle_files),
            "qweekle_secret_required": bool(QWEEKLE_WEBHOOK_SECRET),
        },

        "tokens": {
            "admin_token_set": bool(ADMIN_DASHBOARD_TOKEN),
            "user_history_token_set": bool(USER_HISTORY_TOKEN),
        }
    }), 200


# ---------------------------------------------------------
# ADMIN UI (HTML)
# ---------------------------------------------------------

@app.route("/admin", methods=["GET"])
def admin_page():
    if not _require_admin_token(request):
        return "Forbidden", 403

    static_dir = os.path.join(app.root_path, "static")
    admin_path = os.path.join(static_dir, "admin.html")
    if os.path.exists(admin_path):
        return send_from_directory(static_dir, "admin.html")
    return "admin.html missing in /static", 404


# Back-compat: previously /admin/conversations existed
@app.route("/admin/conversations", methods=["GET"])
def admin_conversations_page():
    return admin_page()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
