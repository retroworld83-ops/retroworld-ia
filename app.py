"""
Flask application for the Retroworld / Runningman conversation assistant.

- Multi-brand KB (kb_retroworld.json / kb_runningman.json)
- Conversation logging (.jsonl)
- Chat endpoint: /chat/<brand>
- User history endpoint (optional token): /user/api/history
- KB upsert endpoint: /kb/upsert/<brand> (optional admin token)
- Qweekle webhook logger: /webhooks/qweekle
- Admin APIs:
    /admin/api/conversations
    /admin/api/conversation/<conversation_id>
    /admin/api/test
  Admin UI:
    /admin (serves static/admin.html)
    /admin/conversations (back-compat, now also serves static/admin.html)
  Extra admin aliases:
    /admin/conversations/list
    /admin/conversation/<conversation_id>
    /admin/test
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory, abort, make_response
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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change if you want
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
# USER -> conversation mapping (optional simple index)
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
# KB
# ---------------------------------------------------------


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        # if KB missing, return empty
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
        try:
            _kb_cache.pop(brand, None)
        except Exception:
            pass
    return ok


# ---------------------------------------------------------
# OPENAI CALL
# ---------------------------------------------------------

# OpenAI SDK is optional; keep code robust on hosts where it's not installed
_openai_client = None
try:
    from openai import OpenAI  # type: ignore

    if OPENAI_API_KEY:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    _openai_client = None


def call_openai_chat(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (assistant_text, usage_dict).
    If OpenAI isn't configured, returns a safe fallback.
    """
    if not _openai_client:
        return (
            "Le service IA n’est pas configuré (clé OpenAI absente).",
            {"skipped": True},
        )

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
# BRAND + GUARDS
# ---------------------------------------------------------


def detect_brand_from_text(text: str) -> str:
    t = (text or "").lower()
    if "runningman" in t or "running man" in t or "action game" in t:
        return "runningman"
    return "retroworld"


def _get_nested(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _retroworld_defaults() -> Dict[str, Any]:
    # Minimal. Your KB should contain the real details.
    return {
        "brand": "retroworld",
        "tone": "pro",
        "signature": "L'équipe Retroworld",
    }


def _runningman_defaults() -> Dict[str, Any]:
    return {"brand": "runningman", "tone": "pro"}


def _merge_defaults(brand: str, kb: Dict[str, Any]) -> Dict[str, Any]:
    base = _retroworld_defaults() if brand == "retroworld" else _runningman_defaults()
    merged = dict(base)
    merged.update(kb or {})
    return merged


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


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
    """
    brand_entry: 'retroworld' | 'runningman' | 'auto'
    messages: list of {role, content}
    metadata: optional dict
    Returns dict with reply, brand_effective, openai_usage, etc.
    """
    metadata = metadata or {}
    brand_entry = (brand_entry or "auto").lower()

    # Effective brand
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
    kb = _merge_defaults(brand_effective, kb)

    # Build prompt
    system = (
        "Vous êtes un assistant conversationnel professionnel.\n"
        "Répondez de façon claire, précise, utile, sans inventer.\n"
        "Si l’utilisateur veut réserver: répondre 'disponible' puis proposer la suite.\n"
        "Respectez la marque et ses règles.\n"
    )

    # Optional: push KB items into system context
    kb_items = kb.get("items") or []
    if isinstance(kb_items, list) and kb_items:
        system += "\nConnaissances (KB) :\n"
        for it in kb_items[:120]:
            if isinstance(it, str):
                system += f"- {it}\n"
            elif isinstance(it, dict):
                title = it.get("title") or it.get("name") or ""
                content = it.get("content") or it.get("text") or it.get("value") or ""
                s = _norm(f"{title} {content}")
                if s:
                    system += f"- {s}\n"

    openai_messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

    # Keep only last N messages to avoid prompt explosion
    clipped = []
    for m in messages[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and content is not None:
            clipped.append({"role": role, "content": str(content)})
    openai_messages.extend(clipped)

    # Reservation rule (simple guard)
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
    return bool(tok) and tok == ADMIN_DASHBOARD_TOKEN


def _require_user_token(req) -> bool:
    tok = (req.args.get("token") or "").strip()
    if not tok:
        tok = (req.headers.get("X-User-Token") or "").strip()
    return bool(tok) and tok == USER_HISTORY_TOKEN


# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": SERVICE_NAME,
            "status": "ok",
            "time_utc": datetime.utcnow().isoformat() + "Z",
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/chat/<brand>", methods=["POST"])
def chat_route(brand: str):
    """
    Input JSON:
      {
        "conversation_id": "optional",
        "user_id": "optional",
        "messages": [{"role":"user","content":"..."}],
        "metadata": {...}
      }
    Output:
      { reply, brand_effective, conversation_id }
    """
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

    # If user_id is provided and conversation_id is not, try to re-use mapping
    if not conversation_id and user_id:
        conversation_id = get_user_conversation(user_id) or ""

    if not conversation_id:
        conversation_id = f"conv_{int(time.time())}_{abs(hash(str(user_id) or str(request.remote_addr) or 'x'))%100000}"

    if user_id:
        set_user_conversation(user_id, conversation_id)

    out = process_chat(brand_entry=brand, messages=messages, metadata=metadata)
    reply_text = out.get("reply") or ""

    # Log
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

    return jsonify(
        {
            "reply": reply_text,
            "brand_effective": out.get("brand_effective"),
            "conversation_id": conversation_id,
        }
    ), 200


@app.route("/user/api/history", methods=["GET"])
def user_api_history():
    """
    Optional: requires USER_HISTORY_TOKEN if set.
    Query:
      ?conversation_id=...
      or ?user_id=...
    """
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
    """
    Admin endpoint to add/replace KB items.
    If ADMIN_DASHBOARD_TOKEN is set, it requires it.
    """
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
        # Replace strategy by default
        kb["items"] = new_items
    else:
        # If single item, append
        it = body.get("item")
        if it is not None:
            kb["items"].append(it)

    ok = save_kb(brand, kb)
    return jsonify({"status": "ok" if ok else "error", "brand": brand}), (200 if ok else 500)


# ---------------- QWEEKLE WEBHOOK ----------------

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


# ---------------- ADMIN API ----------------

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

        convs.append(
            {
                "id": conversation_id,
                "timestamp": ts,
                "datetime": last.get("datetime"),
                "brand": brand_final,
                "channel": channel,
                "source": source,
                "preview": preview,
            }
        )

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

    # Compat: keep original 'records', add nicer 'messages' for UI
    return jsonify(
        {
            "conversation_id": conversation_id,
            "brand_final": brand_info.get("brand_final"),
            "brands_seen": brand_info.get("brands_seen"),
            "messages": history,
            "records": records,
        }
    ), 200


@app.route("/admin/api/test", methods=["POST"])
def admin_api_test():
    """
    Run a batch test:
      JSON input:
        { "brand":"auto|retroworld|runningman", "payload": ["q1","q2"] }
        or { "brand":"...", "payload": "one question per line" }
      Output: results
    """
    if not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403
    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400
    if not isinstance(body, dict):
        return jsonify({"error": "invalid_payload"}), 400

    brand = str(body.get("brand") or "auto").lower()
    payload = body.get("payload")

    questions: List[str] = []

    if isinstance(payload, list):
        for q in payload:
            if isinstance(q, str) and q.strip():
                questions.append(q.strip())
    elif isinstance(payload, str):
        for line in payload.splitlines():
            ln = line.strip()
            if ln:
                questions.append(ln)
    else:
        # also accept body.questions
        q2 = body.get("questions")
        if isinstance(q2, list):
            for q in q2:
                if isinstance(q, str) and q.strip():
                    questions.append(q.strip())

    # Deduplicate keep order
    seen = set()
    qs: List[str] = []
    for q in questions:
        if q in seen:
            continue
        seen.add(q)
        qs.append(q)

    results = []
    for q in qs:
        msgs = [{"role": "user", "content": q}]
        out = process_chat(brand_entry=brand, messages=msgs, metadata={"source": "admin_api_test"})
        results.append({"q": q, "brand_effective": out.get("brand_effective"), "answer": out.get("reply")})

    return jsonify({"status": "ok", "count": len(results), "results": results}), 200


# --------- ADMIN UI helper endpoints (aliases) ---------

@app.route("/admin/conversations/list", methods=["GET"])
def admin_conversations_list():
    """Alias returning the same as /admin/api/conversations (JSON)."""
    return admin_api_conversations()


@app.route("/admin/conversation/<conversation_id>", methods=["GET"])
def admin_conversation_flat(conversation_id: str):
    """Alias returning a UI-friendly conversation payload."""
    if not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403
    records = load_conversation_records(conversation_id)
    if not records:
        return jsonify({"error": "not_found"}), 404
    brand_info = classify_conversation_brands(records)
    history = reconstruct_history_from_logs(conversation_id)
    return jsonify(
        {
            "id": conversation_id,
            "brand": brand_info.get("brand_final"),
            "brands_seen": brand_info.get("brands_seen"),
            "messages": history,
        }
    ), 200


@app.route("/admin/test", methods=["POST"])
def admin_test_batch():
    """Alias for batch testing (expects {brand, questions}). Returns answers."""
    if not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403
    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400
    brand = str(body.get("brand") or "auto").lower()
    questions = body.get("questions") or []
    if not isinstance(questions, list):
        return jsonify({"error": "invalid_questions"}), 400

    results = []
    for q in questions:
        if not isinstance(q, str) or not q.strip():
            continue
        messages = [{"role": "user", "content": q.strip()}]
        out = process_chat(brand_entry=brand, messages=messages, metadata={"source": "admin_test"})
        results.append(
            {"q": q.strip(), "brand_effective": out.get("brand_effective"), "answer": out.get("reply")}
        )

    return jsonify({"status": "ok", "count": len(results), "results": results}), 200


# ---------------- ADMIN UI (HTML) ----------------

@app.route("/admin", methods=["GET"])
def admin_page():
    # Main admin entrypoint (serves static/admin.html).
    if not _require_admin_token(request):
        return "Forbidden", 403
    static_dir = os.path.join(app.root_path, "static")
    admin_path = os.path.join(static_dir, "admin.html")
    if os.path.exists(admin_path):
        return send_from_directory(static_dir, "admin.html")
    return "admin.html missing in /static", 404


@app.route("/admin/conversations", methods=["GET"])
def admin_conversations_page():
    # Back-compat route that now serves the real admin UI
    if not _require_admin_token(request):
        return "Forbidden", 403

    static_dir = os.path.join(app.root_path, "static")
    admin_path = os.path.join(static_dir, "admin.html")

    if os.path.exists(admin_path):
        return send_from_directory(static_dir, "admin.html")

    # Fallback (if admin.html is not present)
    return """<!doctype html>
<html lang="fr">
<meta charset="utf-8">
<title>Admin IA</title>
<body style="font-family:system-ui;background:#0b1220;color:#e5e7eb;padding:24px;">
  <h2>Admin IA</h2>
  <p>La page <code>static/admin.html</code> est absente.</p>
  <p>API disponibles :</p>
  <ul>
    <li><code>/admin/api/conversations</code></li>
    <li><code>/admin/api/conversation/&lt;conversation_id&gt;</code></li>
    <li><code>/admin/api/test</code></li>
  </ul>
</body>
</html>"""


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
