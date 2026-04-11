import json
import sqlite3
import uuid
from typing import Any, Dict, List

from src.retroworld_ia import config
from src.retroworld_ia.services.knowledge import BRAND_ID_DEFAULT, intent_tags, normalize_brand
from src.retroworld_ia.services.logging_store import log_error, now_str

import importlib
import importlib.util

requests = importlib.import_module("requests") if importlib.util.find_spec("requests") else None


FLAG_PATTERNS = {
    "devis": r"\b(devis|privatis|entreprise|ce\b|comit[eé]\s*d['’]?entreprise|team\s*building|groupe)\b",
    "reservation": r"\b(réserv|reservation|réservation|dispo|créneau|anniversaire|goûter|acompte)\b",
    "reclamation": r"\b(rembourse|annul|plainte|probl[eè]me|litige|panne)\b",
    "croise": r"\b(retroworld|runningman|enigmaniac)\b.*\b(retroworld|runningman|enigmaniac)\b",
    "promesse_resa": r"\b(réservé|confirmé|je vous bloque|c['’]?est réservé|bloqué)\b",
    "a_relire": r"\b(peut[- ]?etre|probablement|je pense|a priori|il me semble)\b",
}


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                brand_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                flags_json TEXT NOT NULL DEFAULT '[]',
                meta_json TEXT NOT NULL DEFAULT '{}',
                last_message_preview TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                extra_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id, id);

            CREATE TABLE IF NOT EXISTS admin_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                brand_id TEXT NOT NULL,
                lead_type TEXT NOT NULL,
                score INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'new',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_leads_conversation ON leads(conversation_id);
            """
        )


def compute_flags(conv: Dict[str, Any]) -> List[str]:
    import re
    messages = conv.get("messages") or []
    blob = "\n".join((message.get("content") or "") for message in messages).lower()
    flags = []
    for name, pattern in FLAG_PATTERNS.items():
        if re.search(pattern, blob, flags=re.IGNORECASE | re.DOTALL):
            flags.append(name)
    if any(flag in flags for flag in ("devis", "reclamation")):
        flags.append("a_valider")
    return sorted(set(flags))


def score_lead(conv: Dict[str, Any]) -> Dict[str, Any]:
    flags = set(compute_flags(conv))
    messages = conv.get("messages") or []
    text = "\n".join((m.get("content") or "") for m in messages)
    intents = intent_tags(text)
    score = 0
    lead_type = "general"
    if "devis" in flags:
        score += 80
        lead_type = "devis"
    if "reservation" in flags or "reservation" in intents:
        score += 60
        lead_type = "reservation"
    if "anniversaire" in intents:
        score += 50
        lead_type = "anniversaire"
    if "reclamation" in flags:
        score += 30
        lead_type = "reclamation"
    score += min(len(messages) * 3, 18)
    if score >= 90:
        level = "hot"
    elif score >= 50:
        level = "warm"
    else:
        level = "cold"
    return {"score": score, "lead_type": lead_type, "temperature": level, "intents": intents}


def sync_lead_for_conversation(conv: Dict[str, Any]) -> None:
    lead = score_lead(conv)
    if lead["score"] < 25:
        return
    brand_id = normalize_brand((conv.get("meta") or {}).get("brand_id") or BRAND_ID_DEFAULT)
    payload = {"intents": lead["intents"], "temperature": lead["temperature"]}
    should_notify = False
    with get_db() as conn:
        row = conn.execute("SELECT id FROM leads WHERE conversation_id = ?", (conv["id"],)).fetchone()
        if row:
            conn.execute(
                """
                UPDATE leads
                SET brand_id = ?, lead_type = ?, score = ?, updated_at = ?, payload_json = ?
                WHERE conversation_id = ?
                """,
                (brand_id, lead["lead_type"], lead["score"], now_str(), json.dumps(payload, ensure_ascii=False), conv["id"]),
            )
        else:
            should_notify = True
            conn.execute(
                """
                INSERT INTO leads (conversation_id, brand_id, lead_type, score, status, created_at, updated_at, payload_json)
                VALUES (?, ?, ?, ?, 'new', ?, ?, ?)
                """,
                (conv["id"], brand_id, lead["lead_type"], lead["score"], now_str(), now_str(), json.dumps(payload, ensure_ascii=False)),
            )
    if config.LEAD_WEBHOOK_URL and should_notify and lead["score"] >= 90:
        notify_lead_webhook(conv, brand_id, lead)


def notify_lead_webhook(conv: Dict[str, Any], brand_id: str, lead: Dict[str, Any]) -> None:
    if not config.LEAD_WEBHOOK_URL or requests is None:
        return
    try:
        requests.post(
            config.LEAD_WEBHOOK_URL,
            json={
                "conversation_id": conv["id"],
                "brand_id": brand_id,
                "lead_type": lead["lead_type"],
                "score": lead["score"],
                "temperature": lead["temperature"],
                "intents": lead["intents"],
                "created_at": now_str(),
                "preview": ((conv.get("messages") or [])[-1].get("content", "") if conv.get("messages") else "")[:240],
            },
            timeout=10,
        ).raise_for_status()
    except Exception as err:
        log_error("Lead webhook failed", err, {"conversation_id": conv["id"], "brand_id": brand_id})


def migrate_legacy_json_conversations() -> None:
    try:
        with get_db() as conn:
            imported = conn.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()["n"]
            if imported:
                return
            for path in sorted(config.CONV_DIR.glob("*.json")):
                try:
                    conv = json.loads(path.read_text("utf-8"))
                except Exception:
                    continue
                conv_id = str(conv.get("id") or path.stem)
                meta = conv.get("meta") or {}
                brand_id = normalize_brand(meta.get("brand_id") or BRAND_ID_DEFAULT)
                created = conv.get("created") or now_str()
                messages = conv.get("messages") or []
                flags = compute_flags({"messages": messages})
                preview = (messages[-1].get("content", "") if messages else "")[:240]
                conn.execute(
                    """
                    INSERT OR IGNORE INTO conversations
                    (id, brand_id, created_at, updated_at, flags_json, meta_json, last_message_preview)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (conv_id, brand_id, created, created, json.dumps(flags), json.dumps(meta), preview),
                )
                for message in messages:
                    conn.execute(
                        "INSERT INTO messages (conversation_id, ts, role, content, extra_json) VALUES (?, ?, ?, ?, ?)",
                        (conv_id, message.get("ts") or created, message.get("role") or "assistant", message.get("content") or "", json.dumps(message.get("extra") or {})),
                    )
    except Exception as err:
        log_error("Legacy conversation migration failed", err)


def new_conv_id(prefix: str = "rw") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def fetch_conversation(conv_id: str) -> Dict[str, Any]:
    with get_db() as conn:
        conv_row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        if not conv_row:
            return {"id": conv_id, "created": now_str(), "messages": [], "meta": {}}
        msg_rows = conn.execute("SELECT ts, role, content, extra_json FROM messages WHERE conversation_id = ? ORDER BY id ASC", (conv_id,)).fetchall()
    messages = []
    for row in msg_rows:
        try:
            extra = json.loads(row["extra_json"] or "{}")
        except Exception:
            extra = {}
        messages.append({"ts": row["ts"], "role": row["role"], "content": row["content"], "extra": extra})
    try:
        meta = json.loads(conv_row["meta_json"] or "{}")
    except Exception:
        meta = {}
    return {
        "id": conv_row["id"],
        "created": conv_row["created_at"],
        "updated": conv_row["updated_at"],
        "messages": messages,
        "meta": meta,
        "flags": json.loads(conv_row["flags_json"] or "[]"),
    }


def append_message(conversation: Dict[str, Any], role: str, content: str, extra: Dict[str, Any] | None = None) -> None:
    conversation.setdefault("messages", [])
    conversation["messages"].append({"ts": now_str(), "role": role, "content": content, "extra": extra or {}})


def upsert_conversation(conversation: Dict[str, Any]) -> None:
    flags = compute_flags(conversation)
    meta = dict(conversation.get("meta") or {})
    messages = conversation.get("messages") or []
    preview = (messages[-1].get("content") if messages else "") or ""
    brand_id = normalize_brand(meta.get("brand_id") or BRAND_ID_DEFAULT)
    created = conversation.get("created") or now_str()
    updated = (messages[-1].get("ts") if messages else None) or now_str()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO conversations (id, brand_id, created_at, updated_at, flags_json, meta_json, last_message_preview)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                brand_id = excluded.brand_id,
                updated_at = excluded.updated_at,
                flags_json = excluded.flags_json,
                meta_json = excluded.meta_json,
                last_message_preview = excluded.last_message_preview
            """,
            (conversation["id"], brand_id, created, updated, json.dumps(flags, ensure_ascii=False), json.dumps(meta, ensure_ascii=False), preview[:240]),
        )
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation["id"],))
        for message in messages:
            conn.execute(
                "INSERT INTO messages (conversation_id, ts, role, content, extra_json) VALUES (?, ?, ?, ?, ?)",
                (conversation["id"], message.get("ts") or now_str(), message.get("role") or "assistant", message.get("content") or "", json.dumps(message.get("extra") or {}, ensure_ascii=False)),
            )
    sync_lead_for_conversation(conversation)


def create_or_load_conversation(conv_id: str, brand_id: str) -> Dict[str, Any]:
    conversation = fetch_conversation(conv_id)
    conversation.setdefault("meta", {})
    conversation["meta"]["brand_id"] = brand_id
    conversation["meta"]["last_seen"] = now_str()
    return conversation


def conversation_message_count(conv_id: str) -> int:
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?", (conv_id,)).fetchone()
        return int(row["n"])


def list_conversations() -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute("SELECT id, brand_id, created_at, updated_at, flags_json, meta_json, last_message_preview FROM conversations ORDER BY updated_at DESC, id DESC").fetchall()
    items = []
    for row in rows:
        try:
            meta = json.loads(row["meta_json"] or "{}")
        except Exception:
            meta = {}
        items.append(
            {
                "id": row["id"],
                "brand_id": row["brand_id"],
                "created": row["created_at"],
                "last": row["updated_at"],
                "count": conversation_message_count(row["id"]),
                "flags": json.loads(row["flags_json"] or "[]"),
                "preview": row["last_message_preview"],
                "user_id": meta.get("user_id", ""),
            }
        )
    return items


def list_leads(limit: int = 100) -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM leads ORDER BY score DESC, updated_at DESC LIMIT ?", (limit,)).fetchall()
    items = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except Exception:
            payload = {}
        items.append(dict(row) | {"payload": payload})
    return items


def list_admin_users() -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, username, created_at, updated_at, is_active FROM admin_users ORDER BY username ASC"
        ).fetchall()
    return [dict(row) for row in rows]


def analytics_snapshot() -> Dict[str, Any]:
    conversations = list_conversations()
    leads = list_leads(100)
    by_brand: Dict[str, int] = {}
    by_flag: Dict[str, int] = {}
    for conv in conversations:
        by_brand[conv["brand_id"]] = by_brand.get(conv["brand_id"], 0) + 1
        for flag in conv.get("flags", []):
            by_flag[flag] = by_flag.get(flag, 0) + 1
    return {
        "conversations_total": len(conversations),
        "leads_total": len(leads),
        "hot_leads": len([lead for lead in leads if lead.get("score", 0) >= 90]),
        "brands": by_brand,
        "flags": by_flag,
        "recent_leads": leads[:12],
    }
