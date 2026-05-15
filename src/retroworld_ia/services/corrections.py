import re
import sqlite3
import unicodedata
from typing import Any, Dict, List, Tuple

import importlib
import importlib.util

from src.retroworld_ia import config
from src.retroworld_ia.services.logging_store import log_error, now_str


requests = importlib.import_module("requests") if importlib.util.find_spec("requests") else None


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row | None) -> Dict[str, Any] | None:
    return dict(row) if row is not None else None


def normalize_brand_id(value: str) -> str:
    return (value or "").strip().lower()


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return ascii_text.lower()


def tokens_for(value: str) -> set[str]:
    ignored = {
        "avec",
        "chez",
        "dans",
        "des",
        "est",
        "les",
        "pour",
        "que",
        "qui",
        "une",
        "vous",
        "votre",
    }
    return {token for token in re.findall(r"[a-z0-9]{3,}", normalize_text(value)) if token not in ignored}


def init_corrections_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS response_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand_id TEXT NOT NULL,
                trigger_text TEXT NOT NULL,
                corrected_answer TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                source_conversation_id TEXT NOT NULL DEFAULT '',
                is_active INTEGER NOT NULL DEFAULT 1,
                priority INTEGER NOT NULL DEFAULT 50,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                openai_file_id TEXT NOT NULL DEFAULT '',
                openai_vector_store_file_id TEXT NOT NULL DEFAULT '',
                openai_synced_at TEXT NOT NULL DEFAULT '',
                sync_error TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_response_corrections_brand_active
                ON response_corrections(brand_id, is_active, priority);
            """
        )


def validate_correction_payload(payload: Dict[str, Any], default_brand_id: str) -> Tuple[Dict[str, Any], str]:
    brand_id = normalize_brand_id(payload.get("brand_id") or default_brand_id)
    trigger_text = (payload.get("trigger_text") or payload.get("question") or "").strip()
    corrected_answer = (payload.get("corrected_answer") or payload.get("answer") or "").strip()
    notes = (payload.get("notes") or "").strip()
    source_conversation_id = (payload.get("source_conversation_id") or "").strip()
    try:
        priority = int(payload.get("priority", 50))
    except (TypeError, ValueError):
        priority = 50
    priority = max(0, min(priority, 100))

    if not brand_id:
        return {}, "brand_id_required"
    if len(trigger_text) < 4:
        return {}, "trigger_text_required"
    if len(corrected_answer) < 4:
        return {}, "corrected_answer_required"

    return (
        {
            "brand_id": brand_id[:80],
            "trigger_text": trigger_text[:700],
            "corrected_answer": corrected_answer[:2500],
            "notes": notes[:1200],
            "source_conversation_id": source_conversation_id[:120],
            "priority": priority,
            "is_active": 1 if payload.get("is_active", 1) else 0,
        },
        "",
    )


def create_correction(payload: Dict[str, Any], default_brand_id: str = "retroworld") -> Tuple[Dict[str, Any] | None, str]:
    clean, error = validate_correction_payload(payload, default_brand_id)
    if error:
        return None, error

    ts = now_str()
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO response_corrections
            (brand_id, trigger_text, corrected_answer, notes, source_conversation_id, is_active, priority, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean["brand_id"],
                clean["trigger_text"],
                clean["corrected_answer"],
                clean["notes"],
                clean["source_conversation_id"],
                clean["is_active"],
                clean["priority"],
                ts,
                ts,
            ),
        )
        correction_id = int(cursor.lastrowid)

    correction = get_correction(correction_id)
    if correction:
        sync_correction_to_openai(correction)
        correction = get_correction(correction_id)
    return correction, ""


def get_correction(correction_id: int) -> Dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM response_corrections WHERE id = ?", (correction_id,)).fetchone()
    return _row_to_dict(row)


def list_corrections(brand_id: str = "", active_only: bool = False, limit: int = 100) -> List[Dict[str, Any]]:
    clauses = []
    params: List[Any] = []
    bid = normalize_brand_id(brand_id)
    if bid:
        clauses.append("brand_id = ?")
        params.append(bid)
    if active_only:
        clauses.append("is_active = 1")
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(max(1, min(limit, 250)))
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM response_corrections
            {where}
            ORDER BY is_active DESC, priority DESC, updated_at DESC, id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def update_correction(correction_id: int, payload: Dict[str, Any], default_brand_id: str = "retroworld") -> Tuple[Dict[str, Any] | None, str]:
    current = get_correction(correction_id)
    if not current:
        return None, "not_found"
    merged = dict(current)
    merged.update(payload or {})
    clean, error = validate_correction_payload(merged, default_brand_id)
    if error:
        return None, error

    with _connect() as conn:
        conn.execute(
            """
            UPDATE response_corrections
            SET brand_id = ?, trigger_text = ?, corrected_answer = ?, notes = ?,
                source_conversation_id = ?, is_active = ?, priority = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                clean["brand_id"],
                clean["trigger_text"],
                clean["corrected_answer"],
                clean["notes"],
                clean["source_conversation_id"],
                clean["is_active"],
                clean["priority"],
                now_str(),
                correction_id,
            ),
        )

    correction = get_correction(correction_id)
    if correction and correction.get("is_active"):
        sync_correction_to_openai(correction)
        correction = get_correction(correction_id)
    return correction, ""


def find_relevant_corrections(brand_id: str, user_text: str, limit: int | None = None) -> List[Dict[str, Any]]:
    max_items = config.CORRECTION_MEMORY_MAX if limit is None else max(limit, 0)
    if max_items <= 0:
        return []
    user_norm = normalize_text(user_text)
    user_tokens = tokens_for(user_text)
    if not user_norm.strip() or not user_tokens:
        return []

    known_items = list_corrections(brand_id=brand_id, active_only=False, limit=250)
    known_ids = {str(item.get("id")) for item in known_items if item.get("id") is not None}
    candidates = [item for item in known_items if item.get("is_active")]
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for item in candidates:
        trigger_norm = normalize_text(item.get("trigger_text") or "")
        trigger_tokens = tokens_for(item.get("trigger_text") or "")
        if not trigger_tokens:
            continue
        overlap = user_tokens.intersection(trigger_tokens)
        score = len(overlap) * 10 + int(item.get("priority") or 0)
        if trigger_norm and (trigger_norm in user_norm or user_norm in trigger_norm):
            score += 80
        if len(overlap) >= 2 or score >= 90:
            scored.append((score, item))

    scored.sort(key=lambda pair: (pair[0], pair[1].get("updated_at", "")), reverse=True)
    local_items = [item for _, item in scored[:max_items]]
    if len(local_items) >= max_items:
        return local_items

    remote_items = search_openai_corrections(brand_id, user_text, max_items - len(local_items), known_ids)
    return local_items + remote_items


def correction_prompt_block(corrections: List[Dict[str, Any]]) -> str:
    if not corrections:
        return ""
    lines = [
        "Corrections approuvees par l'equipe:",
        "- Applique ces corrections seulement si la demande utilisateur correspond au declencheur.",
        "- Elles completent la base metier; pour une information permanente de FAQ, la FAQ doit aussi etre corrigee dans l'admin.",
        "- Si une correction contredit une disponibilite, un paiement ou un creneau, demande une validation humaine.",
    ]
    for item in corrections:
        trigger = (item.get("trigger_text") or "").replace("\n", " ").strip()
        answer = (item.get("corrected_answer") or "").strip()
        lines.append(f"- Declencheur: {trigger}")
        lines.append(f"  Reponse corrigee a suivre: {answer}")
    return "\n".join(lines)


def _openai_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}


def _correction_document(correction: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "Retroworld IA response correction",
            f"correction_id: {correction.get('id')}",
            f"brand_id: {correction.get('brand_id')}",
            f"active: {bool(correction.get('is_active'))}",
            f"trigger: {correction.get('trigger_text') or ''}",
            "corrected_answer:",
            correction.get("corrected_answer") or "",
            "notes:",
            correction.get("notes") or "",
        ]
    )


def _parse_remote_correction(text: str, attributes: Dict[str, Any], file_id: str, score: float) -> Dict[str, Any] | None:
    trigger_match = re.search(r"trigger:\s*(.*?)\ncorrected_answer:", text or "", flags=re.S)
    answer_match = re.search(r"corrected_answer:\s*(.*?)\nnotes:", text or "", flags=re.S)
    if not trigger_match or not answer_match:
        return None
    trigger = trigger_match.group(1).strip()
    answer = answer_match.group(1).strip()
    if not trigger or not answer:
        return None
    return {
        "id": attributes.get("correction_id") or f"openai:{file_id}",
        "brand_id": attributes.get("brand_id") or "",
        "trigger_text": trigger,
        "corrected_answer": answer,
        "notes": "OpenAI vector store",
        "is_active": 1,
        "priority": int(score * 100),
        "updated_at": "openai",
    }


def search_openai_corrections(brand_id: str, user_text: str, limit: int, known_ids: set[str] | None = None) -> List[Dict[str, Any]]:
    if limit <= 0 or not (config.OPENAI_API_KEY and config.OPENAI_CORRECTIONS_VECTOR_STORE_ID and requests is not None):
        return []
    known_ids = known_ids or set()
    try:
        response = requests.post(
            f"https://api.openai.com/v1/vector_stores/{config.OPENAI_CORRECTIONS_VECTOR_STORE_ID}/search",
            headers={**_openai_headers(), "Content-Type": "application/json"},
            json={
                "query": user_text,
                "max_num_results": max(1, min(limit * 2, 10)),
                "filters": {
                    "type": "and",
                    "filters": [
                        {"type": "eq", "key": "brand_id", "value": normalize_brand_id(brand_id)},
                        {"type": "eq", "key": "active", "value": True},
                    ],
                },
                "ranking_options": {"score_threshold": 0.35},
            },
            timeout=15,
        )
        response.raise_for_status()
        items = []
        for result in (response.json() or {}).get("data", []):
            attributes = result.get("attributes") or {}
            correction_id = str(attributes.get("correction_id") or "")
            if correction_id and correction_id in known_ids:
                continue
            content_text = "\n".join(
                part.get("text", "")
                for part in (result.get("content") or [])
                if isinstance(part, dict) and part.get("type") == "text"
            )
            parsed = _parse_remote_correction(content_text, attributes, result.get("file_id") or "", float(result.get("score") or 0))
            if parsed:
                items.append(parsed)
            if len(items) >= limit:
                break
        return items
    except Exception as err:
        response_obj = getattr(err, "response", None)
        response_text = ""
        if response_obj is not None:
            try:
                response_text = response_obj.text[:2000]
            except Exception:
                response_text = ""
        log_error("OpenAI correction search failed", err, {"brand_id": brand_id, "response_text": response_text})
        return []


def sync_correction_to_openai(correction: Dict[str, Any]) -> None:
    if not (config.OPENAI_API_KEY and config.OPENAI_CORRECTIONS_VECTOR_STORE_ID and requests is not None):
        return
    if correction.get("openai_file_id"):
        return

    correction_id = int(correction.get("id") or 0)
    try:
        file_response = requests.post(
            "https://api.openai.com/v1/files",
            headers=_openai_headers(),
            data={"purpose": "assistants"},
            files={
                "file": (
                    f"retroworld-correction-{correction_id}.txt",
                    _correction_document(correction).encode("utf-8"),
                    "text/plain",
                )
            },
            timeout=30,
        )
        file_response.raise_for_status()
        file_id = (file_response.json() or {}).get("id") or ""
        if not file_id:
            raise RuntimeError("OpenAI file id missing")

        vector_response = requests.post(
            f"https://api.openai.com/v1/vector_stores/{config.OPENAI_CORRECTIONS_VECTOR_STORE_ID}/files",
            headers={**_openai_headers(), "Content-Type": "application/json"},
            json={
                "file_id": file_id,
                "attributes": {
                    "brand_id": correction.get("brand_id") or "",
                    "correction_id": str(correction_id),
                    "active": bool(correction.get("is_active")),
                },
            },
            timeout=30,
        )
        vector_response.raise_for_status()
        vector_file_id = (vector_response.json() or {}).get("id") or ""
        with _connect() as conn:
            conn.execute(
                """
                UPDATE response_corrections
                SET openai_file_id = ?, openai_vector_store_file_id = ?, openai_synced_at = ?, sync_error = ''
                WHERE id = ?
                """,
                (file_id, vector_file_id, now_str(), correction_id),
            )
    except Exception as err:
        response_obj = getattr(err, "response", None)
        response_text = ""
        if response_obj is not None:
            try:
                response_text = response_obj.text[:2000]
            except Exception:
                response_text = ""
        error_text = str(err)[:1000]
        with _connect() as conn:
            conn.execute(
                "UPDATE response_corrections SET sync_error = ?, updated_at = ? WHERE id = ?",
                (error_text, now_str(), correction_id),
            )
        log_error("OpenAI correction sync failed", err, {"correction_id": correction_id, "response_text": response_text})
