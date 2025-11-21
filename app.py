"""
Flask application for the Retroworld conversational assistant.

This file exposes REST endpoints for the chat experience, a simple administrative
dashboard for reviewing past conversations, and webhook handlers.  It loads
knowledge base (KB) JSON files on demand, routes chat requests to OpenAI’s
Chat API, persists logs to disk, and rebuilds context when the front‑end
provides only the latest user message.  The admin interface has been
re‑designed to offer a more professional look and feel: search, filtering,
and conversation detail views are now easier to use and visually cohesive.

Most of the business logic (detecting brands, building prompts, logging
conversations) is unchanged from the previous version – only the UI and
structural layout have been refreshed to improve maintainability.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib.request
import urllib.error

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retroworld-ia")

# Base directories.  /mnt/data is writeable at runtime on Render and local
# development; /app contains static assets baked into the image.
BASE_DATA_DIR = "/mnt/data"
BASE_APP_DIR = "/app"

BASE_LOG_DIR = os.getenv("LOG_DIR", os.path.join(BASE_DATA_DIR, "logs"))
CONVERSATIONS_LOG_DIR = os.path.join(BASE_LOG_DIR, "conversations")
QWEEKLE_LOG_DIR = os.path.join(BASE_LOG_DIR, "qweekle")
for d in (BASE_LOG_DIR, CONVERSATIONS_LOG_DIR, QWEEKLE_LOG_DIR):
    os.makedirs(d, exist_ok=True)

# OpenAI configuration.  You can override these via environment variables.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Admin dashboard security token.  Set this in your environment when deploying.
ADMIN_DASHBOARD_TOKEN = os.getenv("ADMIN_DASHBOARD_TOKEN", "changeme_admin_token")

# Qweekle (booking system) settings – used only for webhook logging for now.
QWEEKLE_WEBHOOK_SECRET = os.getenv("QWEEKLE_WEBHOOK_SECRET", "")
QWEEKLE_SOURCE_NAME = os.getenv("QWEEKLE_SOURCE_NAME", "retroworld-qweekle")

# Supported brands.  The chat endpoint enforces that the URL path matches one
# of these strings and routes to the appropriate KB.
SUPPORTED_BRANDS: set[str] = {"retroworld", "runningman"}


# ---------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------

def load_kb(brand: str) -> Dict[str, Any]:
    """Load the knowledge base JSON for a given brand.

    KBs live in two locations: /mnt/data/kb_<brand>.json (runtime modifiable)
    and /app/kb_<brand>.json (baked into the container image).  If both exist,
    the version in /mnt/data overrides the embedded one.  If neither exists,
    an empty dict is returned.

    Args:
        brand: Lowercased brand identifier (e.g. "retroworld").

    Returns:
        Parsed JSON dictionary or an empty dict if not found.
    """
    brand = brand.lower()
    candidate_paths = [
        os.path.join(BASE_DATA_DIR, f"kb_{brand}.json"),
        os.path.join(BASE_APP_DIR, f"kb_{brand}.json"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    kb = json.load(f)
                logger.info("Loaded KB for %s from %s", brand, path)
                return kb
            except Exception as e:
                logger.error("Error reading KB %s: %s", path, e)
    logger.warning("KB not found for brand %s; using empty KB", brand)
    return {}


def save_kb(brand: str, kb_data: Dict[str, Any]) -> None:
    """Persist the KB for a brand to the writeable /mnt/data directory.

    This never touches the embedded version in /app; it always writes to
    /mnt/data/kb_<brand>.json so you can safely update the KB in production.

    Args:
        brand: The lowercased brand name.
        kb_data: JSON-serialisable dictionary representing the new KB.
    """
    brand = brand.lower()
    path = os.path.join(BASE_DATA_DIR, f"kb_{brand}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)
    logger.info("KB %s updated at %s", brand, path)


def call_openai_chat(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    """Send chat messages to OpenAI and return the assistant’s reply and usage.

    Args:
        messages: A list of messages in OpenAI chat format (role/content).

    Returns:
        A tuple containing the assistant’s reply text and a dict of usage metrics.

    Raises:
        RuntimeError: If OPENAI_API_KEY is missing.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.4,
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        logger.error("OpenAI HTTPError (%s): %s", e.code, err_body)
        raise
    except urllib.error.URLError as e:
        logger.error("OpenAI URLError: %s", e)
        raise
    obj = json.loads(body)
    content = obj["choices"][0]["message"]["content"]
    usage = obj.get("usage", {})
    return content, usage


def build_prompt(
    brand: str,
    kb: Dict[str, Any],
    messages: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Assemble the complete prompt to send to OpenAI.

    Combines multiple elements from the KB (identity, prompts, instructions, anti‑error
    guards) with conversation context (source, page URL, shared conversation ID)
    and all prior user/assistant messages.  If the KB is missing or empty,
    sensible defaults specific to the brand are used.

    Args:
        brand: The effective brand used for answering this chat turn.
        kb: The knowledge base dictionary.
        messages: The incoming messages (can be a full history or just the latest turn).
        metadata: Metadata from the front‑end (source, page URL, conversation ID).

    Returns:
        A list of messages in OpenAI chat format.
    """
    brand = brand.lower()
    system_parts: List[str] = []

    # Identity
    identite = kb.get("identite") if isinstance(kb, dict) else None
    if isinstance(identite, dict):
        nom = identite.get("nom") or brand.title()
        role = identite.get("role") or ""
        system_parts.append(f"Tu es l'assistant IA de {nom}. {role}".strip())
    elif isinstance(identite, str):
        system_parts.append(identite)

    # Prompt section: sorted for reproducibility
    prompt_section = kb.get("prompt") if isinstance(kb, dict) else None
    if isinstance(prompt_section, str):
        system_parts.append(prompt_section)
    elif isinstance(prompt_section, dict):
        for key in sorted(prompt_section.keys()):
            val = prompt_section[key]
            if isinstance(val, str) and val.strip():
                system_parts.append(val.strip())

    # General instructions
    instr = kb.get("instructions_generales") if isinstance(kb, dict) else None
    if isinstance(instr, list):
        for item in instr:
            if isinstance(item, str):
                system_parts.append(item)
    elif isinstance(instr, str):
        system_parts.append(instr)

    # Brand routing reminder
    brand_entry = metadata.get("brand_entry")
    brand_effective = metadata.get("brand_effective")
    if brand_entry and brand_effective and brand_entry != brand_effective:
        system_parts.append(
            f"La conversation vient d'un canal associé à '{brand_entry}', "
            f"mais tu dois répondre en utilisant les règles et tarifs de '{brand_effective}'. "
            "Explique clairement au client s'il s'agit de Retroworld (VR, quiz, salle enfant) "
            "ou de Runningman (action game, mini-jeux physiques)."
        )

    # Anti‑error instructions
    anti_err = kb.get("anti_erreurs") if isinstance(kb, dict) else None
    if isinstance(anti_err, list):
        for item in anti_err:
            if isinstance(item, str):
                system_parts.append(item)
    elif isinstance(anti_err, str):
        system_parts.append(anti_err)

    # Fallback defaults
    if not kb or not system_parts:
        if brand == "retroworld":
            system_parts.append(
                "Tu es l'assistant officiel de Retroworld France à Draguignan. "
                "Tu réponds en français, vouvoiement uniquement, sur les sujets suivants : "
                "jeux VR, escape games VR, quiz interactif, salle enfant, anniversaires et fidélité. "
                "Tu donnes toujours rapidement les prix, la durée et le nombre de joueurs possibles. "
                "Si tu n'es pas sûr d'une information, tu le dis et proposes au client d'appeler Retroworld au 04 94 47 94 64."
            )
        elif brand == "runningman":
            system_parts.append(
                "Tu es l'assistant pour Runningman Game Zone. "
                "Tu expliques que pour les informations précises (tarifs, réservations) concernant l'action game, "
                "il faut contacter directement Runningman au 04 98 09 30 59 ou via leur site www.runningmangames.fr. "
                "Tu peux néanmoins orienter le client vers Retroworld pour les activités VR, escape games VR, quiz et salle enfant."
            )

    system_text = "\n\n".join([p for p in system_parts if p.strip()])
    prompt_messages: List[Dict[str, str]] = []
    if system_text:
        prompt_messages.append({"role": "system", "content": system_text})

    # Metadata context
    meta_context: List[str] = []
    if metadata.get("source"):
        meta_context.append(f"Source de la demande : {metadata['source']}.")
    if metadata.get("page_url"):
        meta_context.append(f"URL de la page : {metadata['page_url']}.")
    conv_id = metadata.get("conversation_id")
    if conv_id:
        meta_context.append(
            "ID de conversation partagé entre sites : "
            f"{conv_id}. "
            "Si tu proposes un lien vers Runningman ou Retroworld, "
            "tu peux ajouter le paramètre ?convo_id="
            f"{conv_id} pour que la conversation continue sur l'autre site."
        )
    if meta_context:
        prompt_messages.append({"role": "system", "content": " ".join(meta_context)})

    # Conversation history
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role and content is not None and role in ("user", "assistant", "system"):
            prompt_messages.append({"role": role, "content": str(content)})

    return prompt_messages


def append_conversation_log(
    conversation_id: Optional[str],
    brand: str,
    channel: str,
    user_messages: List[Dict[str, Any]],
    assistant_reply: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a single conversation turn to a per‑conversation JSONL file."""
    if not conversation_id:
        conversation_id = f"conv_{int(time.time())}"
    path = os.path.join(CONVERSATIONS_LOG_DIR, f"{conversation_id}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = {
        "timestamp": time.time(),
        "conversation_id": conversation_id,
        "brand": brand,
        "channel": channel,
        "user_messages": user_messages,
        "assistant_reply": assistant_reply,
        "extra": extra or {},
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_conversation_records(conversation_id: str) -> List[Dict[str, Any]]:
    """Read all logged records for a conversation and sort by timestamp."""
    path = os.path.join(CONVERSATIONS_LOG_DIR, f"{conversation_id}.jsonl")
    if not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception:
                continue
    records.sort(key=lambda r: r.get("timestamp") or 0.0)
    return records


def reconstruct_history_from_logs(conversation_id: str) -> List[Dict[str, str]]:
    """Rebuild the OpenAI message history from saved logs."""
    records = load_conversation_records(conversation_id)
    history: List[Dict[str, str]] = []
    for rec in records:
        rec_msgs = rec.get("user_messages") or []
        for m in rec_msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant", "system") and content is not None:
                history.append({"role": role, "content": str(content)})
        assistant_text = rec.get("assistant_reply")
        if assistant_text:
            history.append({"role": "assistant", "content": str(assistant_text)})
    return history


def detect_brand_from_text(text: str, default: str = "runningman") -> str:
    """Detect whether a message refers to Retroworld or Runningman."""
    if not text:
        return default
    t = text.lower()
    retro_keywords = [
        "vr",
        "réalité virtuelle",
        "realite virtuelle",
        "escape vr",
        "escape game vr",
        "jeux vr",
        "jeu vr",
        "casque vr",
        "quiz",
        "quizz",
        "quiz interactif",
        "salle enfant",
        "mur interactif",
        "anniversaire vr",
        "retroworld",
        "rétroworld",
        "fidélité",
        "fidelite",
        "carte de fidélité",
        "points fidélité",
        "points de fidelite",
    ]
    running_keywords = [
        "action game",
        "game zone",
        "runningman",
        "running man",
        "mini-jeux",
        "mini jeux",
        "parcours",
        "parcour",
        "salle runningman",
        "gilet",
        "capteur",
        "mission physique",
    ]
    retro_score = sum(1 for k in retro_keywords if k in t)
    running_score = sum(1 for k in running_keywords if k in t)
    if retro_score > running_score and retro_score > 0:
        return "retroworld"
    if running_score > retro_score and running_score > 0:
        return "runningman"
    if "retroworld" in t or "rétroworld" in t:
        return "retroworld"
    if "runningman" in t or "running man" in t:
        return "runningman"
    return default


def classify_conversation_brands(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Determine which brands have been referenced in a conversation."""
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
        return {"brand_final": list(brands_seen)[0], "brands_seen": list(brands_seen)}
    brand_final = last_effective or "mixed"
    if brand_final not in ("runningman", "retroworld"):
        brand_final = "mixed"
    return {"brand_final": brand_final, "brands_seen": list(brands_seen)}


def append_qweekle_event(event_type: str, payload: Dict[str, Any]) -> None:
    """Persist raw Qweekle webhook events to disk."""
    fname = f"{event_type or 'unknown'}.jsonl"
    path = os.path.join(QWEEKLE_LOG_DIR, fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = {
        "timestamp": time.time(),
        "event_type": event_type,
        "payload": payload,
        "source": QWEEKLE_SOURCE_NAME,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------

@app.route("/", methods=["GET", "HEAD"])
def root() -> Tuple[Dict[str, Any], int]:
    """Lightweight root endpoint for health checks."""
    return jsonify(
        {
            "service": "retroworld-ia",
            "status": "ok",
            "time": time.time(),
            "brands": list(SUPPORTED_BRANDS),
        }
    ), 200


@app.route("/favicon.ico", methods=["GET"])
def favicon():  # type: ignore[override]
    """Return no favicon to avoid 404s in logs."""
    return "", 204


@app.route("/health", methods=["GET"])
def health():  # type: ignore[override]
    """Simple health endpoint."""
    return jsonify({"status": "ok", "time": time.time()}), 200


@app.route("/chat/<brand>", methods=["POST"])
def chat_route(brand: str):  # type: ignore[override]
    """Generic chat endpoint for both brands."""
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid_json"}), 400
    if not isinstance(body, dict):
        return jsonify({"error": "invalid_payload"}), 400
    messages = body.get("messages") or []
    metadata = body.get("metadata") or {}
    if not isinstance(messages, list):
        return jsonify({"error": "messages must be a list"}), 400
    if not isinstance(metadata, dict):
        metadata = {}
    brand_entry = (brand or "").lower()
    if brand_entry not in SUPPORTED_BRANDS:
        return jsonify({"error": "unknown_brand"}), 404
    last_user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_text = str(msg.get("content") or "")
            break
    effective_brand = brand_entry
    if brand_entry == "runningman":
        effective_brand = detect_brand_from_text(last_user_text, default="runningman")
    conversation_id = metadata.get("conversation_id")
    if not conversation_id:
        conversation_id = f"{effective_brand}_{int(time.time() * 1000)}"
        metadata["conversation_id"] = conversation_id
    metadata["brand_entry"] = brand_entry
    metadata["brand_effective"] = effective_brand
    messages_for_prompt: List[Dict[str, Any]] = messages
    try:
        only_user_simple = (
            len(messages) == 1
            and isinstance(messages[0], dict)
            and messages[0].get("role") == "user"
        )
        no_assistant_msgs = all(
            (isinstance(m, dict) and m.get("role") != "assistant") for m in messages
        )
        use_server_history = conversation_id and (only_user_simple or no_assistant_msgs)
        if metadata.get("no_server_history") is True:
            use_server_history = False
        if use_server_history:
            past_history = reconstruct_history_from_logs(conversation_id)
            if past_history:
                messages_for_prompt = past_history + messages
                logger.info(
                    "Reconstructed history for %s (%d past messages + %d new)",
                    conversation_id,
                    len(past_history),
                    len(messages),
                )
    except Exception as e:
        logger.error("Error reconstructing history for %s: %s", conversation_id, e)
        messages_for_prompt = messages
    kb = load_kb(effective_brand)
    try:
        prompt_messages = build_prompt(effective_brand, kb, messages_for_prompt, metadata)
    except Exception as e:
        logger.error("build_prompt failed: %s", e)
        return jsonify({"error": "prompt_build_failed"}), 500
    try:
        reply_text, usage = call_openai_chat(prompt_messages)
    except Exception as e:
        logger.error("OpenAI error: %s", e)
        return jsonify({"error": "openai_error", "details": str(e)}), 502
    try:
        channel = metadata.get("source") or "web"
        append_conversation_log(
            conversation_id=conversation_id,
            brand=effective_brand,
            channel=channel,
            user_messages=messages,
            assistant_reply=reply_text,
            extra={
                "brand_entry": brand_entry,
                "brand_effective": effective_brand,
                "metadata": metadata,
                "openai_usage": usage,
            },
        )
    except Exception as e:
        logger.error("Error logging conversation: %s", e)
    return jsonify(
        {
            "reply": reply_text,
            "brand_used": effective_brand,
            "brand_entry": brand_entry,
        }
    ), 200


@app.route("/kb/upsert/<brand>", methods=["POST"])
def kb_upsert(brand: str):  # type: ignore[override]
    """Upsert (create or overwrite) a brand’s KB via POST."""
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid_json"}), 400
    if not isinstance(body, dict):
        return jsonify({"error": "invalid_kb"}), 400
    try:
        save_kb(brand, body)
    except Exception as e:
        logger.error("save_kb(%s) failed: %s", brand, e)
        return jsonify({"error": "kb_save_failed"}), 500
    return jsonify({"status": "ok", "brand": brand}), 200


@app.route("/webhooks/qweekle", methods=["POST"])
def qweekle_webhook():  # type: ignore[override]
    """Handle Qweekle webhook events by logging them to disk."""
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
    append_qweekle_event(event_type, payload)
    return jsonify({"status": "ok", "event_type": event_type}), 200


@app.route("/admin/api/conversations", methods=["GET"])
def admin_api_conversations():  # type: ignore[override]
    """List recent conversations for the admin dashboard."""
    token = request.args.get("token") or ""
    if token != ADMIN_DASHBOARD_TOKEN:
        return jsonify({"error": "forbidden"}), 403
    convs: List[Dict[str, Any]] = []
    if not os.path.isdir(CONVERSATIONS_LOG_DIR):
        return jsonify(convs), 200
    for fname in os.listdir(CONVERSATIONS_LOG_DIR):
        if not fname.endswith(".jsonl"):
            continue
        conversation_id = fname.replace(".jsonl", "")
        records = load_conversation_records(conversation_id)
        if not records:
            continue
        last = records[-1]
        ts = last.get("timestamp") or 0.0
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
        if len(preview) > 120:
            preview = preview[:117] + "..."
        convs.append(
            {
                "conversation_id": conversation_id,
                "timestamp": ts,
                "channel": channel,
                "source": source,
                "preview": preview,
                "brand_final": brand_final,
            }
        )
    convs.sort(key=lambda c: c["timestamp"], reverse=True)
    return jsonify(convs), 200


@app.route("/admin/api/conversation/<conversation_id>", methods=["GET"])
def admin_api_conversation_detail(conversation_id: str):  # type: ignore[override]
    """Return all records of a single conversation."""
    token = request.args.get("token") or ""
    if token != ADMIN_DASHBOARD_TOKEN:
        return jsonify({"error": "forbidden"}), 403
    records = load_conversation_records(conversation_id)
    if not records:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"conversation_id": conversation_id, "records": records}), 200


@app.route("/admin/conversations", methods=["GET"])
def admin_conversations_page():  # type: ignore[override]
    """Serve the admin dashboard with improved styling and usability."""
    token = request.args.get("token") or ""
    if token != ADMIN_DASHBOARD_TOKEN:
        return "Forbidden", 403
    return """
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <title>Admin IA – Conversations</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --bg: #0f172a;
      --bg-card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #0ea5e9;
      --brand-retro: #6366f1;
      --brand-run: #22c55e;
      --brand-mix: #f97316;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background-color: var(--bg);
      color: var(--text);
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px 20px 40px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      flex-wrap: wrap;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      font-weight: 600;
    }
    .subtitle {
      margin-top: 4px;
      font-size: 13px;
      color: var(--muted);
    }
    .filters {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .chip {
      border-radius: 20px;
      border: 1px solid var(--border);
      padding: 6px 12px;
      font-size: 12px;
      cursor: pointer;
      background-color: var(--bg-card);
      color: var(--muted);
      transition: background-color 0.15s, border-color 0.15s;
    }
    .chip.active {
      background-color: var(--accent);
      border-color: var(--accent);
      color: var(--bg);
    }
    .chip[data-brand="runningman"].active { background-color: var(--brand-run); border-color: var(--brand-run); color: var(--bg); }
    .chip[data-brand="retroworld"].active { background-color: var(--brand-retro); border-color: var(--brand-retro); color: var(--bg); }
    .chip[data-brand="mixed"].active { background-color: var(--brand-mix); border-color: var(--brand-mix); color: var(--bg); }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 20px;
      align-items: center;
    }
    .search-input {
      flex: 1;
      min-width: 200px;
      position: relative;
    }
    .search-input input {
      width: 100%;
      padding: 8px 12px;
      border-radius: 20px;
      border: 1px solid var(--border);
      background-color: var(--bg-card);
      color: var(--text);
      font-size: 14px;
    }
    .btn-refresh {
      border-radius: 20px;
      border: 1px solid var(--border);
      padding: 8px 12px;
      font-size: 12px;
      background-color: var(--bg-card);
      color: var(--muted);
      cursor: pointer;
      transition: color 0.15s, border-color 0.15s;
    }
    .btn-refresh:hover {
      border-color: var(--accent);
      color: var(--accent);
    }
    .table-wrapper {
      border-radius: 16px;
      border: 1px solid var(--border);
      background-color: var(--bg-card);
      overflow: hidden;
      margin-bottom: 20px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    thead {
      background-color: var(--bg);
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }
    th {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }
    tr:hover td {
      background-color: rgba(30, 41, 59, 0.5);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 12px;
      padding: 2px 8px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .badge-run { background-color: rgba(34,197,94,0.2); color: #22c55e; }
    .badge-retro { background-color: rgba(99,102,241,0.2); color: #6366f1; }
    .badge-mix { background-color: rgba(249,115,22,0.2); color: #f97316; }
    .badge-unknown { background-color: rgba(148,163,184,0.2); color: #94a3b8; }
    .pill {
      display: inline-flex;
      align-items: center;
      border-radius: 12px;
      padding: 2px 6px;
      font-size: 11px;
      color: var(--muted);
      border: 1px solid rgba(148,163,184,0.4);
    }
    .pill-channel { text-transform: uppercase; letter-spacing: 0.06em; }
    .preview-text {
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .conversation-detail {
      border-radius: 16px;
      border: 1px solid var(--border);
      background-color: var(--bg-card);
      padding: 16px;
      max-height: 450px;
      overflow-y: auto;
    }
    .conversation-detail h2 {
      margin: 0 0 10px 0;
      font-size: 16px;
      color: var(--muted);
    }
    .bubble {
      max-width: 80%;
      margin-bottom: 8px;
      padding: 8px 12px;
      border-radius: 16px;
      font-size: 14px;
      line-height: 1.4;
      white-space: pre-wrap;
    }
    .bubble-user {
      margin-left: auto;
      background-color: #334155;
      border-bottom-right-radius: 4px;
    }
    .bubble-bot {
      margin-right: auto;
      background-color: #1e293b;
      border-bottom-left-radius: 4px;
    }
    .timestamp {
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 12px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Conversations IA</h1>
        <div class="subtitle">Suivi des échanges Retroworld / Runningman (chat & téléphone)</div>
      </div>
      <div class="filters">
        <button class="chip active" data-filter="all">Tout</button>
        <button class="chip" data-filter="runningman" data-brand="runningman">Runningman</button>
        <button class="chip" data-filter="retroworld" data-brand="retroworld">Retroworld</button>
        <button class="chip" data-filter="mixed" data-brand="mixed">Mix des deux</button>
      </div>
    </header>
    <div class="toolbar">
      <div class="search-input">
        <input type="text" id="search" placeholder="Rechercher dans les questions, sources, IDs…" />
      </div>
      <button class="btn-refresh" id="btn-refresh">Rafraîchir</button>
    </div>
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th style="width: 170px;">Date</th>
            <th style="width: 85px;">Canal</th>
            <th style="width: 110px;">Marque</th>
            <th style="width: 140px;">Source</th>
            <th>Dernier message</th>
          </tr>
        </thead>
        <tbody id="rows">
          <tr><td colspan="5" class="muted">Chargement…</td></tr>
        </tbody>
      </table>
    </div>
    <div class="conversation-detail" id="convDetail">
      <h2>Détail de la conversation</h2>
      <div class="muted">Sélectionnez une conversation ci‑dessus pour voir le fil complet.</div>
    </div>
  </div>
  <script>
  (function() {
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token") || "";
    const rowsEl = document.getElementById("rows");
    const searchInput = document.getElementById("search");
    const btnRefresh = document.getElementById("btn-refresh");
    const chips = Array.from(document.querySelectorAll(".chip"));
    const convDetail = document.getElementById("convDetail");
    let allData = [];
    let currentFilter = "all";
    let searchTerm = "";
    function formatDate(ts) {
      if (!ts) return "";
      try {
        const d = new Date(ts * 1000);
        return d.toLocaleString("fr-FR", {
          day: "2-digit",
          month: "2-digit",
          year: "2-digit",
          hour: "2-digit",
          minute: "2-digit"
        });
      } catch(e) { return ""; }
    }
    function brandBadge(conv) {
      const b = conv.brand_final;
      if (b === "runningman") return '<span class="badge badge-run">Runningman</span>';
      if (b === "retroworld") return '<span class="badge badge-retro">Retroworld</span>';
      if (b === "mixed") return '<span class="badge badge-mix">Mix</span>';
      return '<span class="badge badge-unknown">Inconnu</span>';
    }
    function channelPill(conv) {
      const ch = (conv.channel || "web").toUpperCase();
      return '<span class="pill pill-channel">' + ch + '</span>';
    }
    function sourcePill(conv) {
      const s = conv.source || "n/a";
      return '<span class="pill">' + s + '</span>';
    }
    function render() {
      const term = searchTerm.trim().toLowerCase();
      let filtered = allData.slice();
      if (currentFilter !== "all") {
        filtered = filtered.filter(c => {
          if (currentFilter === "mixed") return c.brand_final === "mixed";
          return c.brand_final === currentFilter;
        });
      }
      if (term) {
        filtered = filtered.filter(c =>
          (c.preview && c.preview.toLowerCase().includes(term)) ||
          (c.source && c.source.toLowerCase().includes(term)) ||
          (c.conversation_id && c.conversation_id.toLowerCase().includes(term))
        );
      }
      if (!filtered.length) {
        rowsEl.innerHTML = '<tr><td colspan="5" class="muted">Aucune conversation trouvée.</td></tr>';
        return;
      }
      const html = filtered.map(c => `
        <tr onclick="viewConversation('${c.conversation_id}')">
          <td>
            <div>${formatDate(c.timestamp)}</div>
            <div class="muted" style="font-size:11px;">${c.conversation_id}</div>
          </td>
          <td>${channelPill(c)}</td>
          <td>${brandBadge(c)}</td>
          <td>${sourcePill(c)}</td>
          <td>
            <div class="preview-text">${c.preview || '<span class="muted">(pas de message)</span>'}</div>
          </td>
        </tr>
      `).join("");
      rowsEl.innerHTML = html;
    }
    async function loadData() {
      rowsEl.innerHTML = '<tr><td colspan="5" class="muted">Chargement…</td></tr>';
      try {
        const res = await fetch(`/admin/api/conversations?token=${encodeURIComponent(token)}`);
        if (!res.ok) {
          rowsEl.innerHTML = '<tr><td colspan="5" class="muted">Erreur de chargement ('+res.status+')</td></tr>';
          return;
        }
        allData = await res.json();
        render();
      } catch (e) {
        console.error(e);
        rowsEl.innerHTML = '<tr><td colspan="5" class="muted">Erreur réseau.</td></tr>';
      }
    }
    window.viewConversation = async function(id) {
      convDetail.innerHTML = '<h2>Conversation ' + id + '</h2><div class="muted">Chargement…</div>';
      try {
        const res = await fetch(`/admin/api/conversation/${encodeURIComponent(id)}?token=${encodeURIComponent(token)}`);
        if (!res.ok) {
          convDetail.innerHTML = '<h2>Conversation ' + id + '</h2><div class="muted">Erreur de chargement ('+res.status+')</div>';
          return;
        }
        const data = await res.json();
        const records = data.records || [];
        if (!records.length) {
          convDetail.innerHTML = '<h2>Conversation ' + id + '</h2><div class="muted">Aucun enregistrement pour cette conversation.</div>';
          return;
        }
        let html = '<h2>Conversation ' + id + '</h2>';
        records.forEach(rec => {
          const userMsgs = rec.user_messages || [];
          const reply = rec.assistant_reply || "";
          userMsgs
            .filter(m => m.role === "user")
            .forEach(m => {
              html += '<div class="bubble bubble-user">' + (m.content || "") + '</div>';
            });
          if (reply) {
            html += '<div class="bubble bubble-bot">' + reply + '</div>';
          }
          if (rec.timestamp) {
            const d = new Date(rec.timestamp * 1000).toLocaleString("fr-FR");
            html += '<div class="timestamp">' + d + '</div>';
          }
        });
        convDetail.innerHTML = html;
        convDetail.scrollTop = convDetail.scrollHeight;
      } catch (e) {
        console.error(e);
        convDetail.innerHTML = '<h2>Conversation ' + id + '</h2><div class="muted">Erreur réseau.</div>';
      }
    };
    searchInput.addEventListener("input", function() {
      searchTerm = this.value;
      render();
    });
    btnRefresh.addEventListener("click", loadData);
    chips.forEach(chip => {
      chip.addEventListener("click", () => {
        chips.forEach(c => c.classList.remove("active"));
        chip.classList.add("active");
        currentFilter = chip.getAttribute("data-filter") || "all";
        render();
      });
    });
    loadData();
  })();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
