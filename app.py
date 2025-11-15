import os
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from flask import Flask, request, jsonify, Response, abort
from flask_cors import CORS
import urllib.request
import urllib.error

# =========================================================
# CONFIG GLOBALE & VARIABLES D’ENVIRONNEMENT
# =========================================================
# À définir dans Render (Environment) :
#
# OPENAI_API_KEY=...
# OPENAI_MODEL=gpt-4.1-mini        (optionnel)
# PUBLIC_BASE_URL=https://retroworld-ia.onrender.com
#
# ADMIN_DASHBOARD_TOKEN=...        (ou ADMIN_API_TOKEN=...)
#
# VONAGE_NUMBER_RETROWORLD=+33...
# VONAGE_NUMBER_RUNNINGMAN=+33...  (optionnel)
#
# FORWARD_NUMBER_RETROWORLD=+33...
# FORWARD_NUMBER_RUNNINGMAN=+33...
#
# (si FORWARD_NUMBER_* absents, on peut réutiliser
#  FORWARD_NUMBERS_* existants en prenant le 1er numéro)
# =========================================================

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retroworld-ia")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# URL publique (Render)
PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").rstrip("/")

# Marques gérées
ALLOWED_BRANDS = {"retroworld", "runningman"}

# Dossiers de log
BASE_LOG_DIR = "/mnt/data/logs"
CHAT_LOG_DIR = os.path.join(BASE_LOG_DIR, "chat")
CALL_LOG_DIR = os.path.join(BASE_LOG_DIR, "calls")

# Token admin (dashboard conversations)
ADMIN_DASHBOARD_TOKEN = (
    os.getenv("ADMIN_DASHBOARD_TOKEN")
    or os.getenv("ADMIN_API_TOKEN")
    or ""
)

# Numéros Vonage (numéros achetés chez Vonage)
VONAGE_NUMBER_RETROWORLD = (os.getenv("VONAGE_NUMBER_RETROWORLD") or "").replace(" ", "")
VONAGE_NUMBER_RUNNINGMAN = (os.getenv("VONAGE_NUMBER_RUNNINGMAN") or "").replace(" ", "")

# Numéros de transfert vers humains
FORWARD_NUMBER_RETROWORLD = os.getenv("FORWARD_NUMBER_RETROWORLD", "").replace(" ", "")
FORWARD_NUMBER_RUNNINGMAN = os.getenv("FORWARD_NUMBER_RUNNINGMAN", "").replace(" ", "")

# Compat avec anciennes variables (au cas où tu les as déjà)
if not FORWARD_NUMBER_RETROWORLD:
    plural = os.getenv("FORWARD_NUMBERS_RETROWORLD", "")
    if plural:
        FORWARD_NUMBER_RETROWORLD = plural.split(",")[0].strip()
if not FORWARD_NUMBER_RUNNINGMAN:
    plural = os.getenv("FORWARD_NUMBERS_RUNNINGMAN", "")
    if plural:
        FORWARD_NUMBER_RUNNINGMAN = plural.split(",")[0].strip()


# =========================================================
# OUTILS UTILITAIRES
# =========================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_kb_paths(brand: str) -> Dict[str, str]:
    filename = f"kb_{brand}.json"
    return {
        "mnt": os.path.join("/mnt/data", filename),
        "app": os.path.join("/app", filename),
    }


def load_kb(brand: str) -> Dict[str, Any]:
    """
    1) /mnt/data/kb_<brand>.json
    2) /app/kb_<brand>.json
    3) KB minimale si rien trouvé
    """
    paths = get_kb_paths(brand)

    for label, path in paths.items():
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    kb = json.load(f)
                    logger.info("KB %s chargée depuis %s (%s)", brand, path, label)
                    return kb
        except Exception as e:
            logger.error("Erreur chargement KB %s (%s): %s", brand, path, e)

    logger.warning("Aucune KB trouvée pour %s, utilisation KB minimale.", brand)
    return {
        "identite": {
            "nom": brand,
            "description": f"KB minimale pour {brand}",
        },
        "prompt": {
            "instructions_generales": [
                "Tu es l'assistant de cette marque.",
                "Réponds de manière professionnelle, claire et cohérente avec les règles.",
            ]
        },
        "anti_erreurs": {},
    }


def save_kb(brand: str, kb_data: Dict[str, Any]) -> str:
    paths = get_kb_paths(brand)
    target = paths["mnt"]
    ensure_dir(os.path.dirname(target))
    with open(target, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)
    logger.info("KB %s sauvegardée dans %s", brand, target)
    return target


def call_openai_chat(
    messages: List[Dict[str, str]],
    model: str = OPENAI_MODEL_DEFAULT,
    temperature: float = 0.3,
) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY non défini.")

    payload = {"model": model, "messages": messages, "temperature": temperature}
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        OPENAI_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            obj = json.loads(body)
            return obj["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        logger.error("HTTPError OpenAI %s: %s", e.code, e.read().decode("utf-8", "ignore"))
        raise
    except Exception as e:
        logger.error("Erreur OpenAI: %s", e)
        raise


def build_system_messages(brand: str, kb: Dict[str, Any], channel: str) -> List[Dict[str, str]]:
    identite = kb.get("identite", {})
    prompt = kb.get("prompt", {})
    anti = kb.get("anti_erreurs", {})

    nom = identite.get("nom", brand)
    description = identite.get("description", "")

    instructions_generales = prompt.get("instructions_generales", [])
    style = prompt.get("style", "")
    objectifs = prompt.get("objectifs", "")

    kb_json = json.dumps(kb, ensure_ascii=False)

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                f"Tu es l'assistant IA officiel de {nom}."
                f" Description : {description}."
                f" Canal : {channel}."
                " Respecte strictement la base de connaissance JSON fournie."
            ),
        },
        {
            "role": "system",
            "content": f"Style : {style}. Objectifs : {objectifs}.",
        },
        {
            "role": "system",
            "content": "Instructions générales :\n" + "\n".join(instructions_generales),
        },
        {
            "role": "system",
            "content": "Règles anti-erreurs : " + json.dumps(anti, ensure_ascii=False),
        },
        {
            "role": "system",
            "content": "BASE_DE_CONNAISSANCE_JSON:\n" + kb_json,
        },
    ]

    if channel == "phone":
        messages.append(
            {
                "role": "system",
                "content": (
                    "Tu es en conversation téléphonique. Parle avec des phrases courtes et claires. "
                    "Ne lis jamais les URLs à voix haute. "
                    "Si tu dois envoyer un lien (paiement, réservation), indique seulement que le client "
                    "le recevra par SMS ou email. "
                    "Si l'appelant demande clairement un humain, écris exactement [[ACTION:TRANSFER_HUMAN]] "
                    "sur une ligne séparée dans ta réponse."
                ),
            }
        )

    return messages


# =========================================================
# LOGS CONVERSATIONS (WEB + TÉLÉPHONE)
# =========================================================

def get_conversation_log_path(conversation_id: str, channel: str) -> str:
    ensure_dir(BASE_LOG_DIR)
    if channel == "phone":
        ensure_dir(CALL_LOG_DIR)
        return os.path.join(CALL_LOG_DIR, f"{conversation_id}.jsonl")
    ensure_dir(CHAT_LOG_DIR)
    return os.path.join(CHAT_LOG_DIR, f"{conversation_id}.jsonl")


def append_conversation_log(
    conversation_id: str,
    brand: str,
    channel: str,
    role: str,
    content: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = get_conversation_log_path(conversation_id, channel)
    rec: Dict[str, Any] = {
        "timestamp": now_iso(),
        "conversation_id": conversation_id,
        "brand": brand,
        "channel": channel,
        "role": role,
        "content": content,
    }
    if extra:
        rec.update(extra)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_conversation(conversation_id: str) -> Tuple[List[Dict[str, Any]], str]:
    for channel in ("web", "phone"):
        path = get_conversation_log_path(conversation_id, channel)
        if os.path.exists(path):
            out: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
            return out, channel
    return [], ""


def list_conversations_summary() -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []

    for channel, base_dir in (("web", CHAT_LOG_DIR), ("phone", CALL_LOG_DIR)):
        if not os.path.isdir(base_dir):
            continue
        for filename in os.listdir(base_dir):
            if not filename.endswith(".jsonl"):
                continue
            path = os.path.join(base_dir, filename)
            conv_id = filename[:-6]

            first = None
            last = None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        if first is None:
                            first = rec
                        last = rec
            except Exception:
                continue

            if not first or not last:
                continue

            summaries.append(
                {
                    "conversation_id": conv_id,
                    "channel": channel,
                    "brand": first.get("brand", ""),
                    "start": first.get("timestamp"),
                    "end": last.get("timestamp"),
                    "last_role": last.get("role"),
                    "last_snippet": (last.get("content") or "")[:160],
                }
            )

    summaries.sort(key=lambda s: s.get("end") or "", reverse=True)
    return summaries


# =========================================================
# DÉTECTION MARQUE & NUMÉROS
# =========================================================

def detect_brand_from_to_number(to_number: str) -> str:
    num = (to_number or "").replace(" ", "")
    if VONAGE_NUMBER_RETROWORLD and num == VONAGE_NUMBER_RETROWORLD:
        return "retroworld"
    if VONAGE_NUMBER_RUNNINGMAN and num == VONAGE_NUMBER_RUNNINGMAN:
        return "runningman"
    return "retroworld"  # défaut


def get_forward_number_for_brand(brand: str) -> str:
    if brand == "runningman" and FORWARD_NUMBER_RUNNINGMAN:
        return FORWARD_NUMBER_RUNNINGMAN
    return FORWARD_NUMBER_RETROWORLD or "+33494479464"


# =========================================================
# ENDPOINTS GÉNÉRAUX
# =========================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# =========================================================
# CHAT WEB : /chat/<brand>
# =========================================================

@app.route("/chat/<brand>", methods=["POST"])
def chat_brand(brand: str):
    """
    Endpoint pour le chat web (Retroworld / Runningman).

    JSON attendu :
    {
      "messages": [ { "role": "user"|"assistant", "content": "..." }, ... ],
      "metadata": { "conversation_id": "...", ... } (optionnel)
    }
    """
    brand = brand.lower()
    if brand not in ALLOWED_BRANDS:
        return jsonify({"error": "Marque inconnue.", "allowed_brands": list(ALLOWED_BRANDS)}), 400

    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "JSON invalide."}), 400

    if not isinstance(body, dict):
        return jsonify({"error": "Corps JSON invalide."}), 400

    user_messages = body.get("messages")
    metadata = body.get("metadata") or {}

    if not user_messages or not isinstance(user_messages, list):
        return jsonify({"error": "'messages' (liste) est requis."}), 400

    # conversation_id type tawk.to
    conversation_id = metadata.get("conversation_id") or str(uuid.uuid4())
    metadata["conversation_id"] = conversation_id

    kb = load_kb(brand)
    system_messages = build_system_messages(brand, kb, channel="web")

    messages_for_openai = system_messages + user_messages

    try:
        assistant_reply = call_openai_chat(messages_for_openai, temperature=0.3)
    except Exception as e:
        logger.error("Erreur OpenAI chat %s: %s", brand, e)
        return jsonify({"error": "Erreur IA interne."}), 500

    # log dernier message user + réponse
    last_user = ""
    for msg in reversed(user_messages):
        if msg.get("role") == "user":
            last_user = msg.get("content", "")
            break

    if last_user:
        append_conversation_log(
            conversation_id,
            brand=brand,
            channel="web",
            role="user",
            content=last_user,
            extra={"metadata": metadata},
        )

    append_conversation_log(
        conversation_id,
        brand=brand,
        channel="web",
        role="assistant",
        content=assistant_reply,
        extra={"metadata": metadata},
    )

    return jsonify(
        {
            "reply": assistant_reply,
            "brand": brand,
            "conversation_id": conversation_id,
            "metadata": metadata,
        }
    ), 200


# =========================================================
# KB UPSERT
# =========================================================

@app.route("/kb/upsert/<brand>", methods=["POST"])
def kb_upsert(brand: str):
    brand = brand.lower()
    if brand not in ALLOWED_BRANDS:
        return jsonify({"error": "Marque inconnue."}), 400

    try:
        kb = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "JSON invalide."}), 400

    if not isinstance(kb, dict):
        return jsonify({"error": "La KB doit être un objet JSON."}), 400

    try:
        path = save_kb(brand, kb)
    except Exception as e:
        logger.error("Erreur sauvegarde KB %s: %s", brand, e)
        return jsonify({"error": "Impossible de sauvegarder la KB."}), 500

    return jsonify({"status": "ok", "brand": brand, "saved_to": path}), 200


# =========================================================
# PARTIE TÉLÉPHONE / VONAGE
# =========================================================

@app.route("/voice/incoming", methods=["POST"])
def voice_incoming():
    """
    Answer URL Vonage.
    Renvoie une NCCO (JSON) : message de bienvenue + attente de parole.
    """
    data = request.get_json(force=True, silent=True) or {}
    logger.info("Vonage incoming: %s", data)

    to_num = data.get("to") or data.get("to_number", "")
    from_num = data.get("from") or data.get("from_number", "")
    call_uuid = data.get("uuid") or data.get("conversation_uuid") or str(uuid.uuid4())

    brand = detect_brand_from_to_number(str(to_num))
    kb = load_kb(brand)
    nom = kb.get("identite", {}).get("nom", brand.capitalize())

    append_conversation_log(
        call_uuid,
        brand=brand,
        channel="phone",
        role="system",
        content="CALL_STARTED",
        extra={"from": from_num, "to": to_num},
    )

    base_url = PUBLIC_BASE_URL or request.url_root.rstrip("/")
    event_url = f"{base_url}/voice/assistant?brand={brand}"

    greeting = (
        f"Bonjour, vous êtes bien chez {nom}. "
        "Je suis l'assistant vocal. Dites-moi en quelques mots ce que vous souhaitez : "
        "par exemple réserver une partie, organiser un anniversaire ou avoir des informations."
    )

    ncco = [
        {"action": "talk", "text": greeting, "language": "fr-FR"},
        {
            "action": "input",
            "type": ["speech"],
            "speech": {"language": "fr-FR", "endOnSilence": 1},
            "eventUrl": [event_url],
        },
    ]
    return jsonify(ncco), 200


@app.route("/voice/assistant", methods=["POST"])
def voice_assistant():
    """
    Event URL Vonage pour les résultats de reconnaissance vocale.
    Reçoit le texte, appelle OpenAI, renvoie une nouvelle NCCO.
    """
    event = request.get_json(force=True, silent=True) or {}
    logger.info("Vonage assistant event: %s", event)

    brand = request.args.get("brand", "retroworld").lower()
    if brand not in ALLOWED_BRANDS:
        brand = "retroworld"

    call_uuid = (
        event.get("uuid")
        or event.get("conversation_uuid")
        or event.get("call_uuid")
        or str(uuid.uuid4())
    )
    from_num = event.get("from") or ""
    to_num = event.get("to") or ""

    # texte reconnu
    speech = event.get("speech") or {}
    results = speech.get("results") or []
    transcript = ""
    if results and isinstance(results, list):
        transcript = results[0].get("text", "") or ""

    if not transcript.strip():
        base_url = PUBLIC_BASE_URL or request.url_root.rstrip("/")
        event_url = f"{base_url}/voice/assistant?brand={brand}"
        ncco = [
            {
                "action": "talk",
                "text": "Je n'ai pas bien compris. Pouvez-vous répéter, s'il vous plaît ?",
                "language": "fr-FR",
            },
            {
                "action": "input",
                "type": ["speech"],
                "speech": {"language": "fr-FR", "endOnSilence": 1},
                "eventUrl": [event_url],
            },
        ]
        return jsonify(ncco), 200

    append_conversation_log(
        call_uuid,
        brand=brand,
        channel="phone",
        role="user",
        content=transcript,
        extra={"from": from_num, "to": to_num},
    )

    kb = load_kb(brand)
    system_messages = build_system_messages(brand, kb, channel="phone")

    # petit historique
    old_records, _ = load_conversation(call_uuid)
    history: List[Dict[str, str]] = []
    for rec in old_records:
        if rec.get("channel") != "phone":
            continue
        role = rec.get("role")
        content = rec.get("content", "")
        if role in ("user", "assistant"):
            history.append({"role": role, "content": content})

    messages = system_messages + history + [{"role": "user", "content": transcript}]

    try:
        assistant_reply = call_openai_chat(messages, temperature=0.4)
    except Exception as e:
        logger.error("Erreur IA téléphone %s: %s", call_uuid, e)
        text = (
            "Je rencontre un problème technique pour le moment. "
            "Je vous invite à rappeler un peu plus tard ou à nous contacter via le site Retroworld France."
        )
        append_conversation_log(
            call_uuid,
            brand=brand,
            channel="phone",
            role="assistant",
            content=text,
            extra={"error": str(e)},
        )
        return jsonify([{"action": "talk", "text": text, "language": "fr-FR"}]), 200

    append_conversation_log(
        call_uuid,
        brand=brand,
        channel="phone",
        role="assistant",
        content=assistant_reply,
        extra={},
    )

    # détection transfert humain
    transfer_requested = "[[ACTION:TRANSFER_HUMAN]]" in assistant_reply
    spoken_reply = assistant_reply.replace("[[ACTION:TRANSFER_HUMAN]]", "").strip()
    if len(spoken_reply) > 800:
        spoken_reply = spoken_reply[:800]

    base_url = PUBLIC_BASE_URL or request.url_root.rstrip("/")
    event_url = f"{base_url}/voice/assistant?brand={brand}"

    if transfer_requested:
        human_number = get_forward_number_for_brand(brand)
        ncco = [
            {
                "action": "talk",
                "text": spoken_reply
                or "Je vous mets en relation avec un membre de l'équipe.",
                "language": "fr-FR",
            },
            {
                "action": "connect",
                "endpoint": [{"type": "phone", "number": human_number}],
            },
        ]
        return jsonify(ncco), 200

    ncco = [
        {"action": "talk", "text": spoken_reply, "language": "fr-FR"},
        {
            "action": "input",
            "type": ["speech"],
            "speech": {"language": "fr-FR", "endOnSilence": 1},
            "eventUrl": [event_url],
        },
    ]
    return jsonify(ncco), 200


@app.route("/voice/events", methods=["POST"])
def voice_events():
    event = request.get_json(force=True, silent=True) or {}
    logger.info("Vonage event: %s", event)
    call_uuid = (
        event.get("uuid")
        or event.get("conversation_uuid")
        or event.get("call_uuid")
        or ""
    )
    if call_uuid:
        append_conversation_log(
            call_uuid,
            brand="retroworld",
            channel="phone",
            role="system",
            content=f"EVENT:{event.get('status') or ''}",
            extra={"raw": event},
        )
    return "", 204


@app.route("/voice/fallback", methods=["POST"])
def voice_fallback():
    data = request.get_json(force=True, silent=True) or {}
    logger.error("Vonage FALLBACK: %s", data)
    text = (
        "Nous rencontrons un problème temporaire avec le service automatique. "
        "Merci de rappeler un peu plus tard ou de nous contacter via le site internet."
    )
    return jsonify([{"action": "talk", "text": text, "language": "fr-FR"}]), 200


# =========================================================
# DASHBOARD ADMIN SIMPLE (type Tawk minimal)
# =========================================================

def require_admin_token() -> None:
    if not ADMIN_DASHBOARD_TOKEN:
        return
    token = request.args.get("token", "")
    if token != ADMIN_DASHBOARD_TOKEN:
        abort(403)


@app.route("/admin/conversations", methods=["GET"])
def admin_conversations():
    """
    Liste simple des conversations, style tableau HTML.
    """
    require_admin_token()
    summaries = list_conversations_summary()

    token_suffix = ""
    if ADMIN_DASHBOARD_TOKEN:
        token_suffix = f"?token={ADMIN_DASHBOARD_TOKEN}"

    lines = [
        "<!DOCTYPE html>",
        "<html lang='fr'>",
        "<head>",
        "<meta charset='UTF-8'/>",
        "<title>Conversations IA</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;background:#101016;color:#f5f5f5;padding:20px;}",
        "h1{margin-bottom:10px;}",
        "table{width:100%;border-collapse:collapse;margin-top:10px;}",
        "th,td{padding:8px 10px;border-bottom:1px solid #333;font-size:13px;}",
        "th{background:#181820;text-align:left;}",
        "tr:hover{background:#1d1d26;}",
        "a{color:#4fc3f7;text-decoration:none;}",
        "a:hover{text-decoration:underline;}",
        ".tag{padding:2px 6px;border-radius:4px;font-size:11px;}",
        ".tag-web{background:#2e7d32;}",
        ".tag-phone{background:#c62828;}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Conversations IA Retroworld / Runningman</h1>",
        f"<p>Total : {len(summaries)}</p>",
        "<table>",
        "<tr><th>ID</th><th>Canal</th><th>Marque</th><th>Début</th><th>Fin</th><th>Dernier rôle</th><th>Dernier message</th></tr>",
    ]

    for s in summaries:
        conv_id = s["conversation_id"]
        channel = s["channel"]
        brand = s.get("brand", "")
        start = s.get("start", "") or ""
        end = s.get("end", "") or ""
        last_role = s.get("last_role", "")
        snippet = (s.get("last_snippet", "") or "").replace("<", "&lt;").replace(">", "&gt;")

        tag_class = "tag-phone" if channel == "phone" else "tag-web"

        lines.append(
            "<tr>"
            f"<td><a href='/admin/conversations/{conv_id}{token_suffix}'>{conv_id}</a></td>"
            f"<td><span class='tag {tag_class}'>{channel}</span></td>"
            f"<td>{brand}</td>"
            f"<td>{start}</td>"
            f"<td>{end}</td>"
            f"<td>{last_role}</td>"
            f"<td>{snippet}</td>"
            "</tr>"
        )

    lines.extend(["</table>", "</body>", "</html>"])
    return Response("\n".join(lines), mimetype="text/html")


@app.route("/admin/conversations/<conversation_id>", methods=["GET"])
def admin_conversation_detail(conversation_id: str):
    """
    Détail d’une conversation sous forme de bulles simples.
    """
    require_admin_token()
    records, channel = load_conversation(conversation_id)
    if not records:
        return Response("<h1>Conversation introuvable</h1>", mimetype="text/html", status=404)

    brand = records[0].get("brand", "")
    lines = [
        "<!DOCTYPE html>",
        "<html lang='fr'>",
        "<head>",
        "<meta charset='UTF-8'/>",
        f"<title>Conversation {conversation_id}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;background:#101016;color:#f5f5f5;padding:20px;}",
        ".meta{font-size:13px;color:#bbb;margin-bottom:10px;}",
        ".container{display:flex;flex-direction:column;gap:4px;}",
        ".bubble{max-width:70%;padding:8px 12px;border-radius:10px;font-size:14px;white-space:pre-wrap;}",
        ".user{background:#263238;align-self:flex-start;}",
        ".assistant{background:#283593;align-self:flex-end;}",
        ".system{background:#424242;align-self:center;font-size:11px;}",
        ".ts{font-size:11px;color:#888;margin-bottom:4px;}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Conversation {conversation_id}</h1>",
        f"<div class='meta'>Canal : {channel} — Marque : {brand}</div>",
        "<div class='container'>",
    ]

    for rec in records:
        role = rec.get("role", "")
        content = rec.get("content", "") or ""
        ts = rec.get("timestamp", "")

        safe_content = content.replace("<", "&lt;").replace(">", "&gt;")

        if role == "user":
            css = "user"
        elif role == "assistant":
            css = "assistant"
        else:
            css = "system"

        if ts:
            lines.append(f"<div class='ts'>{ts} — {role}</div>")
        lines.append(f"<div class='bubble {css}'>{safe_content}</div>")

    lines.extend(["</div>", "</body>", "</html>"])
    return Response("\n".join(lines), mimetype="text/html")


# =========================================================
# MAIN LOCAL
# =========================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
