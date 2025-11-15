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

# ---------------------------------------------------------
# CONFIG GLOBALE
# ---------------------------------------------------------

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retroworld-ia")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# URL publique de ce service (Render)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

# Marques gérées
ALLOWED_BRANDS = {"retroworld", "runningman"}

# Dossiers de logs
BASE_LOG_DIR = "/mnt/data/logs"
CHAT_LOG_DIR = os.path.join(BASE_LOG_DIR, "chat")
CALL_LOG_DIR = os.path.join(BASE_LOG_DIR, "calls")

# Dashboard admin (type tawk.to) – token simple en query param
ADMIN_DASHBOARD_TOKEN = os.getenv("ADMIN_DASHBOARD_TOKEN") or os.getenv("ADMIN_API_TOKEN", "")

# Numéros Vonage pour détection de marque (format E.164, ex : +322XXXXXXX)
VONAGE_NUMBER_RETROWORLD = os.getenv("VONAGE_NUMBER_RETROWORLD", "").replace(" ", "")
VONAGE_NUMBER_RUNNINGMAN = os.getenv("VONAGE_NUMBER_RUNNINGMAN", "").replace(" ", "")

# Numéros vers lesquels transférer vers humain
FORWARD_NUMBER_RETROWORLD = os.getenv("FORWARD_NUMBER_RETROWORLD", "+33494479464")
FORWARD_NUMBER_RUNNINGMAN = os.getenv("FORWARD_NUMBER_RUNNINGMAN", "+33498093059")

# ---------------------------------------------------------
# OUTILS GÉNÉRIQUES
# ---------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_kb_paths(brand: str) -> Dict[str, str]:
    """
    Retourne les chemins possibles pour la KB d'une marque.
    Priorité à /mnt/data, fallback sur /app.
    """
    filename = f"kb_{brand}.json"
    return {
        "mnt": os.path.join("/mnt/data", filename),
        "app": os.path.join("/app", filename),
    }


def load_kb(brand: str) -> Dict[str, Any]:
    """
    Charge la base de connaissance pour une marque donnée.
    1) Essaie /mnt/data/kb_<brand>.json
    2) Sinon /app/kb_<brand>.json
    3) Sinon renvoie un dict minimal.
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
            logger.error("Erreur lors du chargement de la KB %s depuis %s: %s", brand, path, e)

    logger.warning("Aucune KB trouvée pour la marque %s, utilisation d'une KB minimale.", brand)
    return {
        "identite": {
            "nom": brand,
            "description": f"KB minimale pour {brand}.",
        },
        "prompt": {
            "instructions_generales": [
                "Tu es un assistant IA pour cette marque.",
                "Réponds de manière professionnelle et claire.",
            ]
        }
    }


def save_kb(brand: str, kb_data: Dict[str, Any]) -> str:
    """
    Sauvegarde la KB dans /mnt/data/kb_<brand>.json (créé si besoin).
    Renvoie le chemin utilisé.
    """
    paths = get_kb_paths(brand)
    target_path = paths["mnt"]

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)

    logger.info("KB %s sauvegardée dans %s", brand, target_path)
    return target_path


def call_openai_chat(
    messages: List[Dict[str, str]],
    model: str = OPENAI_MODEL_DEFAULT,
    temperature: float = 0.3,
) -> str:
    """
    Appelle l'API OpenAI /v1/chat/completions via urllib.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY non défini dans les variables d'environnement.")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

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
            resp_data = resp.read().decode("utf-8")
            resp_json = json.loads(resp_data)
            return resp_json["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="ignore")
        logger.error("Erreur HTTP OpenAI: %s - %s", e.code, error_body)
        raise
    except urllib.error.URLError as e:
        logger.error("Erreur réseau OpenAI: %s", e.reason)
        raise
    except Exception as e:
        logger.error("Erreur inconnue OpenAI: %s", e)
        raise


def build_system_messages(brand: str, kb: Dict[str, Any], channel: str = "web") -> List[Dict[str, str]]:
    """
    Construit les messages 'system' pour OpenAI, en utilisant la KB.
    """
    identite = kb.get("identite", {})
    prompt = kb.get("prompt", {})
    anti_erreurs = kb.get("anti_erreurs", {})

    nom = identite.get("nom", brand)
    description = identite.get("description", "")
    style = prompt.get("style", "")
    objectifs = prompt.get("objectifs", "")
    instructions_generales = prompt.get("instructions_generales", [])

    system_intro = (
        f"Tu es l'assistant IA officiel de {nom}. "
        f"Description : {description}. "
        f"Canal de conversation : {channel}. "
        "Tu dois toujours respecter strictement la base de connaissance fournie."
    )

    system_style = f"Style attendu : {style}. Objectifs : {objectifs}."

    kb_json = json.dumps(kb, ensure_ascii=False)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_intro},
        {"role": "system", "content": system_style},
        {"role": "system", "content": "Instructions générales :\n" + "\n".join(instructions_generales)},
        {"role": "system", "content": "Règles anti-erreurs : " + json.dumps(anti_erreurs, ensure_ascii=False)},
        {"role": "system", "content": "BASE DE CONNAISSANCE JSON : " + kb_json},
    ]

    if channel == "phone":
        messages.append({
            "role": "system",
            "content": (
                "Tu es en conversation téléphonique. Utilise des phrases courtes, claires, faciles à prononcer. "
                "Ne lis pas les URLs à voix haute. Si tu dois envoyer un lien (paiement, réservation), "
                "explique simplement que le client va le recevoir par SMS ou email. "
                "Si l'appelant demande explicitement à parler à un humain, écris sur une ligne séparée exactement : "
                "[[ACTION:TRANSFER_HUMAN]]."
            )
        })

    return messages

# ---------------------------------------------------------
# LOGGING CONVERSATIONS (WEB + VOCAL)
# ---------------------------------------------------------


def get_conversation_log_path(conversation_id: str, channel: str) -> str:
    """
    Retourne le chemin du fichier de log pour une conversation.
    channel: 'web' ou 'phone'
    """
    ensure_dir(BASE_LOG_DIR)
    if channel == "phone":
        ensure_dir(CALL_LOG_DIR)
        return os.path.join(CALL_LOG_DIR, f"{conversation_id}.jsonl")
    else:
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
    """
    Ajoute une ligne JSONL à la conversation (web ou phone).
    """
    path = get_conversation_log_path(conversation_id, channel)
    record: Dict[str, Any] = {
        "timestamp": now_iso(),
        "conversation_id": conversation_id,
        "brand": brand,
        "channel": channel,
        "role": role,
        "content": content,
    }
    if extra:
        record.update(extra)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_conversation(conversation_id: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Charge une conversation (web ou téléphone).
    Renvoie (records, channel).
    """
    # On cherche d'abord côté chat, puis côté calls
    for channel in ("web", "phone"):
        path = get_conversation_log_path(conversation_id, channel)
        if os.path.exists(path):
            records: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
            return records, channel
    return [], ""


def list_conversations_summary() -> List[Dict[str, Any]]:
    """
    Parcourt les dossiers de logs et renvoie un résumé de chaque conversation.
    """
    summaries: List[Dict[str, Any]] = []

    for channel, base_dir in (("web", CHAT_LOG_DIR), ("phone", CALL_LOG_DIR)):
        if not os.path.isdir(base_dir):
            continue
        for filename in os.listdir(base_dir):
            if not filename.endswith(".jsonl"):
                continue
            path = os.path.join(base_dir, filename)
            conv_id = filename[:-6]
            first: Optional[Dict[str, Any]] = None
            last: Optional[Dict[str, Any]] = None
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

            summaries.append({
                "conversation_id": conv_id,
                "channel": channel,
                "brand": first.get("brand", ""),
                "start": first.get("timestamp"),
                "end": last.get("timestamp"),
                "last_role": last.get("role"),
                "last_snippet": (last.get("content") or "")[:120],
            })

    # Tri par date de fin décroissante
    summaries.sort(key=lambda s: s.get("end") or "", reverse=True)
    return summaries


# ---------------------------------------------------------
# DÉTECTION MARQUE POUR LES APPELS (VONAGE)
# ---------------------------------------------------------


def detect_brand_from_to_number(to_number: str) -> str:
    """
    Détecte la marque en fonction du numéro appelé (Vonage).
    """
    num = (to_number or "").replace(" ", "")
    if VONAGE_NUMBER_RETROWORLD and num == VONAGE_NUMBER_RETROWORLD:
        return "retroworld"
    if VONAGE_NUMBER_RUNNINGMAN and num == VONAGE_NUMBER_RUNNINGMAN:
        return "runningman"
    # défaut : Retroworld
    return "retroworld"


def get_forward_number_for_brand(brand: str) -> str:
    """
    Numéro humain vers lequel transférer l'appel si nécessaire.
    """
    if brand == "runningman":
        return FORWARD_NUMBER_RUNNINGMAN
    return FORWARD_NUMBER_RETROWORLD

# ---------------------------------------------------------
# ENDPOINTS GÉNÉRAUX
# ---------------------------------------------------------


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# ---------------------------------------------------------
# CHAT WEB : /chat/<brand>
# ---------------------------------------------------------


@app.route("/chat/<brand>", methods=["POST"])
def chat_brand(brand: str):
    """
    Endpoint principal de chat web :
    body JSON attendu :
    {
      "messages": [ {"role": "user","content": "..."} , ... ],
      "channel": "web" | "widget" | "phone" ... (optionnel, défaut: "web"),
      "metadata": { "conversation_id": "...", ... } (optionnel)
    }
    """
    brand = brand.lower()
    if brand not in ALLOWED_BRANDS:
        return jsonify({"error": "Marque inconnue.", "allowed_brands": list(ALLOWED_BRANDS)}), 400

    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Corps de requête JSON invalide."}), 400

    if not isinstance(body, dict):
        return jsonify({"error": "Format de requête incorrect."}), 400

    user_messages = body.get("messages")
    channel = body.get("channel", "web") or "web"
    metadata = body.get("metadata") or {}

    if not user_messages or not isinstance(user_messages, list):
        return jsonify({"error": "Le champ 'messages' (liste) est requis."}), 400

    # Conversation ID (pour regrouper les messages type tawk.to)
    conversation_id = metadata.get("conversation_id") or str(uuid.uuid4())
    metadata["conversation_id"] = conversation_id

    kb = load_kb(brand)
    system_messages = build_system_messages(brand, kb, channel="web")

    messages_for_openai = system_messages + user_messages

    try:
        assistant_reply = call_openai_chat(messages_for_openai)
    except Exception as e:
        logger.error("Erreur OpenAI chat %s: %s", brand, e)
        return jsonify({"error": "Erreur interne lors de la génération de la réponse."}), 500

    # LOGGING : on log uniquement le dernier message utilisateur + la réponse
    last_user_msg = None
    for msg in reversed(user_messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    if last_user_msg:
        append_conversation_log(
            conversation_id,
            brand=brand,
            channel="web",
            role="user",
            content=last_user_msg,
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

    response_payload = {
        "reply": assistant_reply,
        "brand": brand,
        "metadata": metadata,
        "conversation_id": conversation_id,
    }
    return jsonify(response_payload), 200

# ---------------------------------------------------------
# KB UPSERT
# ---------------------------------------------------------


@app.route("/kb/upsert/<brand>", methods=["POST"])
def kb_upsert_brand(brand: str):
    brand = brand.lower()
    if brand not in ALLOWED_BRANDS:
        return jsonify({"error": "Marque inconnue.", "allowed_brands": list(ALLOWED_BRANDS)}), 400

    try:
        kb_data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Corps de requête JSON invalide."}), 400

    if not isinstance(kb_data, dict):
        return jsonify({"error": "La KB doit être un objet JSON."}), 400

    try:
        path_used = save_kb(brand, kb_data)
    except Exception as e:
        logger.error("Erreur sauvegarde KB %s: %s", brand, e)
        return jsonify({"error": "Impossible de sauvegarder la KB."}), 500

    return jsonify({"status": "ok", "brand": brand, "saved_to": path_used}), 200

# ---------------------------------------------------------
# PARTIE TÉLÉPHONIE VONAGE (VOIX / IA)
# ---------------------------------------------------------


@app.route("/voice/incoming", methods=["POST"])
def voice_incoming():
    """
    Answer URL Vonage.
    Vonage envoie un JSON à chaque appel entrant.
    On répond avec une NCCO (liste d'actions JSON).
    """
    data = request.get_json(force=True, silent=True) or {}
    logger.info("Vonage incoming: %s", data)

    to_num = data.get("to") or data.get("to_number", "")
    from_num = data.get("from") or data.get("from_number", "")
    call_uuid = data.get("uuid") or data.get("call_uuid") or str(uuid.uuid4())

    brand = detect_brand_from_to_number(str(to_num))
    kb = load_kb(brand)
    identite = kb.get("identite", {})
    nom = identite.get("nom", brand.capitalize())

    append_conversation_log(
        conversation_id=call_uuid,
        brand=brand,
        channel="phone",
        role="system",
        content="CALL_STARTED",
        extra={"from": from_num, "to": to_num},
    )

    if not PUBLIC_BASE_URL:
        # fallback local (dev)
        base_url = request.url_root.rstrip("/")
    else:
        base_url = PUBLIC_BASE_URL

    event_url = f"{base_url}/voice/assistant?brand={brand}"

    greeting = (
        f"Bonjour, vous êtes bien chez {nom}. "
        "Je suis l'assistant vocal. Pouvez-vous me dire en quelques mots ce que vous souhaitez ? "
        "Par exemple : réserver une partie, organiser un anniversaire ou avoir des informations."
    )

    ncco = [
        {
            "action": "talk",
            "text": greeting,
            "language": "fr-FR"
        },
        {
            "action": "input",
            "type": ["speech"],
            "speech": {
                "language": "fr-FR",
                "endOnSilence": 1
            },
            "eventUrl": [event_url]
        }
    ]

    return jsonify(ncco), 200


@app.route("/voice/assistant", methods=["POST"])
def voice_assistant():
    """
    Event URL pour les résultats de reconnaissance vocale (Vonage input).
    On reçoit le texte de l'appelant, on appelle OpenAI,
    puis on renvoie une nouvelle NCCO (talk + input ou transfert).
    """
    event = request.get_json(force=True, silent=True) or {}
    logger.info("Vonage assistant event: %s", event)

    brand = request.args.get("brand", "retroworld").lower()
    if brand not in ALLOWED_BRANDS:
        brand = "retroworld"

    # Récup info appel
    call_uuid = event.get("uuid") or event.get("call_uuid") or event.get("conversation_uuid") or str(uuid.uuid4())
    from_num = event.get("from") or ""
    to_num = event.get("to") or ""

    # Texte reconnu
    speech = event.get("speech") or {}
    results = speech.get("results") or []
    transcript = ""
    if results and isinstance(results, list):
        transcript = results[0].get("text", "") or ""

    if not transcript.strip():
        # rien compris, redemander
        if not PUBLIC_BASE_URL:
            base_url = request.url_root.rstrip("/")
        else:
            base_url = PUBLIC_BASE_URL
        event_url = f"{base_url}/voice/assistant?brand={brand}"

        ncco = [
            {
                "action": "talk",
                "text": "Je n'ai pas bien compris. Pouvez-vous répéter, s'il vous plaît ?",
                "language": "fr-FR"
            },
            {
                "action": "input",
                "type": ["speech"],
                "speech": {"language": "fr-FR", "endOnSilence": 1},
                "eventUrl": [event_url]
            }
        ]
        return jsonify(ncco), 200

    # LOG utilisateur
    append_conversation_log(
        conversation_id=call_uuid,
        brand=brand,
        channel="phone",
        role="user",
        content=transcript,
        extra={"from": from_num, "to": to_num}
    )

    # Construit le contexte IA
    kb = load_kb(brand)
    system_messages = build_system_messages(brand, kb, channel="phone")

    # Recharger un peu d'historique pour la mémoire
    history_records, _ = load_conversation(call_uuid)
    history_messages: List[Dict[str, str]] = []
    for rec in history_records:
        if rec.get("channel") != "phone":
            continue
        role = rec.get("role")
        content = rec.get("content", "")
        if role == "user":
            history_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            history_messages.append({"role": "assistant", "content": content})

    messages_for_openai: List[Dict[str, str]] = system_messages + history_messages + [
        {"role": "user", "content": transcript}
    ]

    try:
        assistant_reply = call_openai_chat(messages_for_openai, temperature=0.4)
    except Exception as e:
        logger.error("Erreur IA pendant appel %s: %s", call_uuid, e)
        fallback_text = (
            "Je rencontre un problème technique. "
            "Je vous invite à rappeler un peu plus tard ou à nous contacter via le site Retroworld France."
        )
        append_conversation_log(
            conversation_id=call_uuid,
            brand=brand,
            channel="phone",
            role="assistant",
            content=fallback_text,
            extra={"error": str(e)}
        )
        ncco = [{"action": "talk", "text": fallback_text, "language": "fr-FR"}]
        return jsonify(ncco), 200

    append_conversation_log(
        conversation_id=call_uuid,
        brand=brand,
        channel="phone",
        role="assistant",
        content=assistant_reply,
        extra={}
    )

    # Détection transfert humain
    transfer_requested = "[[ACTION:TRANSFER_HUMAN]]" in assistant_reply
    spoken_reply = assistant_reply.replace("[[ACTION:TRANSFER_HUMAN]]", "").strip()
    if len(spoken_reply) > 800:
        spoken_reply = spoken_reply[:800]

    if not PUBLIC_BASE_URL:
        base_url = request.url_root.rstrip("/")
    else:
        base_url = PUBLIC_BASE_URL
    event_url = f"{base_url}/voice/assistant?brand={brand}"

    if transfer_requested:
        human_number = get_forward_number_for_brand(brand)
        ncco = [
            {
                "action": "talk",
                "text": spoken_reply or "Je vous mets en relation avec un membre de l'équipe.",
                "language": "fr-FR",
            },
            {
                "action": "connect",
                "endpoint": [
                    {
                        "type": "phone",
                        "number": human_number
                    }
                ]
            }
        ]
        return jsonify(ncco), 200

    # Sinon, on continue la conversation IA
    ncco = [
        {
            "action": "talk",
            "text": spoken_reply,
            "language": "fr-FR",
        },
        {
            "action": "input",
            "type": ["speech"],
            "speech": {"language": "fr-FR", "endOnSilence": 1},
            "eventUrl": [event_url]
        }
    ]
    return jsonify(ncco), 200


@app.route("/voice/events", methods=["POST"])
def voice_events():
    """
    Event URL Vonage générique (status, ringing, completed...).
    On loggue seulement.
    """
    event = request.get_json(force=True, silent=True) or {}
    logger.info("Vonage event: %s", event)
    call_uuid = event.get("uuid") or event.get("call_uuid") or ""
    status = event.get("status") or event.get("conversation_status") or ""

    if call_uuid:
        append_conversation_log(
            conversation_id=call_uuid,
            brand="retroworld",  # inconnu -> default
            channel="phone",
            role="system",
            content=f"EVENT:{status}",
            extra={"raw": event}
        )
    return "", 204


@app.route("/voice/fallback", methods=["POST"])
def voice_fallback():
    """
    Fallback Vonage si l'Answer URL ne répond pas.
    """
    data = request.get_json(force=True, silent=True) or {}
    logger.error("Vonage FALLBACK triggered: %s", data)
    text = (
        "Nous rencontrons un problème temporaire avec le service automatique. "
        "Nous vous invitons à rappeler un peu plus tard ou à nous contacter via le site internet."
    )
    ncco = [{"action": "talk", "text": text, "language": "fr-FR"}]
    return jsonify(ncco), 200

# ---------------------------------------------------------
# DASHBOARD ADMIN TYPE TAWK.TO
# ---------------------------------------------------------


def require_admin_token() -> None:
    if not ADMIN_DASHBOARD_TOKEN:
        return  # aucune protection si pas de token défini (à éviter en prod)
    token = request.args.get("token", "")
    if token != ADMIN_DASHBOARD_TOKEN:
        abort(403)


@app.route("/admin/conversations", methods=["GET"])
def admin_conversations_list():
    """
    Dashboard simple listant toutes les conversations (web + phone),
    similaire à une vue tawk.to basique.
    Protection simple par token en query param: ?token=...
    """
    require_admin_token()
    summaries = list_conversations_summary()

    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='fr'>",
        "<head>",
        "<meta charset='UTF-8'/>",
        "<title>Conversations - Retroworld IA</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; background:#111; color:#f3f3f3; padding:20px; }",
        "a { color:#4fc3f7; text-decoration:none; }",
        "a:hover { text-decoration:underline; }",
        "table { width:100%; border-collapse:collapse; margin-top:10px; }",
        "th, td { padding:8px 10px; border-bottom:1px solid #333; font-size:14px; }",
        "th { background:#222; text-align:left; }",
        "tr:hover { background:#1b1b1b; }",
        ".tag { display:inline-block; padding:2px 6px; border-radius:4px; font-size:11px; margin-right:4px; }",
        ".tag-web { background:#2e7d32; }",
        ".tag-phone { background:#c62828; }",
        ".tag-brand { background:#1565c0; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Conversations IA Retroworld / Runningman</h1>",
        "<p>Total : %d</p>" % len(summaries),
        "<table>",
        "<tr><th>ID</th><th>Canal</th><th>Marque</th><th>Début</th><th>Fin</th><th>Dernier rôle</th><th>Dernier message</th></tr>",
    ]

    token_param = ""
    if ADMIN_DASHBOARD_TOKEN:
        token_param = f"?token={ADMIN_DASHBOARD_TOKEN}"

    for s in summaries:
        conv_id = s["conversation_id"]
        channel = s["channel"]
        brand = s.get("brand", "")
        start = s.get("start", "") or ""
        end = s.get("end", "") or ""
        last_role = s.get("last_role", "")
        snippet = (s.get("last_snippet", "") or "").replace("<", "&lt;").replace(">", "&gt;")

        channel_tag_class = "tag-phone" if channel == "phone" else "tag-web"
        brand_tag = f"<span class='tag tag-brand'>{brand}</span>" if brand else ""

        html_lines.append(
            "<tr>"
            f"<td><a href='/admin/conversations/{conv_id}{token_param}'>{conv_id}</a></td>"
            f"<td><span class='tag {channel_tag_class}'>{channel}</span></td>"
            f"<td>{brand_tag}</td>"
            f"<td>{start}</td>"
            f"<td>{end}</td>"
            f"<td>{last_role}</td>"
            f"<td>{snippet}</td>"
            "</tr>"
        )

    html_lines.extend([
        "</table>",
        "</body>",
        "</html>"
    ])

    return Response("\n".join(html_lines), mimetype="text/html")


@app.route("/admin/conversations/<conversation_id>", methods=["GET"])
def admin_conversation_detail(conversation_id: str):
    """
    Détail d'une conversation : messages dans l'ordre, style bulles.
    """
    require_admin_token()
    records, channel = load_conversation(conversation_id)
    if not records:
        return Response("<h1>Conversation introuvable</h1>", mimetype="text/html", status=404)

    brand = records[0].get("brand", "")
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='fr'>",
        "<head>",
        "<meta charset='UTF-8'/>",
        f"<title>Conversation {conversation_id}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; background:#111; color:#f3f3f3; padding:20px; }",
        ".meta { font-size:13px; color:#aaa; margin-bottom:10px; }",
        ".bubble { max-width:70%; padding:8px 12px; border-radius:12px; margin:6px 0; font-size:14px; white-space:pre-wrap; }",
        ".user { background:#263238; align-self:flex-start; }",
        ".assistant { background:#283593; align-self:flex-end; }",
        ".system { background:#424242; align-self:center; font-size:12px; }",
        ".container { display:flex; flex-direction:column; }",
        ".timestamp { font-size:11px; color:#888; margin-top:2px; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Conversation {conversation_id}</h1>",
        f"<div class='meta'>Canal : {channel} &nbsp;&nbsp; Marque : {brand}</div>",
        "<div class='container'>",
    ]

    for rec in records:
        role = rec.get("role", "")
        content = rec.get("content", "") or ""
        ts = rec.get("timestamp", "")
        safe_content = content.replace("<", "&lt;").replace(">", "&gt;")
        css_class = "system"
        if role == "user":
            css_class = "user"
        elif role == "assistant":
            css_class = "assistant"

        html_lines.append(f"<div class='bubble {css_class}'>{safe_content}</div>")
        if ts:
            html_lines.append(f"<div class='timestamp'>{ts} — {role}</div>")

    html_lines.extend([
        "</div>",
        "</body>",
        "</html>"
    ])

    return Response("\n".join(html_lines), mimetype="text/html")


# ---------------------------------------------------------
# MAIN (LOCAL)
# ---------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
