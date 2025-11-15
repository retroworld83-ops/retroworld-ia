import os
import json
import logging
from typing import Dict, Any, List

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

import urllib.request
import urllib.error

# ---------------------------------------------------------
# CONFIG & INITIALISATION
# ---------------------------------------------------------

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

ALLOWED_BRANDS = {"retroworld", "runningman"}

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")  # ex: https://retroworld-ia.onrender.com

LOG_DIR = "/mnt/data/logs/calls"


# ---------------------------------------------------------
# OUTILS : LECTURE / ÉCRITURE BASE DE CONNAISSANCE
# ---------------------------------------------------------

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

    # S'assurer que le dossier /mnt/data existe
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)

    logger.info("KB %s sauvegardée dans %s", brand, target_path)
    return target_path


# ---------------------------------------------------------
# OUTIL : APPEL OPENAI VIA URLLIB
# ---------------------------------------------------------

def call_openai_chat(messages: List[Dict[str, str]], model: str = "gpt-4.1-mini", temperature: float = 0.3) -> str:
    """
    Appelle l'API OpenAI /v1/chat/completions via urllib.
    Ne dépend pas du SDK officiel.
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


# ---------------------------------------------------------
# CONSTRUCTION DU CONTEXTE POUR /chat/<brand>
# ---------------------------------------------------------

def build_system_messages(brand: str, kb: Dict[str, Any], channel: str = "web") -> list:
    """
    Construit les messages de rôle 'system' envoyés à OpenAI,
    en utilisant la KB de la marque.
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

    system_style = (
        f"Style attendu : {style}. "
        f"Objectifs : {objectifs}."
    )

    kb_json = json.dumps(kb, ensure_ascii=False)

    system_messages = [
        {
            "role": "system",
            "content": system_intro
        },
        {
            "role": "system",
            "content": system_style
        },
        {
            "role": "system",
            "content": "Instructions générales : " + "\n".join(instructions_generales)
        },
        {
            "role": "system",
            "content": "Règles anti-erreurs et points d'attention : " +
                       json.dumps(anti_erreurs, ensure_ascii=False)
        },
        {
            "role": "system",
            "content": "BASE DE CONNAISSANCE (au format JSON) : " + kb_json
        },
    ]

    if channel == "phone":
        # Directive spéciale pour permettre à l'IA de déclencher des actions
        system_messages.append({
            "role": "system",
            "content": (
                "Tu es en conversation téléphonique. Les réponses doivent être courtes, simples à prononcer et naturelles. "
                "Si l'appelant demande clairement à parler à un humain, écris sur une ligne séparée exactement : "
                "[[ACTION:TRANSFER_HUMAN]]. "
                "Tu peux ensuite ajouter une phrase courte pour annoncer le transfert."
            )
        })

    return system_messages


# ---------------------------------------------------------
# OUTILS : LOG DES APPELS (TRANSCRIPTIONS TEXTE)
# ---------------------------------------------------------

def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def append_call_log(call_sid: str, brand: str, role: str, content: str, extra: Dict[str, Any] = None) -> None:
    """
    Sauvegarde chaque tour de conversation téléphonique dans un fichier .jsonl
    /mnt/data/logs/calls/<CallSid>.jsonl
    """
    ensure_log_dir()
    path = os.path.join(LOG_DIR, f"{call_sid}.jsonl")
    record = {
        "role": role,
        "brand": brand,
        "content": content,
    }
    if extra:
        record.update(extra)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_call_history(call_sid: str) -> List[Dict[str, str]]:
    """
    Charge l'historique d'un appel sous forme de messages pour le LLM.
    On convertit les logs en alternance user/assistant, ignorés pour l'instant les métadonnées.
    """
    path = os.path.join(LOG_DIR, f"{call_sid}.jsonl")
    if not os.path.exists(path):
        return []

    history: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                role = rec.get("role")
                content = rec.get("content", "")
                if role == "user":
                    history.append({"role": "user", "content": content})
                elif role == "assistant":
                    history.append({"role": "assistant", "content": content})
            except Exception as e:
                logger.error("Erreur de lecture log appel %s: %s", call_sid, e)
    return history


# ---------------------------------------------------------
# OUTIL : RÉPONSE TWIML
# ---------------------------------------------------------

def twiml_response(xml_body: str, status: int = 200) -> Response:
    """
    Construit une réponse HTTP TwiML (XML) pour Twilio.
    """
    if not xml_body.startswith("<?xml"):
        xml_body = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_body
    return Response(xml_body, status=status, mimetype="text/xml")


def detect_brand_from_number(to_number: str) -> str:
    """
    Détermine la marque en fonction du numéro appelé.
    Pour l'instant : tout vers retroworld.
    Si un jour Runningman a un numéro Twilio distinct, tu pourras router ici.
    """
    # TODO: logique plus fine si nécessaire
    return "retroworld"


def get_public_base_url() -> str:
    """
    Retourne l'URL publique de base (Render) pour construire les webhooks Twilio.
    """
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL.rstrip("/")
    # Fallback : URL racine de la requête (utile en dev/local)
    root = request.url_root.rstrip("/")
    return root


# ---------------------------------------------------------
# ENDPOINTS GÉNÉRAUX (HEALTH, CHAT, KB)
# ---------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """
    Endpoint de santé / monitoring.
    Permet à Render ou à une sonde externe de vérifier que le service répond.
    """
    return jsonify({"status": "ok"}), 200


@app.route("/chat/<brand>", methods=["POST"])
def chat_brand(brand: str):
    """
    Endpoint principal de chat :
    - brand : 'retroworld' ou 'runningman' (extensible)
    - body JSON attendu :
      {
        "messages": [
          {"role": "user", "content": "Bonjour..."},
          {"role": "assistant", "content": "..."}  // historique optionnel
        ],
        "channel": "web" | "phone" | "whatsapp" | ... (optionnel),
        "metadata": {...}                           (optionnel)
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
    channel = body.get("channel", "web")
    metadata = body.get("metadata", {})

    if not user_messages or not isinstance(user_messages, list):
        return jsonify({"error": "Le champ 'messages' (liste) est requis."}), 400

    kb = load_kb(brand)
    system_messages = build_system_messages(brand, kb, channel=channel)

    messages_for_openai = system_messages + user_messages

    try:
        assistant_reply = call_openai_chat(messages_for_openai)
    except Exception as e:
        logger.error("Erreur lors de l'appel OpenAI pour %s: %s", brand, e)
        return jsonify({"error": "Erreur interne lors de la génération de la réponse."}), 500

    response_payload = {
        "reply": assistant_reply,
        "brand": brand,
        "metadata_echo": metadata,
    }

    return jsonify(response_payload), 200


@app.route("/kb/upsert/<brand>", methods=["POST"])
def kb_upsert_brand(brand: str):
    """
    Endpoint pour mettre à jour la KB d'une marque.
    - brand : 'retroworld', 'runningman', etc.
    - body JSON : la KB complète à enregistrer.
    -> Sauvegarde dans /mnt/data/kb_<brand>.json
    """
    brand = brand.lower()
    if brand not in ALLOWED_BRANDS:
        return jsonify({"error": "Marque inconnue.", "allowed_brands": list(ALLOWED_BRANDS)}), 400

    try:
        kb_data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Corps de requête JSON invalide."}), 400

    if not isinstance(kb_data, dict):
        return jsonify({"error": "La KB doit être un objet JSON (dict)."}), 400

    try:
        path_used = save_kb(brand, kb_data)
    except Exception as e:
        logger.error("Erreur lors de la sauvegarde de la KB %s: %s", brand, e)
        return jsonify({"error": "Impossible de sauvegarder la KB."}), 500

    return jsonify({
        "status": "ok",
        "brand": brand,
        "saved_to": path_used
    }), 200


# ---------------------------------------------------------
# PARTIE TÉLÉPHONIE (TWILIO)
# ---------------------------------------------------------

@app.route("/call/incoming", methods=["POST"])
def call_incoming():
    """
    Webhook Twilio : un appel arrive sur le numéro virtuel.
    Twilio envoie une requête POST (application/x-www-form-urlencoded) ici.
    On répond avec du TwiML pour accueillir l'appelant et lancer la première question.
    """
    from_number = request.form.get("From", "")
    to_number = request.form.get("To", "")
    call_sid = request.form.get("CallSid", "")

    brand = detect_brand_from_number(to_number)  # pour l'instant : retroworld
    base_url = get_public_base_url()
    action_url = f"{base_url}/call/assistant?brand={brand}"

    logger.info("Nouvel appel entrant: CallSid=%s, From=%s, To=%s, brand=%s", call_sid, from_number, to_number, brand)

    # On log juste l'arrivée de l'appel
    append_call_log(call_sid, brand, role="system", content="CALL_STARTED", extra={
        "from": from_number,
        "to": to_number
    })

    greeting = (
        "Bonjour, vous êtes bien chez Retroworld à Draguignan. "
        "Je suis l'assistant vocal automatique. "
        "Pouvez-vous m'indiquer en quelques mots ce que vous souhaitez ? "
        "Par exemple : réserver une partie de VR, organiser un anniversaire, ou avoir des informations."
    )

    xml = f"""
<Response>
  <Gather input="speech" language="fr-FR" action="{action_url}" method="POST" timeout="6">
    <Say voice="woman">{greeting}</Say>
  </Gather>
  <Say voice="woman">Je n'ai pas entendu votre réponse. Je vais raccrocher, n'hésitez pas à nous rappeler.</Say>
  <Hangup/>
</Response>
"""
    return twiml_response(xml)


@app.route("/call/assistant", methods=["POST"])
def call_assistant():
    """
    Webhook Twilio appelé après chaque prise de parole de l'appelant.
    On reçoit le texte transcrit par Twilio (SpeechResult), on l'envoie à l'IA,
    puis on répond avec du TwiML <Say> + éventuellement un nouveau <Gather> pour continuer la conversation.
    """
    call_sid = request.form.get("CallSid", "")
    from_number = request.form.get("From", "")
    to_number = request.form.get("To", "")
    speech_text = (request.form.get("SpeechResult") or "").strip()

    brand = request.args.get("brand", "retroworld").lower()
    if brand not in ALLOWED_BRANDS:
        brand = "retroworld"

    base_url = get_public_base_url()
    action_url = f"{base_url}/call/assistant?brand={brand}"

    if not speech_text:
        # Rien compris / pas de texte -> redemander
        logger.info("Aucune parole comprise pour CallSid=%s", call_sid)
        xml = f"""
<Response>
  <Gather input="speech" language="fr-FR" action="{action_url}" method="POST" timeout="6">
    <Say voice="woman">Je n'ai pas bien compris. Pouvez-vous répéter votre demande ?</Say>
  </Gather>
  <Say voice="woman">Je n'ai toujours pas compris, je vais raccrocher pour le moment.</Say>
  <Hangup/>
</Response>
"""
        return twiml_response(xml)

    logger.info("CallSid=%s | Parole reconnue: %s", call_sid, speech_text)
    append_call_log(call_sid, brand, role="user", content=speech_text, extra={
        "from": from_number,
        "to": to_number
    })

    # Construire le contexte de conversation pour l'IA
    kb = load_kb(brand)
    system_messages = build_system_messages(brand, kb, channel="phone")
    history = load_call_history(call_sid)

    messages_for_openai: List[Dict[str, str]] = system_messages + history + [
        {"role": "user", "content": speech_text}
    ]

    try:
        assistant_reply = call_openai_chat(messages_for_openai, temperature=0.4)
    except Exception as e:
        logger.error("Erreur IA pendant l'appel CallSid=%s: %s", call_sid, e)
        error_msg = (
            "Je rencontre un souci technique pour le moment. "
            "Je vous invite à rappeler un peu plus tard ou à nous contacter par notre site Retroworld France."
        )
        xml = f"""
<Response>
  <Say voice="woman">{error_msg}</Say>
  <Hangup/>
</Response>
"""
        return twiml_response(xml, status=500)

    append_call_log(call_sid, brand, role="assistant", content=assistant_reply, extra={})

    # Vérifier si l'IA demande un transfert vers un humain
    transfer_requested = "[[ACTION:TRANSFER_HUMAN]]" in assistant_reply

    # Nettoyer la réponse vocale (on enlève le tag d'action si présent)
    spoken_reply = assistant_reply.replace("[[ACTION:TRANSFER_HUMAN]]", "").strip()

    if transfer_requested:
        # On transfère vers un humain (par défaut le fixe Retroworld)
        human_number = "+33494479464"  # format E.164 recommandé (à adapter)
        logger.info("CallSid=%s | Transfert vers un humain: %s", call_sid, human_number)

        # On annonce puis on compose
        xml = f"""
<Response>
  <Say voice="woman">{spoken_reply or "Je vous transfère vers un membre de l'équipe."}</Say>
  <Dial>{human_number}</Dial>
</Response>
"""
        return twiml_response(xml)

    # Sinon, on continue la conversation IA avec un nouveau Gather
    # On garde la réponse courte pour éviter un monologue trop long en TTS
    if len(spoken_reply) > 800:
        spoken_reply = spoken_reply[:800]

    xml = f"""
<Response>
  <Gather input="speech" language="fr-FR" action="{action_url}" method="POST" timeout="8">
    <Say voice="woman">{spoken_reply}</Say>
  </Gather>
  <Say voice="woman">Je n'ai pas entendu de réponse, je mets fin à l'appel pour le moment. À bientôt chez Retroworld.</Say>
  <Hangup/>
</Response>
"""
    return twiml_response(xml)


@app.route("/call/recording-callback", methods=["POST"])
def call_recording_callback():
    """
    Callback optionnel Twilio si tu décides d'enregistrer les appels ou d'activer la transcription Twilio.
    Tu pourras y récupérer recordingUrl, transcriptionText, etc., pour stockage complémentaire.
    """
    call_sid = request.form.get("CallSid", "")
    recording_url = request.form.get("RecordingUrl", "")
    transcription_text = request.form.get("TranscriptionText", "")

    extra = {
        "recording_url": recording_url,
        "transcription_text": transcription_text,
    }
    append_call_log(call_sid, brand="retroworld", role="system", content="RECORDING_INFO", extra=extra)

    logger.info("Recording callback pour CallSid=%s | recording_url=%s", call_sid, recording_url)
    return "", 204


# ---------------------------------------------------------
# MAIN (pour exécution locale / debug)
# ---------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
