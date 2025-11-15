import os
import json
import logging
from typing import Dict, Any

from flask import Flask, request, jsonify
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
                    logger.info(f"KB {brand} chargée depuis {path} ({label})")
                    return kb
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la KB {brand} depuis {path}: {e}")

    logger.warning(f"Aucune KB trouvée pour la marque {brand}, utilisation d'une KB minimale.")
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

    logger.info(f"KB {brand} sauvegardée dans {target_path}")
    return target_path


# ---------------------------------------------------------
# OUTIL : APPEL OPENAI VIA URLLIB
# ---------------------------------------------------------

def call_openai_chat(messages, model: str = "gpt-4.1-mini", temperature: float = 0.3) -> str:
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
        logger.error(f"Erreur HTTP OpenAI: {e.code} - {error_body}")
        raise
    except urllib.error.URLError as e:
        logger.error(f"Erreur réseau OpenAI: {e.reason}")
        raise
    except Exception as e:
        logger.error(f"Erreur inconnue OpenAI: {e}")
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

    # On fournit aussi la KB brute en JSON pour que le modèle s'y réfère.
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

    return system_messages


# ---------------------------------------------------------
# ENDPOINTS
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

    # On pourrait ici ajouter une logique de logging / classification
    # (détection de devis / réservation / anniversaire) basée sur 'metadata'
    # ou sur le dernier message utilisateur.
    #
    # Pour l'instant, on se contente d'envoyer la requête à OpenAI.

    messages_for_openai = system_messages + user_messages

    try:
        assistant_reply = call_openai_chat(messages_for_openai)
    except Exception as e:
        logger.error(f"Erreur lors de l'appel OpenAI pour {brand}: {e}")
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
        logger.error(f"Erreur lors de la sauvegarde de la KB {brand}: {e}")
        return jsonify({"error": "Impossible de sauvegarder la KB."}), 500

    return jsonify({
        "status": "ok",
        "brand": brand,
        "saved_to": path_used
    }), 200


# ---------------------------------------------------------
# MAIN (pour exécution locale / debug)
# ---------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
