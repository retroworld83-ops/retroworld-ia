import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from flask import Flask, request, jsonify
from flask_cors import CORS

import urllib.request
import urllib.error

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

app = Flask(__name__)
CORS(app)

DEBUG_LOGS = True

DEFAULT_BRAND = os.environ.get("DEFAULT_BRAND", "retroworld").lower()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

DATA_DIR = Path("/mnt/data")           # stockage persistant Render
APP_DIR = Path(__file__).resolve().parent  # répertoire de l'image Docker (/app)


# ----------------------------------------------------
# UTIL / KB
# ----------------------------------------------------

def safe_brand(b: str) -> str:
    if not b:
        return DEFAULT_BRAND
    return b.lower().strip()


def kb_path(brand: str) -> Path:
    """Chemin priorité: stockage persistant (/mnt/data)."""
    return DATA_DIR / f"kb_{brand}.json"


def ensure_data_dir():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


DEFAULT_KB: Dict[str, Any] = {
    "identite": {"nom": "Entreprise"},
    "prompt": "Tu es un assistant d'information. La base de connaissance est vide.",
    "faqs": []
}


def load_kb(brand: str) -> Dict[str, Any]:
    """
    Charge la KB d'une marque :
    1) /mnt/data/kb_<brand>.json (priorité)
    2) /app/kb_<brand>.json (copié par Dockerfile)
    """
    ensure_data_dir()
    b = safe_brand(brand)

    candidates = [
        DATA_DIR / f"kb_{b}.json",
        APP_DIR / f"kb_{b}.json",
    ]

    for p in candidates:
        try:
            if p.is_file():
                if DEBUG_LOGS:
                    print(f"[KB] Chargement depuis : {p}")
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            if DEBUG_LOGS:
                print(f"[KB] Erreur lecture {p}: {e}")
            continue

    return DEFAULT_KB.copy()


def save_kb(brand: str, kb: Dict[str, Any]) -> None:
    """Sauvegarde la KB dans /mnt/data uniquement (pour les updates dynamiques)."""
    ensure_data_dir()
    b = safe_brand(brand)
    dest = DATA_DIR / f"kb_{b}.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    if DEBUG_LOGS:
        print(f"[KB] Sauvegardé dans {dest}")


def count_faqs(kb: Dict[str, Any]) -> int:
    if "faqs" in kb and isinstance(kb["faqs"], list):
        return len(kb["faqs"])
    if "faq" in kb and isinstance(kb["faq"], list):
        return len(kb["faq"])
    return 0


# ----------------------------------------------------
# OpenAI via HTTP (pas de dépendance openai)
# ----------------------------------------------------

def call_openai(messages: List[Dict[str, str]]) -> str:
    """Appel basique à l'API OpenAI Chat Completions."""
    if not OPENAI_API_KEY:
        return "Erreur interne : la clé OPENAI_API_KEY n'est pas configurée sur le serveur."

    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 700,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            result = json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        print("[OpenAI HTTPError]", e, err_body)
        return "Désolé, une erreur technique est survenue avec le service d'IA."
    except Exception as e:
        print("[OpenAI ERROR]", e)
        return "Désolé, une erreur technique est survenue avec le service d'IA."

    try:
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("[OpenAI PARSE ERROR]", e, result)
        return "Désolé, je n'ai pas pu générer de réponse pour le moment."


def build_system_prompt(kb: Dict[str, Any], brand: str) -> str:
    identite = kb.get("identite", {})
    nom = identite.get("nom", "l'établissement")
    site = identite.get("site", "")
    tel = identite.get("telephone", "")
    horaires = identite.get("horaires", "")

    kb_prompt = kb.get("prompt", "")

    header = f"""
Tu es l'assistant officiel de {nom}.
Tu réponds en français, avec un ton professionnel, chaleureux et clair.

Coordonnées connues :
- Téléphone : {tel}
- Site : {site}
- Horaires : {horaires}

Tu dois t'appuyer UNIQUEMENT sur les informations présentes dans la base de connaissance JSON suivante.
Si une information n'est pas disponible ou pas claire, tu le dis honnêtement et tu proposes de mettre le client en contact avec un agent humain.
"""

    if safe_brand(brand) == "retroworld":
        brand_rules = """
Spécifique Retroworld :
- Si la personne parle de réservation, devis ou anniversaire :
  tu demandes toujours : activité, date, heure précise (ou fourchette), nombre de joueurs, nom, e-mail, téléphone.
- Tu ne donnes les liens de réservation QUE si la personne est clairement décidée.
  Dans ce cas, tu dis d'abord : « Parfait, je vous laisse réserver via notre lien. »
  Puis tu envoies le lien seul sur la ligne suivante.
- Tu ne promets jamais de créneau garanti, tu parles de créneau à confirmer par un agent.
- Si la personne parle de Runningman ou Game Zone, tu expliques que c'est géré par Runningman (www.runningmangames.fr, 04 98 09 30 59).
"""
    else:
        brand_rules = """
Spécifique Runningman :
- Tu donnes les informations sur les action games Runningman.
- Si la personne parle de VR, escape game VR, quiz ou salle enfant, tu expliques que ces activités sont gérées par Retroworld dans le même bâtiment (https://www.retroworldfrance.com, 04 94 47 94 64).
- Si la personne parle d'escape game physique ou d'Enigmaniac, tu expliques que c'est un partenaire spécialisé (www.enigmaniac.fr).
"""

    kb_json_text = json.dumps(kb, ensure_ascii=False)

    return (
        kb_prompt
        + "\n\n"
        + header
        + "\n"
        + brand_rules
        + "\nVoici la base de connaissance au format JSON :\n"
        + kb_json_text
    ).strip()


def ask_ai(kb: Dict[str, Any], message: str, brand: str) -> str:
    system_prompt = build_system_prompt(kb, brand)
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    return call_openai(msgs)


# ----------------------------------------------------
# ROUTES
# ----------------------------------------------------

@app.route("/")
def home():
    return jsonify({
        "ok": True,
        "message": "Retroworld IA online (multi-marque Retroworld / Runningman).",
        "endpoints": ["/health", "/chat/<brand>", "/kb/upsert/<brand>"]
    })


@app.route("/health", methods=["GET"])
def health():
    kb_r = load_kb("retroworld")
    kb_rm = load_kb("runningman")

    return jsonify({
        "ok": True,
        "default_brand": DEFAULT_BRAND,
        "kb_faqs_retroworld": count_faqs(kb_r),
        "kb_faqs_runningman": count_faqs(kb_rm),
        "time": time.time()
    })


@app.route("/chat/<brand>", methods=["POST"])
def chat_brand(brand: str):
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Champ 'message' manquant ou vide."}), 400

    b = safe_brand(brand)
    kb = load_kb(b)
    answer = ask_ai(kb, message, b)

    return jsonify({"brand": b, "answer": answer})


@app.route("/kb/upsert/<brand>", methods=["POST"])
def kb_upsert(brand: str):
    """Ajoute une entrée dans la KB (dans la liste 'faqs')."""
    b = safe_brand(brand)
    body = request.get_json(force=True, silent=True) or {}
    question = (body.get("question") or "").strip()
    answer = (body.get("answer") or "").strip()

    if not question or not answer:
        return jsonify({"error": "Champs 'question' et 'answer' requis."}), 400

    kb = load_kb(b)
    if "faqs" not in kb or not isinstance(kb["faqs"], list):
        kb["faqs"] = []

    kb["faqs"].append({"question": question, "answer": answer})
    save_kb(b, kb)

    return jsonify({
        "ok": True,
        "brand": b,
        "faqs_count": len(kb["faqs"])
    })


# ----------------------------------------------------
# MAIN (local)
# ----------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
