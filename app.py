import os
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List

from flask import Flask, request, jsonify
from flask_cors import CORS

# -----------------------
# Configuration de base
# -----------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
ADMIN_API_TOKEN = os.environ.get("ADMIN_API_TOKEN", "").strip()
DEFAULT_BRAND = "retroworld"

if not OPENAI_API_KEY:
    print("⚠️  ATTENTION: OPENAI_API_KEY n'est pas défini dans les variables d'environnement.")

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = Path("/mnt/data")  # stockage persistant Render

# Flask
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)


# -----------------------
# Helpers KB (bases de connaissance)
# -----------------------

def safe_brand(b: str) -> str:
    b = (b or "").strip().lower()
    if b in ("retroworld", "runningman"):
        return b
    return DEFAULT_BRAND


def kb_candidates(brand: str) -> List[Path]:
    """
    Chemins possibles pour la KB :
    1) /mnt/data/kb_<brand>.json  (écrasé par API / shell)
    2) fichier dans le repo : kb_<brand>.json
    """
    b = safe_brand(brand)
    return [
        PERSIST_DIR / f"kb_{b}.json",
        BASE_DIR / f"kb_{b}.json",
    ]


DEFAULT_KB: Dict[str, Any] = {
    "identite": {
        "nom": "Retroworld France",
        "adresse": "815 avenue Pierre Brossolette, 83300 Draguignan",
        "telephone": "04 94 47 94 64",
        "site": "https://www.retroworldfrance.com",
        "horaires": "Du mardi au dimanche, de 11h à 22h",
    },
    "prompt": "Tu es l'assistant de Retroworld France. Si une information manque, dis-le.",
    "faqs": [],
}


def ensure_data_dir() -> None:
    try:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Pas grave, on tombera sur les fichiers du repo
        pass


def load_kb(brand: str) -> Dict[str, Any]:
    """Charge la KB de la marque. Essaie /mnt/data puis les fichiers du repo."""
    ensure_data_dir()
    for path in kb_candidates(brand):
        try:
            if path.is_file():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Erreur lecture KB {path}: {e}")
            continue
    return DEFAULT_KB.copy()


def save_kb(brand: str, data: Dict[str, Any]) -> None:
    """Enregistre la KB uniquement dans /mnt/data (pour les mises à jour dynamiques)."""
    ensure_data_dir()
    path = PERSIST_DIR / f"kb_{safe_brand(brand)}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def kb_faq_count(brand: str) -> int:
    kb = load_kb(brand)
    if "faqs" in kb and isinstance(kb["faqs"], list):
        return len(kb["faqs"])
    if "faq" in kb and isinstance(kb["faq"], list):
        return len(kb["faq"])
    return 0


# -----------------------
# Appel à l'API OpenAI
# -----------------------

def call_openai(messages: List[Dict[str, str]]) -> str:
    """
    Appelle l'API OpenAI Chat Completions via HTTP brut (urllib)
    pour éviter d'avoir à installer d'autres libs.
    """
    if not OPENAI_API_KEY:
        return "Erreur interne : la clé OPENAI_API_KEY n'est pas configurée sur le serveur."

    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.3,
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
        print("HTTPError OpenAI:", e, err_body)
        return "Désolé, une erreur technique est survenue avec le service d'IA."
    except Exception as e:
        print("Erreur OpenAI:", e)
        return "Désolé, une erreur technique est survenue avec le service d'IA."

    try:
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Erreur parsing réponse OpenAI:", e, result)
        return "Désolé, je n'ai pas pu générer de réponse pour le moment."


# -----------------------
# Construction du prompt
# -----------------------

def build_system_prompt(brand: str, kb: Dict[str, Any]) -> str:
    identite = kb.get("identite", {})
    nom = identite.get("nom", "l'établissement")
    site = identite.get("site", "")
    telephone = identite.get("telephone", "")
    horaires = identite.get("horaires", "")

    base_prompt = kb.get("prompt", "")

    # Règles communes Retroworld / Runningman
    rules_common = f"""
Tu es l'assistant officiel de {nom}.
Tu réponds en français, avec un ton professionnel, chaleureux et clair.

Coordonnées connues :
- Téléphone : {telephone}
- Site web : {site}
- Horaires : {horaires}

Tu t'appuies UNIQUEMENT sur les informations fournies dans la base de connaissance JSON.
Si une information n’est pas présente ou pas claire, tu le dis clairement
et tu proposes de mettre le client en relation avec un agent humain.

Ne donne jamais de faux liens, ni de prix que tu n'as pas dans la base.
N'invente pas de créneaux disponibles : parle toujours de créneau "à confirmer par un agent".
"""

    if safe_brand(brand) == "retroworld":
        rules_brand = """
Spécifique Retroworld :
- Dès qu'il est question de réservation, devis ou anniversaire :
  tu demandes toujours : activité + date + heure précise (ou fourchette) + nombre de joueurs
  + coordonnées (nom, e-mail, téléphone).
- Tu ne donnes les liens de réservation QUE si le client est clairement décidé.
  Dans ce cas, tu dis d'abord : « Parfait, je vous laisse réserver via notre lien. »
  puis tu fournis le lien seul sur la ligne suivante.
- Si la demande concerne Runningman ou "Game Zone", tu rediriges vers Runningman (www.runningmangames.fr, 04 98 09 30 59).
"""
    else:
        rules_brand = """
Spécifique Runningman :
- Tu donnes les infos sur les action games et minijeux de Runningman.
- Si la personne parle de réalité virtuelle, jeux VR, escape game VR, quiz ou salle enfant,
  tu expliques que ces activités sont gérées par Retroworld (même bâtiment) et tu rediriges vers
  Retroworld France (https://www.retroworldfrance.com, 04 94 47 94 64).
- Si la personne parle d'escape game physique ou d'Enigmaniac, tu expliques que c'est un partenaire
  spécialisé dans les escape games et tu donnes le site www.enigmaniac.fr.
"""

    # On injecte le JSON complet de KB comme "connaissance brute" pour l'IA
    kb_json_text = json.dumps(kb, ensure_ascii=False)

    final_prompt = f"""{base_prompt}

{rules_common}

{rules_brand}

Voici la base de connaissance au format JSON, que tu peux utiliser comme source de vérité :

{kb_json_text}

Quand tu réponds :
- sois précis, mais pas trop long,
- reformule de manière naturelle (ne recopie pas le JSON brut),
- propose d'organiser une réservation uniquement si la personne semble intéressée.
"""

    return final_prompt.strip()


# -----------------------
# Routes API
# -----------------------

@app.route("/health", methods=["GET"])
def health() -> Any:
    """Petit endpoint de statut pour Render & debug."""
    return jsonify({
        "ok": True,
        "time": time.time(),
        "default_brand": DEFAULT_BRAND,
        "kb_faqs_retroworld": kb_faq_count("retroworld"),
        "kb_faqs_runningman": kb_faq_count("runningman"),
    })


@app.route("/chat/<brand>", methods=["POST"])
def chat_brand(brand: str) -> Any:
    """Endpoint principal de chat: /chat/retroworld ou /chat/runningman."""
    brand = safe_brand(brand)
    body = request.get_json(force=True, silent=True) or {}
    user_message = (body.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Champ 'message' manquant."}), 400

    kb = load_kb(brand)
    system_prompt = build_system_prompt(brand, kb)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    answer = call_openai(messages)
    return jsonify({"answer": answer})


@app.route("/kb/upsert/<brand>", methods=["POST"])
def kb_upsert(brand: str) -> Any:
    """
    Ajout / mise à jour d'une FAQ dans la KB.
    Protégé par ADMIN_API_TOKEN (header X-Admin-Token).
    """
    brand = safe_brand(brand)

    if ADMIN_API_TOKEN:
        token = request.headers.get("X-Admin-Token", "")
        if token.strip() != ADMIN_API_TOKEN:
            return jsonify({"error": "Non autorisé"}), 403

    body = request.get_json(force=True, silent=True) or {}
    q = (body.get("question") or "").strip()
    a = (body.get("answer") or "").strip()

    if not q or not a:
        return jsonify({"error": "Champs 'question' et 'answer' requis."}), 400

    kb = load_kb(brand)
    faqs = kb.get("faqs")
    if not isinstance(faqs, list):
        faqs = []
    # On ajoute en fin
    faqs.append({"question": q, "answer": a})
    kb["faqs"] = faqs

    save_kb(brand, kb)

    return jsonify({
        "ok": True,
        "brand": brand,
        "faqs_count": len(faqs),
    })


@app.route("/", methods=["GET"])
def root() -> Any:
    """Petite page info."""
    return jsonify({
        "message": "Retroworld IA – endpoints disponibles : /chat/retroworld, /chat/runningman, /health",
        "static_chat_widget": "/static/chat-widget.html",
    })


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
