import os
import json
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import time

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
app = Flask(__name__)
CORS(app)

DEBUG_LOGS = True

DEFAULT_BRAND = os.environ.get("DEFAULT_BRAND", "retroworld").lower()

DATA_DIR = Path("/mnt/data")  # persistant (Render)
APP_DIR = Path(__file__).resolve().parent  # Docker image


# ----------------------------------------------------
# UTIL
# ----------------------------------------------------
def safe_brand(b: str):
    if not b:
        return DEFAULT_BRAND
    return b.lower().strip()


def kb_path(brand: str) -> Path:
    return DATA_DIR / f"kb_{brand}.json"


def ensure_data_dir():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


# ----------------------------------------------------
# KB LOADING (nouveau : fallback vers /app/)
# ----------------------------------------------------
DEFAULT_KB = {
    "identite": {"nom": "Entreprise"},
    "prompt": "Base de connaissance vide.",
    "faqs": []
}


def load_kb(brand: str):
    """Cherche la KB :
    1) /mnt/data/kb_brand.json (priorité)
    2) /app/kb_brand.json (fallback)
    """
    ensure_data_dir()
    b = safe_brand(brand)

    candidates = [
        DATA_DIR / f"kb_{b}.json",
        APP_DIR / f"kb_{b}.json"
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
                print(f"[KB] Erreur lecture {p} :", e)

    return DEFAULT_KB.copy()


def count_faqs(kb: dict):
    if "faqs" in kb and isinstance(kb["faqs"], list):
        return len(kb["faqs"])
    if "faq" in kb and isinstance(kb["faq"], list):
        return len(kb["faq"])
    return 0


# ----------------------------------------------------
# IA (OpenAI)
# ----------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)


def ask_ai(kb: dict, message: str, brand: str):
    """Envoie la question à OpenAI avec tout le contexte KB"""
    system_prompt = (
        kb.get("prompt", "")
        + "\n\n"
        + "Voici la base de connaissance JSON à utiliser STRICTEMENT :\n"
        + json.dumps(kb, ensure_ascii=False)
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print("[AI ERROR]", e)
        return "Désolé, une erreur interne est survenue."


# ----------------------------------------------------
# ROUTES
# ----------------------------------------------------

@app.route("/")
def home():
    return jsonify({"ok": True, "message": "Retroworld IA online"})


@app.route("/health")
def health():
    kb_r = load_kb("retroworld")
    kb_run = load_kb("runningman")

    return jsonify({
        "ok": True,
        "default_brand": DEFAULT_BRAND,
        "kb_faqs_retroworld": count_faqs(kb_r),
        "kb_faqs_runningman": count_faqs(kb_run),
        "time": time.time()
    })


# ---------------------------
# CHAT : /chat/<brand>
# ---------------------------
@app.route("/chat/<brand>", methods=["POST"])
def chat_brand(brand):
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "")
    b = safe_brand(brand)

    kb = load_kb(b)
    answer = ask_ai(kb, message, b)

    return jsonify({"brand": b, "answer": answer})


# ---------------------------
# KB UPSERT : /kb/upsert/<brand>
# Permet d'ajouter / modifier une FAQ dans /mnt/data
# ---------------------------
@app.route("/kb/upsert/<brand>", methods=["POST"])
def kb_upsert(brand):
    b = safe_brand(brand)

    kb = load_kb(b)
    new_item = request.get_json(force=True, silent=True) or {}

    if "faqs" not in kb or not isinstance(kb["faqs"], list):
        kb["faqs"] = []

    kb["faqs"].append(new_item)

    ensure_data_dir()
    dest = kb_path(b)

    try:
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(kb, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": True, "brand": b, "faqs_count": len(kb["faqs"])})


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
