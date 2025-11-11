# app.py — Backend Retroworld IA (unique & sécurisé)
import os, json, urllib.request, urllib.error
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# --- App & static ---
app = Flask(__name__, static_url_path="/static", static_folder="static")

# --- ENV / Sécurité / Marque ---
BRAND_ID = os.environ.get("BRAND_ID", "retroworld")  # tag de marque (nom du fichier KB)
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS","").split(",") if o.strip()]
ADMIN_API_TOKEN = os.environ.get("ADMIN_API_TOKEN","")  # jeton pour /kb/upsert
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY","")
TZ = os.environ.get("TZ", "Europe/Paris")

# CORS: seul ton site peut appeler /chat (sinon * en dev)
if ALLOWED_ORIGINS:
    CORS(app, resources={r"/chat": {"origins": ALLOWED_ORIGINS}})
else:
    CORS(app, resources={r"/chat": {"origins": "*"}})  # dev: à restreindre en prod

# --- Base de connaissances privée (persistée dans le conteneur) ---
KB_FILE = f"/mnt/data/kb_{BRAND_ID}.json"

DEFAULT_KB = {
  "identite": {
    "nom": "Retroworld France",
    "adresse": "815 avenue Pierre Brossolette, 83300 Draguignan",
    "telephone": "04 94 47 94 64",
    "site": "https://www.retroworldfrance.com",
    "horaires": "Du mardi au dimanche, de 11h à 22h"
  },
  "activites": [
    {"nom":"Jeux VR", "prix":"15 €/joueur", "joueurs":"jusqu’à 5"},
    {"nom":"Escape Game VR", "prix":"30 €/joueur", "joueurs":"jusqu’à 5"},
    {"nom":"Quiz interactifs", "prix":"8€ (30m) / 15€ (60m) / 20€ (90m)", "joueurs":"jusqu’à 12"},
    {"nom":"Salle enfant", "prix":"50 €/h", "extra":"20 €/demi-heure sup."}
  ],
  "faqs": []
}

def _ensure_data_dir():
    try:
        os.makedirs("/mnt/data", exist_ok=True)
    except Exception:
        pass

def load_kb():
    try:
        with open(KB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_KB

def save_kb(data):
    _ensure_data_dir()
    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

KB = load_kb()

# --- Prompt système (Retroworld only) ---
SYSTEM_PROMPT = f"""
Tu es l'assistant officiel **Retroworld France** (Draguignan). Parle français, ton pro, clair et concis (1–3 phrases).
Si réservation: propose un lien/numéro ou un rappel. Ne promets rien d'incertain. Si hors périmètre, dis-le poliment.
Infos fixes:
- Nom: {KB['identite']['nom']}
- Adresse: {KB['identite']['adresse']}
- Téléphone: {KB['identite']['telephone']}
- Site: {KB['identite']['site']}
- Horaires: {KB['identite']['horaires']}
- Activités: {", ".join(a["nom"] for a in KB["activites"])}
- Prix: Jeux VR 15€/joueur; Escape VR 30€/joueur; Quiz 8/15/20€; Salle enfant 50€/h.
Tu ne réponds que pour Retroworld France. Si question hors Retroworld, réponds: "Je peux répondre uniquement pour Retroworld France."
"""

def call_openai_chat(user_text, history=None):
    """Appelle OpenAI pour répondre (mêmes connaissances côté voix & chat)."""
    if not OPENAI_API_KEY:
        return "Le service démarre, merci de réessayer dans quelques instants."
    msgs = [{"role":"system","content": SYSTEM_PROMPT}]
    # Injecte nos FAQ privées
    try:
        kb_short = json.dumps({"faqs": KB.get("faqs", [])}, ensure_ascii=False)[:3000]
        msgs.append({"role":"system","content": f"Base de connaissances Retroworld (FAQ): {kb_short}"})
    except Exception:
        pass
    if history:
        for u, a in history[-6:]:
            msgs.append({"role":"user","content": u})
            msgs.append({"role":"assistant","content": a})
    msgs.append({"role":"user","content": user_text})

    payload = {"model": "gpt-4o-mini", "temperature": 0.3, "messages": msgs}
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"},
        data=json.dumps(payload).encode("utf-8")
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            out = json.loads(r.read().decode("utf-8"))
            return out["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        return f"Désolé, une erreur est survenue ({e.code})."
    except Exception:
        return "Désolé, une erreur technique est survenue."

# --- Healthcheck ---
@app.get("/health")
def health():
    return jsonify(ok=True, brand=BRAND_ID, kb_faqs=len(KB.get("faqs", [])))

# --- Chat Web (site Retroworld) ---
@app.post("/chat")
def chat():
    data = request.get_json(force=True, silent=True) or {}
    user_text = (data.get("message") or "").strip()
    history = data.get("history") or []
    if not user_text:
        return jsonify(error="message manquant"), 400
    answer = call_openai_chat(user_text, history=history)
    return jsonify(answer=answer)

# --- Ajout/MàJ de FAQ privées (protégé par jeton) ---
@app.post("/kb/upsert")
def kb_upsert():
    if ADMIN_API_TOKEN and request.headers.get("X-Admin-Token") != ADMIN_API_TOKEN:
        return jsonify(error="unauthorized"), 401
    payload = request.get_json(force=True)
    q = (payload.get("question") or "").strip()
    a = (payload.get("answer") or "").strip()
    if not q or not a:
        return jsonify(error="question/answer requis"), 400

    data = load_kb()
    faqs = data.setdefault("faqs", [])
    lower_q = q.lower()
    for f in faqs:
        if f.get("question","").strip().lower() == lower_q:
            f["answer"] = a
            save_kb(data)
            global KB; KB = data
            return jsonify(status="updated")
    faqs.append({"question": q, "answer": a})
    save_kb(data)
    KB = data
    return jsonify(status="created")

# --- Voix (hook dédié Retroworld) : Quicktalk/Ringover pointeront ici ---
@app.post("/voice/retroworld")
def voice_retroworld():
    # Placeholder de vérification (OK Render <-> Quicktalk).
    # Étape suivante : brancher STT/TTS + logique de transfert.
    return Response("Webhook vocal Retroworld actif.", mimetype="text/plain")

@app.post("/status/retroworld")
def status_retroworld():
    return ("", 204)

if __name__ == "__main__":
    # Render écoute sur $PORT (exposé 8080)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
