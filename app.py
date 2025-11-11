# app.py — Backend Retroworld IA (Flask + OpenAI)
import os, json, urllib.request, urllib.error
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__, static_url_path="/static", static_folder="static")
CORS(app, resources={r"/chat": {"origins": "*"}})  # autorise le widget web à appeler /chat

# --------- Config via ENV ---------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ADMIN_EMAILS = [e.strip() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()]
TZ = os.environ.get("TZ", "Europe/Paris")

# KB (base de connaissances) en mémoire + fichier persistant si présent
KB_FILE = "kb.json"

DEFAULT_KB = {
  "identite": {
    "nom": "Retroworld France",
    "adresse": "815 avenue Pierre Brossolette, 83300 Draguignan",
    "telephone": "04 94 47 94 64",
    "site": "https://www.retroworldfrance.com",
    "horaires": "Du mardi au dimanche, de 11h à 22h"
  },
  "activites": [
    {"nom":"Jeux VR", "prix":"15 €/joueur", "joueurs":"jusqu’à 5", "note":"ne comprend pas l’escape game VR"},
    {"nom":"Escape Game VR", "prix":"30 €/joueur", "joueurs":"jusqu’à 5", "note":"jeu de complétion si fin anticipée"},
    {"nom":"Quiz interactifs", "prix":"8€ (30min) / 15€ (60min) / 20€ (90min)", "joueurs":"jusqu’à 12, dès 10 ans avec accompagnant"},
    {"nom":"Salle enfant", "prix":"50 €/h", "extra":"20 €/demi-heure supp.", "inclut":"jeux en bois, mur interactif, ballayeuse, stockage goûter"},
  ],
  "faqs": []
}

def load_kb():
    try:
        with open(KB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_KB

def save_kb(data):
    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

KB = load_kb()

SYSTEM_PROMPT = f"""
Tu es l'assistant officiel de Retroworld France. Parle français, ton pro, clair et concis.
Objectifs: répondre aux questions, orienter vers réservation, informer tarifs/horaires/activités.
Ne promets rien d'incertain, reste factuel. Si tu ne sais pas, propose qu'on rappelle rapidement.
Infos fixes:
- Nom: {KB['identite']['nom']}
- Adresse: {KB['identite']['adresse']}
- Téléphone: {KB['identite']['telephone']}
- Site: {KB['identite']['site']}
- Horaires: {KB['identite']['horaires']}
- Activités clés: {", ".join(a["nom"] for a in KB["activites"])}
- Prix: Jeux VR 15€/joueur; Escape VR 30€/joueur; Quiz 8/15/20€; Salle enfant 50€/h.

Réponds en 1-3 phrases. Si la personne veut réserver, propose un lien ou une prise de contact par téléphone.
"""

def call_openai_chat(user_text, history=None):
    if not OPENAI_API_KEY:
        return "Le service est en initialisation. Merci de réessayer dans quelques instants."
    msgs = [{"role":"system","content": SYSTEM_PROMPT}]
    # Injecte FAQs de la KB (résumé) pour la pertinence
    try:
        kb_short = json.dumps({"faqs": KB.get("faqs", [])}, ensure_ascii=False)[:3000]
        msgs.append({"role":"system","content": f"Base de connaissances (FAQ): {kb_short}"} )
    except Exception:
        pass
    if history:
        for turn in history[-6:]:
            msgs.append({"role":"user","content": turn[0]})
            msgs.append({"role":"assistant","content": turn[1]})
    msgs.append({"role":"user","content": user_text})

    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "messages": msgs
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
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

@app.get("/health")
def health():
    return jsonify(ok=True, service="retroworld-ia", kb_faqs=len(KB.get("faqs", [])))

# --------- Chat Web (site) ----------
@app.post("/chat")
def chat():
    data = request.get_json(force=True, silent=True) or {}
    user_text = (data.get("message") or "").strip()
    history = data.get("history") or []  # optionnel: liste de [ [user, assistant], ... ]
    if not user_text:
        return jsonify(error="message manquant"), 400
    answer = call_openai_chat(user_text, history=history)
    return jsonify(answer=answer)

# --------- FAQ Upsert ----------
@app.post("/kb/upsert")
def kb_upsert():
    # Option simple: protection par token (ajoute X-Admin-Token dans l'en-tête côté client)
    admin_token = os.environ.get("ADMIN_API_TOKEN", "")
    if admin_token and request.headers.get("X-Admin-Token") != admin_token:
        return jsonify(error="unauthorized"), 401
    payload = request.get_json(force=True)
    q = (payload.get("question") or "").strip()
    a = (payload.get("answer") or "").strip()
    if not q or not a:
        return jsonify(error="question/answer requis"), 400
    data = load_kb()
    faqs = data.setdefault("faqs", [])
    # upsert par question (insensible à la casse)
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

# --------- VOIX (placeholder pour Quicktalk/Ringover) ----------
# Remarque: Quicktalk/Ringover appelle un webhook quand un appel arrive.
# Ici on renvoie un texte "OK" pour vérifier la chaîne Render <-> Quicktalk
# Ensuite, on branchera la vraie logique vocale (STT/TTS) selon l'API fournie.
@app.post("/voice")
def voice():
    return Response("Webhook vocal Retroworld actif.", mimetype="text/plain")

@app.post("/status")
def status():
    return ("", 204)

if __name__ == "__main__":
    # Render écoutera sur le port défini (Docker expose 8080)
    app.run(host="0.0.0.0", port=8080)
