# app.py — Backend IA multimarques (Retroworld + Runningman)
import os, json, urllib.request, urllib.error
from typing import Dict, Any, List
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# --- App & static ---
app = Flask(__name__, static_url_path="/static", static_folder="static")

# --- ENV ---
DEFAULT_BRAND = os.environ.get("BRAND_ID", "retroworld").strip().lower()
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS","").split(",") if o.strip()]
ADMIN_API_TOKEN = os.environ.get("ADMIN_API_TOKEN","")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY","")
DEBUG_LOGS = os.environ.get("DEBUG_LOGS","").lower() in ("1","true","yes")
TZ = os.environ.get("TZ", "Europe/Paris")

# Transferts / SMS (placeholders pour logique voix)
def _env_list(name: str) -> List[str]:
    raw = os.environ.get(name, "") or ""
    return [x.strip() for x in raw.split(",") if x.strip()]

FORWARD_NUMBERS = {
    "retroworld": _env_list("FORWARD_NUMBERS_RETROWORLD"),
    "runningman": _env_list("FORWARD_NUMBERS_RUNNINGMAN"),
}
ADMIN_SMS = {
    "retroworld": _env_list("ADMIN_SMS_RETROWORLD"),
    "runningman": _env_list("ADMIN_SMS_RUNNINGMAN"),
}

# --- CORS : restreindre /chat (les autres routes peuvent rester ouvertes)
if ALLOWED_ORIGINS:
    CORS(app, resources={r"/chat/*": {"origins": ALLOWED_ORIGINS}})
else:
    CORS(app, resources={r"/chat/*": {"origins": "*"}})  # dev only

# --- Helpers marque & fichiers ---
def safe_brand(b: str) -> str:
    b = (b or "").strip().lower()
    return b if b in ("retroworld","runningman") else DEFAULT_BRAND

def kb_path(brand: str) -> str:
    return f"/mnt/data/kb_{safe_brand(brand)}.json"

DEFAULT_KB: Dict[str, Any] = {
    "identite": {
        "nom": "Retroworld France",
        "adresse": "815 avenue Pierre Brossolette, 83300 Draguignan",
        "telephone": "04 94 47 94 64",
        "site": "https://www.retroworldfrance.com",
        "horaires": "Du mardi au dimanche, de 11h à 22h"
    },
    "faqs": []
}

def ensure_data_dir():
    try:
        os.makedirs("/mnt/data", exist_ok=True)
    except Exception:
        pass

def load_kb(brand: str) -> Dict[str, Any]:
    """Charge la KB de la marque; retourne DEFAULT_KB si absent."""
    ensure_data_dir()
    p = kb_path(brand)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_KB.copy()

def save_kb(brand: str, data: Dict[str, Any]) -> None:
    ensure_data_dir()
    p = kb_path(brand)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def kb_faq_count(brand: str) -> int:
    kb = load_kb(brand)
    # compat : certaines KB utilisent "faq" ou "faqs"
    if "faqs" in kb and isinstance(kb["faqs"], list):
        return len(kb["faqs"])
    if "faq" in kb and isinstance(kb["faq"], list):
        return len(kb["faq"])
    return 0

# --- Prompt système dynamique par marque ---
def build_system_prompt(brand: str, kb: Dict[str, Any]) -> str:
    ident = kb.get("identite", {})
    nom = ident.get("nom", "Retroworld France")
    adresse = ident.get("adresse", "")
    telephone = ident.get("telephone", "")
    site = ident.get("site", "")
    horaires = ident.get("horaires", "Du mardi au dimanche, de 11h à 22h")

    # Prompt spécifique marque si présent
    prompt_extra = kb.get("prompt", "")
    # Règles générales Retroworld (sécurité ton/infos)
    base_rules = (
        "Parle français, ton professionnel, chaleureux et clair, en 1–3 phrases.\n"
        "Réponds uniquement avec les informations confirmées par la marque.\n"
        "Si tu ne sais pas, dis-le et propose un rappel.\n"
    )
    # Note multi-marques
    if brand == "retroworld":
        multi = ("Tu réponds uniquement pour Retroworld (VR, escape VR, quiz, salle enfant). "
                 "Si question Runningman/Game Zone → renvoie poliment vers Runningman : "
                 "https://www.runningmangames.fr ou 04 98 09 30 59.")
    else:
        multi = ("Tu réponds uniquement pour Runningman (action game). "
                 "Si question VR/escape VR/quiz/salle enfant → renvoie vers Retroworld : "
                 "https://www.retroworldfrance.com ou 04 94 47 94 64.")

    return (
        f"Tu es l'assistant officiel **{nom}**.\n"
        f"Adresse: {adresse}\n"
        f"Téléphone: {telephone}\n"
        f"Site: {site}\n"
        f"Horaires: {horaires}\n\n"
        f"{base_rules}\n{multi}\n\n"
        f"{prompt_extra}"
    ).strip()

# --- Appel OpenAI ---
def call_openai_chat(user_text: str, brand: str, history=None) -> str:
    if not OPENAI_API_KEY:
        return "Service en initialisation. Merci de réessayer dans quelques instants."

    kb = load_kb(brand)
    system_prompt = build_system_prompt(brand, kb)

    msgs = [{"role": "system", "content": system_prompt}]
    # Injecte la KB (limitation de taille simple)
    # On supporte "faqs" ou "faq"
    faqs = kb.get("faqs", kb.get("faq", []))
    try:
        kb_short = json.dumps({"faqs": faqs}, ensure_ascii=False)[:6000]
        msgs.append({"role": "system", "content": f"Base de connaissances: {kb_short}"})
    except Exception:
        pass

    if history:
        for turn in history[-6:]:
            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                msgs.append({"role": "user", "content": turn[0]})
                msgs.append({"role": "assistant", "content": turn[1]})

    msgs.append({"role": "user", "content": user_text})

    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "messages": msgs
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8")
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            out = json.loads(r.read().decode("utf-8"))
            return out["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        if DEBUG_LOGS: print("OpenAI HTTPError:", e)
        return f"Désolé, une erreur est survenue ({e.code})."
    except Exception as e:
        if DEBUG_LOGS: print("OpenAI Error:", e)
        return "Désolé, une erreur technique est survenue."

# --- Health ---
@app.get("/health")
def health():
    # Retourne un état global et le nb de FAQ par marque
    data = {
        "ok": True,
        "brand": DEFAULT_BRAND,
        "kb_faqs_retroworld": kb_faq_count("retroworld"),
        "kb_faqs_runningman": kb_faq_count("runningman")
    }
    return jsonify(data)

# --- Chat : /chat/<brand> et /chat (brand par défaut) ---
@app.post("/chat/<brand>")
def chat_brand(brand):
    brand = safe_brand(brand)
    data = request.get_json(force=True, silent=True) or {}
    user_text = (data.get("message") or "").strip()
    history = data.get("history") or []
    if not user_text:
        return jsonify(error="message manquant"), 400
    answer = call_openai_chat(user_text, brand=brand, history=history)
    return jsonify(answer=answer)

@app.post("/chat")
def chat_default():
    # fallback brand via body.brand ou DEFAULT_BRAND
    data = request.get_json(force=True, silent=True) or {}
    brand = safe_brand(data.get("brand") or DEFAULT_BRAND)
    user_text = (data.get("message") or "").strip()
    history = data.get("history") or []
    if not user_text:
        return jsonify(error="message manquant"), 400
    answer = call_openai_chat(user_text, brand=brand, history=history)
    return jsonify(answer=answer)

# --- Admin: upsert FAQ (supporte /kb/upsert et /kb/upsert/<brand>) ---
def do_kb_upsert(brand: str, q: str, a: str):
    brand = safe_brand(brand)
    kb = load_kb(brand)
    # compat : faqs vs faq
    faqs = kb.setdefault("faqs", kb.get("faq", []))
    if "faq" in kb and "faqs" not in kb:
        kb["faqs"] = faqs
        kb.pop("faq", None)

    lower_q = (q or "").strip().lower()
    if not lower_q or not a.strip():
        return {"error": "question/answer requis"}, 400

    for f in faqs:
        if (f.get("question") or f.get("q","")).strip().lower() == lower_q:
            # normaliser la structure
            f["question"] = f.get("question") or f.get("q")
            f.pop("q", None)
            f["answer"] = a
            f.pop("a", None)
            save_kb(brand, kb)
            return {"status": "updated", "brand": brand, "faqs": len(faqs)}, 200

    faqs.append({"question": q, "answer": a})
    save_kb(brand, kb)
    return {"status": "created", "brand": brand, "faqs": len(faqs)}, 200

@app.post("/kb/upsert")
def kb_upsert_default():
    if ADMIN_API_TOKEN and request.headers.get("X-Admin-Token") != ADMIN_API_TOKEN:
        return jsonify(error="unauthorized"), 401
    payload = request.get_json(force=True) or {}
    q = (payload.get("question") or payload.get("q") or "").strip()
    a = (payload.get("answer") or payload.get("a") or "").strip()
    brand = safe_brand(payload.get("brand") or DEFAULT_BRAND)
    resp, code = do_kb_upsert(brand, q, a)
    return jsonify(resp), code

@app.post("/kb/upsert/<brand>")
def kb_upsert_brand(brand):
    if ADMIN_API_TOKEN and request.headers.get("X-Admin-Token") != ADMIN_API_TOKEN:
        return jsonify(error="unauthorized"), 401
    payload = request.get_json(force=True) or {}
    q = (payload.get("question") or payload.get("q") or "").strip()
    a = (payload.get("answer") or payload.get("a") or "").strip()
    resp, code = do_kb_upsert(brand, q, a)
    return jsonify(resp), code

# --- Voix (placeholders : à connecter à Quicktalk/Ringover) ---
def brand_ok(brand: str) -> bool:
    return brand in ("retroworld","runningman")

@app.post("/voice/<brand>")
def voice_brand(brand):
    brand = safe_brand(brand)
    # Ici, on branchera STT/TTS + logique d'accueil + transfert
    return Response(f"Webhook vocal {brand} actif.", mimetype="text/plain")

@app.post("/status/<brand>")
def status_brand(brand):
    # Callback de statut d'appel (si dispo côté opérateur)
    return ("", 204)

# Transfert manuel (utilitaire) — renvoie la cible prévue (pour test)
@app.post("/voice/transfer/<brand>")
def voice_transfer_brand(brand):
    brand = safe_brand(brand)
    numbers = FORWARD_NUMBERS.get(brand, [])
    if not numbers:
        return Response("Aucun numéro de transfert configuré", status=500)
    return Response(f"Transfert {brand} vers: {', '.join(numbers)}", mimetype="text/plain")

# --- Run ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
