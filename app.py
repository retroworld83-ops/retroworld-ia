"""
Application Flask pour le Pôle Loisirs Draguignan.

Cette application fournit un chatbot multi‑établissements piloté par un prompt unique,
une FAQ publique et une interface d’administration pour consulter les conversations.

Les informations commerciales et les règles à respecter sont définies dans
`src/data/system_data.py` (voir `SYSTEM_PROMPT`).

Le chatbot ne s’appuie sur aucune base de connaissances externe ; il utilise uniquement
le prompt système et les variables d’environnement pour ajuster son comportement.
"""

import csv
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, make_response, redirect, request, send_from_directory

from src.data.system_data import SYSTEM_PROMPT

try:
    import requests  # Utilisé pour appeler l’API OpenAI
except Exception:
    requests = None  # type: ignore


# -----------------------------------------------------------------------------
# Configuration et variables globales
# -----------------------------------------------------------------------------

# Dossiers de travail
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CONV_DIR = DATA_DIR / "conversations"
STATIC_DIR = BASE_DIR / "static"

CONV_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Lecture des variables d’environnement
def _env(key: str, default: str = "") -> str:
    return (os.getenv(key, default) or "").strip()

# OpenAI
OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-5.2")
OPENAI_REASONING_EFFORT = _env("OPENAI_REASONING_EFFORT", "none")
OPENAI_TEMPERATURE = float(_env("OPENAI_TEMPERATURE", "0.3") or 0.0)
OPENAI_MAX_OUTPUT_TOKENS = int(_env("OPENAI_MAX_OUTPUT_TOKENS", "900"))

# Sécurité admin / CORS
ADMIN_API_TOKEN = _env("ADMIN_API_TOKEN")
ADMIN_DASHBOARD_TOKEN = _env("ADMIN_DASHBOARD_TOKEN")
ALLOWED_ORIGINS = [o for o in _env("ALLOWED_ORIGINS").split(",") if o.strip()]

# Marque par défaut
BRAND_ID_DEFAULT = _env("BRAND_ID", "retroworld").lower() or "retroworld"

# FAQ publique : marques activées
FAQ_ENABLED_BRANDS = [b for b in _env("FAQ_ENABLED_BRANDS", "retroworld,runningman,enigmaniac").split(",") if b.strip()]
# Marques proposées dans le widget public
PUBLIC_BRANDS = [b for b in _env("PUBLIC_BRANDS", ",".join(FAQ_ENABLED_BRANDS)).split(",") if b.strip()]

# URL publique (utilisée dans /brands.json)
PUBLIC_BASE_URL = _env("PUBLIC_BASE_URL")

# Logs de debug
DEBUG_LOGS = _env("DEBUG_LOGS").lower() in ("1", "true", "yes", "on")

# Informations par établissement (nom, contacts, domaines…)
DEFAULT_BRANDS: Dict[str, Dict[str, Any]] = {
    "retroworld": {
        "name": "Retroworld",
        "short": "Retroworld",
        "contact_phone": "04 94 47 94 64",
        "contact_email": "contact@retroworldfrance.com",
        "website": "https://www.retroworldfrance.com",
        "domains": ["retroworldfrance.com", "www.retroworldfrance.com"],
    },
    "runningman": {
        "name": "Runningman",
        "short": "Runningman",
        "contact_phone": "04 98 09 30 59",
        "contact_email": "",
        "website": "https://www.runningmangames.fr",
        "domains": ["runningmangames.fr", "www.runningmangames.fr"],
    },
    "enigmaniac": {
        "name": "Enigmaniac",
        "short": "Enigmaniac",
        "contact_phone": "04 94 50 74 63",
        "contact_email": "",
        "website": "https://enigmaniac-escapegame.com",
        "domains": [],
    },
}

# Fusion avec un éventuel fichier YAML de configuration de marques (optionnel)
def _load_brands_from_yaml() -> Dict[str, Dict[str, Any]]:
    cfg_path = BASE_DIR / "config" / "brands.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        raw = yaml.safe_load(cfg_path.read_text("utf-8")) or {}
        brands = raw.get("brands", {})
        out: Dict[str, Dict[str, Any]] = {}
        for bid, cfg in brands.items():
            if not isinstance(cfg, dict):
                continue
            bid2 = str(bid).strip().lower()
            base = DEFAULT_BRANDS.get(bid2, {"name": bid2.title(), "short": bid2.title()})
            base.update(cfg)
            out[bid2] = base
        return out
    except Exception:
        return {}

BRANDS: Dict[str, Dict[str, Any]] = DEFAULT_BRANDS.copy()
BRANDS.update(_load_brands_from_yaml())
for bid in list(BRANDS.keys()):
    BRANDS[bid]["id"] = bid

# -----------------------------------------------------------------------------
# Utilitaires de journalisation
# -----------------------------------------------------------------------------

def log(*args: Any) -> None:
    """Affiche un message si DEBUG_LOGS est activé."""
    if DEBUG_LOGS:
        print("[DBG]", *args, flush=True)


# -----------------------------------------------------------------------------
# Fonctions d’assistance (détection de marque, sécurité, post‑traitement)
# -----------------------------------------------------------------------------

def normalize_brand(b: str) -> str:
    return (b or "").strip().lower()


def detect_brand_from_origin() -> Optional[str]:
    """Détecte une marque à partir de l’en‑tête Origin/Referer/Host."""
    origin = (request.headers.get("Origin") or "").strip().lower()
    referer = (request.headers.get("Referer") or "").strip().lower()
    host = (request.host or "").strip().lower()
    candidates = [origin, referer, host]
    for cand in candidates:
        for bid, cfg in BRANDS.items():
            for d in cfg.get("domains", []) or []:
                if d and d in cand:
                    return bid
    return None


def detect_brand_from_text(text: str) -> Optional[str]:
    """Détecte une marque si elle est mentionnée dans le texte de l’utilisateur."""
    t = (text or "").lower()
    if "runningman" in t:
        return "runningman"
    if "enigmaniac" in t or "enigma" in t:
        return "enigmaniac"
    if "retroworld" in t or "retro world" in t:
        return "retroworld"
    return None


def get_brand_id(payload: Dict[str, Any]) -> str:
    """Détermine la marque à partir du payload, des en‑têtes ou du texte."""
    # ordre de priorité : JSON -> Header -> Query -> Texte -> Origin -> défaut
    b = normalize_brand(payload.get("brand_id") or "")
    if b and b in BRANDS:
        return b
    hb = normalize_brand(request.headers.get("X-Brand-Id") or "")
    if hb and hb in BRANDS:
        return hb
    qb = normalize_brand(request.args.get("brand_id") or request.args.get("brand") or "")
    if qb and qb in BRANDS:
        return qb
    msg = payload.get("message") or ""
    bt = detect_brand_from_text(msg)
    if bt and bt in BRANDS:
        return bt
    bo = detect_brand_from_origin()
    if bo and bo in BRANDS:
        return bo
    return BRAND_ID_DEFAULT if BRAND_ID_DEFAULT in BRANDS else "retroworld"


def format_contact(bid: str) -> str:
    cfg = BRANDS.get(bid, {})
    phone = cfg.get("contact_phone") or ""
    email = cfg.get("contact_email") or ""
    website = cfg.get("website") or ""
    parts = []
    if phone:
        parts.append(f"📞 {phone}")
    if email:
        parts.append(f"📧 {email}")
    if website:
        parts.append(f"🌐 {website}")
    return " | ".join(parts)


# Patterns pour détecter les risques de promesse de réservation
RESERVATION_FORBIDDEN_PATTERNS = [
    r"\b(c['’]?est réservé|réservé|confirmé|confirmée|je vous bloque|on vous bloque|bloqué|bloquée)\b",
]
# Patterns pour détecter l’intention de réservation ou de prix
RESERVATION_INTENT_PATTERNS = [
    r"\b(réserv|reservation|réservation|dispo|disponibilit|créneau|horaire|anniversaire|goûter|acompte)\b",
]


def _booking_intent(text: str) -> bool:
    t = (text or "").lower()
    for pat in RESERVATION_INTENT_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    return False


def enforce_no_reservation_promises(text: str) -> (str, bool):
    """Neutralise les formulations dangereuses de type « c’est réservé »."""
    lowered = (text or "").lower()
    promised = False
    for pat in RESERVATION_FORBIDDEN_PATTERNS:
        if re.search(pat, lowered, flags=re.IGNORECASE):
            promised = True
            # Remplace les phrases risquées par un avertissement générique
            text = re.sub(
                pat,
                "à confirmer par l’équipe (je n’ai pas accès au planning en direct)",
                text,
                flags=re.IGNORECASE,
            )
    return text, promised


def needs_reservation_disclaimer(user_msg: str) -> bool:
    return _booking_intent(user_msg)


def _price_intent(text: str) -> bool:
    t = (text or "").lower()
    return bool(re.search(r"\b(prix|tarif|tarifs|combien|co[uû]t|co[uû]te|€|euro)\b", t, flags=re.I))


def _qweekle_links_for_retroworld(user_text: str) -> List[str]:
    t = (user_text or "").lower()
    links: List[str] = []
    # Choisir les liens en fonction du contenu
    if re.search(r"\b(escape|escape\s*vr|escape\s*game)\b", t, flags=re.I):
        links.append(
            "https://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=escape%20game&lang=fr"
        )
    if re.search(r"\b(quiz|quizz)\b", t, flags=re.I):
        links.append(
            "https://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=quizz&lang=fr"
        )
    # Par défaut, lien vers les jeux VR arcade
    if not links:
        links.append(
            "https://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=Jeu%20%C3%A0%20la%20partie&lang=fr"
        )
    return links


def _append_retroworld_links_if_missing(user_text: str, reply: str) -> str:
    if "qweekle.com" in (reply or "").lower():
        return reply
    if not (_booking_intent(user_text) or _price_intent(user_text)):
        return reply
    links = _qweekle_links_for_retroworld(user_text)
    block = "\n".join([f"🔗 Lien réservation (Retroworld) : {u}" for u in links])
    return (reply or "").rstrip() + "\n\n" + block


def add_disclaimer_if_needed(answer: str, bid: str, user_msg: str) -> str:
    if not needs_reservation_disclaimer(user_msg):
        return answer
    disclaimer = "Je n’ai pas accès au planning en temps réel, c’est à confirmer par l’équipe. "
    contact = format_contact(bid)
    if contact:
        disclaimer += f"Contact : {contact}"
    # Évite d’ajouter le disclaimer en double
    if disclaimer.lower() not in (answer or "").lower():
        return answer.rstrip() + "\n\n" + disclaimer
    return answer


def build_system_prompt(brand_id: str, user_text: str) -> str:
    """Construit le prompt système à envoyer à OpenAI."""
    bid = brand_id if brand_id in BRANDS else BRAND_ID_DEFAULT
    cfg = BRANDS.get(bid, {}) or {}
    who = cfg.get("name", bid)
    contact = format_contact(bid)
    tech_rules = """Règles techniques (très important) :
- Répondez en français, ton professionnel, vouvoiement.
- Vous n'avez PAS accès au planning ni au logiciel de réservation : ne promettez jamais un créneau "bloqué", "confirmé" ou "réservé".
- Si la demande concerne une disponibilité ou une réservation : recueillez les informations (date, heure, nombre de personnes, activité) puis orientez vers le contact officiel.
- Ne pas inventer : si une information n'est pas dans la base fournie, dites‑le clairement et proposez le bon contact.
- En cas de question multi‑établissements (croisement Retroworld / Runningman / Enigmaniac) : répondez sans confusion, en séparant clairement par établissement.
"""
    session_ctx = f"""--- CONTEXTE SESSION ---\nSite / entité courante : {who} ({bid}).\nContact : {contact}\n"""
    base = SYSTEM_PROMPT.strip() or "Vous êtes l'IA d'accueil du Pôle Loisirs Draguignan. Ne pas inventer. Pas d'accès au planning."
    return (tech_rules + "\n\n" + base + "\n\n" + session_ctx).strip()


def openai_ready() -> bool:
    return bool(OPENAI_API_KEY and requests is not None)


def openai_answer(system: str, user: str) -> str:
    """Interroge l’API OpenAI Responses et renvoie le texte de sortie."""
    if not openai_ready():
        return ""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ],
        "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
    }
    # Mode reasoning effort
    if OPENAI_REASONING_EFFORT:
        payload["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}
    if OPENAI_REASONING_EFFORT == "none":
        payload["temperature"] = OPENAI_TEMPERATURE
    # Requête HTTP
    try:
        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        out_texts: List[str] = []
        for item in data.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    out_texts.append(c.get("text", ""))
        return "\n".join(out_texts).strip()
    except Exception as e:
        log("OpenAI error:", e)
        return "Désolé, je rencontre un souci technique. Pouvez‑vous réessayer ou contacter l’équipe ?"


# -----------------------------------------------------------------------------
# Gestion des conversations (stockage JSON)
# -----------------------------------------------------------------------------

def conv_path(conv_id: str) -> Path:
    return CONV_DIR / f"{conv_id}.json"


def new_conv_id(prefix: str = "rw") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def load_conv(conv_id: str) -> Dict[str, Any]:
    p = conv_path(conv_id)
    if not p.exists():
        return {"id": conv_id, "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "messages": [], "meta": {}}
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:
        return {"id": conv_id, "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "messages": [], "meta": {}}


def save_conv(conv: Dict[str, Any]) -> None:
    p = conv_path(conv.get("id", "unknown"))
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(conv, ensure_ascii=False, indent=2), "utf-8")
    except Exception as e:
        log("save_conv error:", e)


def append_message(conv: Dict[str, Any], role: str, content: str, extra: Optional[Dict[str, Any]] = None) -> None:
    conv.setdefault("messages", [])
    conv["messages"].append(
        {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "role": role,
            "content": content,
            "extra": extra or {},
        }
    )


# -----------------------------------------------------------------------------
# Flags pour l’admin (détection heuristique de devis, réclamations, etc.)
# -----------------------------------------------------------------------------

FLAG_PATTERNS = {
    "devis": r"\b(devis|privatis|entreprise|ce\b|comit[ée]\s*d['’]?entreprise|team\s*building|groupe)\b",
    "reservation": r"\b(réserv|réservation|dispo|créneau|anniversaire|goûter|acompte)\b",
    "reclamation": r"\b(rembourse|annul|plainte|probl[eè]me|litige|panne)\b",
    "croise": r"\b(retroworld|runningman|enigmaniac)\b.*\b(retroworld|runningman|enigmaniac)\b",
    "promesse_resa": r"\b(réservé|confirmé|je vous bloque|c['’]?est réservé|bloqué)\b",
    "a_relire": r"\b(peut[- ]?être|probablement|je pense|à priori|il me semble)\b",
}


def compute_flags(conv: Dict[str, Any]) -> List[str]:
    msgs = conv.get("messages") or []
    blob = "\n".join([(m.get("content") or "") for m in msgs]).lower()
    flags: List[str] = []
    for name, pat in FLAG_PATTERNS.items():
        if re.search(pat, blob, flags=re.IGNORECASE | re.DOTALL):
            flags.append(name)
    # a_valider si devis ou reclamation
    if any(f in flags for f in ("devis", "reclamation")):
        flags.append("a_valider")
    return sorted(set(flags))


# -----------------------------------------------------------------------------
# Création de l’application Flask
# -----------------------------------------------------------------------------

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


@app.after_request
def after(resp):
    """Applique la politique CORS en fin de requête."""
    origin = (request.headers.get("Origin") or "").strip().rstrip("/")
    if ALLOWED_ORIGINS:
        if origin in ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
    else:
        # si non configuré, autoriser tout en dev
        if origin:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
        else:
            resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Brand-Id"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


# -----------------------------------------------------------------------------
# Endpoints publics
# -----------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Redirige vers le widget de chat."""
    return redirect("/static/chat-widget.html")


@app.route("/health", methods=["GET"])
def health():
    """Renvoie un état synthétique du service."""
    return jsonify(
        {
            "ok": True,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "openai_configured": openai_ready(),
            "brands": list(BRANDS.keys()),
            "faq_enabled_brands": FAQ_ENABLED_BRANDS,
            "public_brands": PUBLIC_BRANDS,
        }
    )


@app.route("/brands.json", methods=["GET"])
def brands_json():
    """Renvoie la liste des marques publiques, utilisée par le widget."""
    out = []
    for bid in PUBLIC_BRANDS:
        cfg = BRANDS.get(bid, {})
        out.append(
            {
                "id": bid,
                "name": cfg.get("name", bid),
                "short": cfg.get("short", bid),
                "website": cfg.get("website", ""),
                "contact_phone": cfg.get("contact_phone", ""),
                "contact_email": cfg.get("contact_email", ""),
            }
        )
    return jsonify({"items": out, "base_url": PUBLIC_BASE_URL})


# ------------------ FAQ ------------------
@app.route("/faq", methods=["GET"])
def faq_page():
    """Redirige vers l’onglet FAQ du widget pour une marque donnée."""
    brand_id = normalize_brand(request.args.get("brand_id") or request.args.get("brand") or BRAND_ID_DEFAULT)
    if brand_id not in FAQ_ENABLED_BRANDS:
        return make_response("FAQ indisponible pour le moment.", 404)
    return redirect(f"/static/chat-widget.html?tab=faq&brand={brand_id}")


@app.route("/faq.json", methods=["GET"])
def faq_json():
    """Renvoie la FAQ publique au format JSON pour une marque."""
    brand_id = normalize_brand(request.args.get("brand_id") or request.args.get("brand") or BRAND_ID_DEFAULT)
    if brand_id == "all":
        payload = []
        for bid in FAQ_ENABLED_BRANDS:
            p = STATIC_DIR / f"faq_{bid}.json"
            try:
                data = json.loads(p.read_text("utf-8"))
            except Exception:
                data = {"brand": bid, "items": []}
            payload.append({"brand": bid, "items": data.get("items", [])})
        return jsonify({"items": payload, "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    if brand_id not in FAQ_ENABLED_BRANDS:
        return jsonify({"brand": brand_id, "items": [], "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}), 404
    p = STATIC_DIR / f"faq_{brand_id}.json"
    try:
        data = json.loads(p.read_text("utf-8"))
    except Exception:
        data = {"brand": brand_id, "items": []}
    return jsonify({"brand": brand_id, "updated": data.get("updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "items": data.get("items", [])})


# Alias legacy pour certains widgets
@app.route("/faq_retroworld.json", methods=["GET"])
def faq_retroworld_alias():
    return send_from_directory(str(STATIC_DIR), "faq_retroworld.json")


@app.route("/faq_runningman.json", methods=["GET"])
def faq_runningman_alias():
    return send_from_directory(str(STATIC_DIR), "faq_runningman.json")


# ------------------ CHAT ------------------
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    """Endpoint principal pour le chatbot."""
    if request.method == "OPTIONS":
        return ("", 204)
    payload = request.get_json(silent=True) or {}
    msg = (payload.get("message") or "").strip()
    if not msg:
        return jsonify({"ok": False, "error": "message manquant"}), 400
    brand_id = get_brand_id(payload)
    conv_id = (payload.get("conversation_id") or "").strip()
    if not conv_id:
        conv_id = new_conv_id(prefix=brand_id[:2] if brand_id else "rw")
    conv = load_conv(conv_id)
    conv.setdefault("meta", {})
    conv["meta"]["brand_id"] = brand_id
    conv["meta"]["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_message(conv, "user", msg)
    # Détection d’une marque supplémentaire mentionnée dans la question
    cross = []
    mentioned = detect_brand_from_text(msg)
    if mentioned and mentioned != brand_id and mentioned in BRANDS:
        cross.append(mentioned)
    # Construction du prompt
    sys_prompt = build_system_prompt(brand_id, msg)
    user_prompt = msg
    # Appel OpenAI
    if not openai_ready():
        answer = "Le service IA n'est pas configuré (OPENAI_API_KEY manquante)."
        append_message(conv, "assistant", answer, extra={"brand_id": brand_id, "flags": ["openai_missing"]})
        save_conv(conv)
        return jsonify({"ok": True, "conversation_id": conv_id, "brand_id": brand_id, "answer": answer})
    raw_answer = openai_answer(sys_prompt, user_prompt)
    safe_answer, promised = enforce_no_reservation_promises(raw_answer)
    safe_answer = add_disclaimer_if_needed(safe_answer, brand_id, msg)
    # Ajout des liens Qweekle pour Retroworld si pertinent
    if brand_id == "retroworld":
        safe_answer = _append_retroworld_links_if_missing(msg, safe_answer)
    flags: List[str] = []
    if promised:
        flags.append("promesse_resa")
    append_message(conv, "assistant", safe_answer, extra={"brand_id": brand_id, "flags": flags})
    save_conv(conv)
    return jsonify({"ok": True, "conversation_id": conv_id, "brand_id": brand_id, "answer": safe_answer})


# ------------------ ROUTES ADMIN UI ------------------
@app.route("/admin", methods=["GET"])
def admin_page():
    return send_from_directory(str(STATIC_DIR), "admin.html")


@app.route("/admin/faq", methods=["GET"])
def admin_faq_page():
    return send_from_directory(str(STATIC_DIR), "admin-faq.html")


# ------------------ ROUTES ADMIN API ------------------

def require_admin_token() -> bool:
    token = ""
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    else:
        token = (request.args.get("token") or request.headers.get("X-Admin-Token") or "").strip()
    valid = {t for t in [ADMIN_API_TOKEN, ADMIN_DASHBOARD_TOKEN] if t}
    if not valid:
        # Pas de token configuré -> interdire
        return False
    return token in valid


@app.route("/admin/api/diag", methods=["GET"])
def admin_diag():
    if not require_admin_token():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    return jsonify(
        {
            "ok": True,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "openai_configured": openai_ready(),
            "brands": list(BRANDS.keys()),
            "faq_enabled_brands": FAQ_ENABLED_BRANDS,
            "public_brands": PUBLIC_BRANDS,
            "allowed_origins": ALLOWED_ORIGINS,
        }
    )


@app.route("/admin/api/conversations", methods=["GET"])
def admin_list_conversations():
    if not require_admin_token():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    items: List[Dict[str, Any]] = []
    for p in sorted(CONV_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        conv = json.loads(p.read_text("utf-8")) if p.exists() else {}
        cid = conv.get("id", p.stem)
        meta = conv.get("meta") or {}
        brand_id = meta.get("brand_id") or ""
        msgs = conv.get("messages") or []
        last = msgs[-1]["ts"] if msgs else conv.get("created", "")
        flags = compute_flags(conv)
        items.append(
            {
                "id": cid,
                "brand_id": brand_id,
                "created": conv.get("created", ""),
                "last": last,
                "count": len(msgs),
                "flags": flags,
            }
        )
    return jsonify({"ok": True, "items": items})


@app.route("/admin/api/conversation/<conv_id>", methods=["GET"])
def admin_get_conversation(conv_id: str):
    if not require_admin_token():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    conv = load_conv(conv_id)
    conv["flags"] = compute_flags(conv)
    return jsonify({"ok": True, "conversation": conv})


@app.route("/admin/api/export.csv", methods=["GET"])
def admin_export_csv():
    if not require_admin_token():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    # Génération rapide d’un CSV pour diagnostic
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["conversation_id", "brand_id", "ts", "role", "content", "flags"])
    for p in sorted(CONV_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        conv = json.loads(p.read_text("utf-8")) if p.exists() else {}
        cid = conv.get("id", p.stem)
        brand_id = (conv.get("meta") or {}).get("brand_id", "")
        flags = "|".join(compute_flags(conv))
        for m in conv.get("messages") or []:
            writer.writerow([
                cid,
                brand_id,
                m.get("ts", ""),
                m.get("role", ""),
                (m.get("content", "") or "").replace("\n", " "),
                flags,
            ])
    resp = make_response(output.getvalue())
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = 'attachment; filename="conversations.csv"'
    return resp


@app.route("/admin/api/faq/get", methods=["GET"])
def admin_faq_get():
    if not require_admin_token():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    bid = normalize_brand(request.args.get("brand_id") or BRAND_ID_DEFAULT)
    if bid not in BRANDS:
        return jsonify({"ok": False, "error": "unknown brand"}), 400
    kb_path = STATIC_DIR / f"faq_{bid}.json"
    try:
        kb = json.loads(kb_path.read_text("utf-8"))
    except Exception:
        kb = {"brand": bid, "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "items": []}
    return jsonify({"ok": True, "kb": kb})


@app.route("/admin/api/faq/save", methods=["POST"])
def admin_faq_save():
    if not require_admin_token():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    payload = request.get_json(silent=True) or {}
    bid = normalize_brand(payload.get("brand") or payload.get("brand_id") or BRAND_ID_DEFAULT)
    kb = payload.get("kb") or payload
    if bid not in BRANDS:
        return jsonify({"ok": False, "error": "unknown brand"}), 400
    # vérification du schéma
    items = kb.get("items") if isinstance(kb, dict) else None
    if not isinstance(items, list):
        return jsonify({"ok": False, "error": "items must be a list"}), 400
    cleaned = []
    for it in items:
        if not isinstance(it, dict):
            continue
        q = (it.get("question") or "").strip()
        a = (it.get("answer") or "").strip()
        tags = it.get("tags") or []
        if not q or not a:
            continue
        cleaned.append({"question": q, "answer": a, "tags": tags if isinstance(tags, list) else []})
    kb_out = {"brand": bid, "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "items": cleaned}
    out_path = STATIC_DIR / f"faq_{bid}.json"
    try:
        out_path.write_text(json.dumps(kb_out, ensure_ascii=False, indent=2), "utf-8")
        # mettre à jour le chemin legacy pour runningman
        if bid == "runningman":
            (STATIC_DIR / "static" / "faq_runningman.json").write_text(
                json.dumps(kb_out, ensure_ascii=False, indent=2), "utf-8"
            )
    except Exception as e:
        log("faq_save error:", e)
        return jsonify({"ok": False, "error": "save failed"}), 500
    return jsonify({"ok": True, "saved": True, "updated": kb_out["updated"], "count": len(cleaned)})


# -----------------------------------------------------------------------------
# Fichiers statiques de repli
# -----------------------------------------------------------------------------
@app.route("/static/<path:filename>", methods=["GET"])
def static_files(filename):
    return send_from_directory(str(STATIC_DIR), filename)


if __name__ == "__main__":
    port = int(_env("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=DEBUG_LOGS)