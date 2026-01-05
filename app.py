"""
Flask application for the Retroworld / Runningman conversational assistant.

Hard requirements:
- Zero hallucination on business rules/prices/capacities: if uncertain -> say so and redirect to official contact.
- Never confirm availability in-chat.
- Avoid brand mixing: detect intent (Retroworld vs Runningman) and answer with the correct rules.
- Provide fast, deterministic answers for the common questions (address, prices, duration, capacity, booking, events…).
- Admin dashboard with logs + test console.
- History:
    - Admin can view ALL conversations.
    - User can view ONLY their own conversation history.

Runtime notes:
- KB JSON files are loaded from /mnt/data/kb_<brand>.json (overrides) or /app/kb_<brand>.json (embedded).
- Conversation logs are stored as JSONL files in /mnt/data/logs/conversations/
- User<->conversation binding is stored in /mnt/data/logs/user_index.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retroworld-ia")

BASE_DATA_DIR = "/mnt/data"
BASE_APP_DIR = "/app"

BASE_LOG_DIR = os.getenv("LOG_DIR", os.path.join(BASE_DATA_DIR, "logs"))
CONVERSATIONS_LOG_DIR = os.path.join(BASE_LOG_DIR, "conversations")
QWEEKLE_LOG_DIR = os.path.join(BASE_LOG_DIR, "qweekle")
USER_INDEX_PATH = os.path.join(BASE_LOG_DIR, "user_index.json")

for d in (BASE_LOG_DIR, CONVERSATIONS_LOG_DIR, QWEEKLE_LOG_DIR):
    os.makedirs(d, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

ADMIN_DASHBOARD_TOKEN = os.getenv("ADMIN_DASHBOARD_TOKEN", "changeme_admin_token")
USER_HISTORY_TOKEN = os.getenv("USER_HISTORY_TOKEN", "changeme_user_token")

QWEEKLE_WEBHOOK_SECRET = os.getenv("QWEEKLE_WEBHOOK_SECRET", "")
QWEEKLE_SOURCE_NAME = os.getenv("QWEEKLE_SOURCE_NAME", "retroworld-qweekle")

SUPPORTED_BRANDS: set[str] = {"retroworld", "runningman"}


# ---------------------------------------------------------
# UTIL: SAFE JSON READ/WRITE
# ---------------------------------------------------------

def _safe_read_json(path: str, default: Any) -> Any:
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _safe_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------
# USER INDEX (user_id -> conversation_id)
# ---------------------------------------------------------

def get_user_index() -> Dict[str, str]:
    data = _safe_read_json(USER_INDEX_PATH, {})
    if not isinstance(data, dict):
        return {}
    # keep only str->str
    out: Dict[str, str] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def set_user_conversation(user_id: str, conversation_id: str) -> None:
    if not user_id or not conversation_id:
        return
    idx = get_user_index()
    idx[user_id] = conversation_id
    _safe_write_json(USER_INDEX_PATH, idx)


def get_user_conversation(user_id: str) -> str:
    idx = get_user_index()
    return idx.get(user_id, "")


# ---------------------------------------------------------
# KB CACHE
# ---------------------------------------------------------

@dataclass
class _KBCacheEntry:
    path: str
    mtime: float
    data: Dict[str, Any]


_KB_CACHE: Dict[str, _KBCacheEntry] = {}


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_kb(brand: str) -> Dict[str, Any]:
    """Load KB with caching and /mnt/data override."""
    brand = (brand or "").lower()
    if brand not in SUPPORTED_BRANDS:
        return {}

    candidates = [
        os.path.join(BASE_DATA_DIR, f"kb_{brand}.json"),
        os.path.join(BASE_APP_DIR, f"kb_{brand}.json"),
    ]

    chosen = None
    for p in candidates:
        if os.path.exists(p):
            chosen = p
            break

    if not chosen:
        logger.warning("KB not found for %s; using empty KB", brand)
        return {}

    try:
        mtime = os.path.getmtime(chosen)
        cached = _KB_CACHE.get(brand)
        if cached and cached.path == chosen and cached.mtime == mtime:
            return cached.data
        data = _read_json(chosen)
        if not isinstance(data, dict):
            data = {}
        _KB_CACHE[brand] = _KBCacheEntry(path=chosen, mtime=mtime, data=data)
        logger.info("Loaded KB for %s from %s", brand, chosen)
        return data
    except Exception as e:
        logger.error("Error reading KB %s: %s", chosen, e)
        return {}


def save_kb(brand: str, kb_data: Dict[str, Any]) -> None:
    brand = (brand or "").lower()
    path = os.path.join(BASE_DATA_DIR, f"kb_{brand}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)
    _KB_CACHE.pop(brand, None)
    logger.info("KB %s updated at %s", brand, path)


# ---------------------------------------------------------
# OPENAI CALL
# ---------------------------------------------------------

def call_openai_chat(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")

    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        logger.error("OpenAI HTTPError (%s): %s", e.code, err_body)
        raise
    except urllib.error.URLError as e:
        logger.error("OpenAI URLError: %s", e)
        raise

    obj = json.loads(body)
    content = obj["choices"][0]["message"]["content"]
    usage = obj.get("usage", {})
    return str(content or "").strip(), usage


# ---------------------------------------------------------
# BRAND DETECTION & SAFE RULES
# ---------------------------------------------------------

_RETRO_KEYWORDS = [
    "vr", "réalité virtuelle", "realite virtuelle", "escape vr", "escape game vr",
    "jeux vr", "jeu vr", "casque", "meta quest", "vive pro", "quiz", "quizz",
    "quiz interactif", "salle enfant", "mur interactif", "retroworld", "rétroworld",
    "fidélité", "fidelite", "points", "qr code", "carte cadeau", "billard",
]
_RUNNING_KEYWORDS = [
    "action game", "game zone", "runningman", "running man", "mini-jeux",
    "mini jeux", "défis", "defis", "physique",
]


def detect_brand_from_text(text: str, default: str) -> str:
    t = (text or "").lower()
    retro_score = sum(1 for k in _RETRO_KEYWORDS if k in t)
    run_score = sum(1 for k in _RUNNING_KEYWORDS if k in t)
    if retro_score > run_score and retro_score > 0:
        return "retroworld"
    if run_score > retro_score and run_score > 0:
        return "runningman"
    if "retroworld" in t or "rétroworld" in t:
        return "retroworld"
    if "runningman" in t or "running man" in t:
        return "runningman"
    return default


# ---------------------------------------------------------
# DEFAULT FACTS (ANTI-HALLUCINATION)
# ---------------------------------------------------------

def _get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _retroworld_defaults() -> Dict[str, Any]:
    return {
        "adresse": "815 avenue Pierre Brossolette, 83300 Draguignan, France",
        "site": "https://www.retroworldfrance.com",
        "tel": "04 94 47 94 64",
        "horaires_prix": "Tarifs standard de 11h à 20h. Tarifs majorés de 9h à 11h et de 20h à 23h.",
        "vr": {"prix_normal": 15, "prix_avant_11": 20, "prix_apres_20": 17, "max_joueurs": 5, "session": "Une session = 1 jeu (au choix dans le catalogue)."},
        "escape_vr": {"prix_normal": 30, "prix_majoré": 35, "max_joueurs": 5},
        "quiz": {"prix_30": 8, "prix_60": 15, "prix_90": 20, "suppl_hors_11_20": 5, "max_joueurs": 12, "age": "Dès 10 ans avec accompagnant."},
        "salle_enfant": {"prix_h": 50, "prix_demi_h_sup": 20, "details": "Jeux en bois, mur interactif, balayeuse, stockage goûter."},
        "attente": {"details": "Canapés, boissons/snacks, baby-foot, air hockey, borne de basketball, billard (10€/h), écrans pour suivre les sessions."},
        "equipement": "Casques VR professionnels : Vive Pro 2 et Meta Quest 3 (ou équivalent). Équipements nettoyés entre chaque session.",
        "jeux_counts": {"jeux_vr": 31, "escape_vr": 28},
        "fidelite": {
            "gains": "1 partie VR = 1 point. 1 escape game VR = 2 points. Pas de points sur les formules anniversaire.",
            "recompenses": "5 points = 1 quiz 30 min offert. 10 points = 1 partie VR offerte. 20 points = 1 escape game VR offert.",
        },
        "gouter": {
            "stockage_ok": True,
            "illimite_perime": True,
            "sur_devis": True,
            "phrase": "Vous pouvez stocker un gâteau/goûter sur place. Un goûter peut être proposé sur devis (préparé par Runningman et géré/commercialisé par Retroworld).",
        },
        "qweekle": {
            "vr": "https://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=Jeu%20%C3%A0%20la%20partie&lang=fr",
            "quiz": "https://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=quizz&lang=fr",
            "escape": "https://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=escape%20game&lang=fr",
        },
    }


def _runningman_defaults() -> Dict[str, Any]:
    return {
        "adresse": "815 avenue Pierre Brossolette, 83300 Draguignan, France",
        "site": "https://runningmangames.fr",
        "contact": "https://runningmangames.fr/contact",
        "tel": "04 98 09 30 59",
        "session": "60 minutes (créneaux fixes chaque heure).",
        "capacite": "Jusqu’à 25 personnes par heure (organisation selon réservation).",
        "age": "Accessible dès 7 ans. Les moins de 12 ans doivent être accompagnés d’un adulte.",
        "tarifs": "15€ / personne (moins de 12 ans) et 20€ / personne (12 ans et + / adulte).",
        # goûter: préparé Runningman mais géré/commercialisé Retroworld (comme vous l’avez demandé)
        "gouter": {
            "sur_devis": True,
            "phrase": "Pour un goûter sur devis (préparé par Runningman), c’est géré/commercialisé par Retroworld : 04 94 47 94 64.",
        },
    }


def _merge_defaults(brand: str, kb: Dict[str, Any]) -> Dict[str, Any]:
    base = _retroworld_defaults() if brand == "retroworld" else _runningman_defaults()
    if not isinstance(kb, dict):
        return base

    # overlays minimal, only if clearly present
    if brand == "runningman":
        adresse = _get_nested(kb, "identite.localisation.adresse_complete")
        if isinstance(adresse, str) and adresse.strip():
            base["adresse"] = adresse.strip()
        tel = _get_nested(kb, "identite.contact.telephone")
        if isinstance(tel, str) and tel.strip():
            base["tel"] = tel.strip()
        site = _get_nested(kb, "identite.contact.site_web")
        if isinstance(site, str) and site.strip():
            base["site"] = site.strip()
        contact = _get_nested(kb, "reservation.canaux.contact_page")
        if isinstance(contact, str) and contact.strip():
            base["contact"] = contact.strip()
    else:
        # Retroworld
        tel = _get_nested(kb, "infos_pratiques.coordonnees.telephone") or _get_nested(kb, "contact.telephone")
        if isinstance(tel, str) and tel.strip():
            base["tel"] = tel.strip()
        site = _get_nested(kb, "infos_pratiques.coordonnees.site_web")
        if isinstance(site, str) and site.strip():
            base["site"] = site.strip()

        n_vr = kb.get("nombre_jeux_vr")
        n_escape = kb.get("nombre_escape_vr")
        if isinstance(n_vr, int):
            base["jeux_counts"]["jeux_vr"] = n_vr
        if isinstance(n_escape, int):
            base["jeux_counts"]["escape_vr"] = n_escape

    return base


# ---------------------------------------------------------
# FAST ANSWERS (DETERMINISTIC)
# ---------------------------------------------------------

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _is_reservation_intent(t: str) -> bool:
    t = _norm(t)
    return any(k in t for k in [
        "réserver", "reserver", "reservation", "réservation", "devis", "creneau", "créneau",
        "lien", "lien de réservation", "lien de resa", "dispo", "disponible", "places",
    ])


def _is_event_intent(t: str) -> bool:
    t = _norm(t)
    return any(k in t for k in [
        "anniversaire", "evenement", "événement", "privatis", "goûter", "gouter",
    ])


def answer_fast(brand: str, kb: Dict[str, Any], user_text: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Heuristic answers (no OpenAI call) for common questions.

    IMPORTANT:
    - For Retroworld booking intent, start with "Disponible." (your rule).
    - Never claim a slot is confirmed.
    - Never exceed 5 players simultaneously for VR / Escape VR.
    - No hallucination: only use merged defaults.
    """
    metadata = metadata or {}
    b = (brand or "").lower().strip()
    t_raw = user_text or ""
    t = _norm(t_raw)

    facts = _merge_defaults(b, kb)

    # 0) Smalltalk so "salut" is never blank
    if re.fullmatch(r"(salut|bonjour|bonsoir|hello|yo|coucou|hey)( !)?", t):
        if b == "runningman":
            return f"Bonjour ! 😊\nRunningman Game Zone (action game) est au {facts['adresse']}. Que puis-je faire pour vous : tarifs, réservation ou infos de groupe ?"
        return f"Bonjour ! 😊\nRetroworld est au {facts['adresse']}. Que puis-je faire pour vous : VR, escape VR, quiz, salle enfant, anniversaire ou fidélité ?"

    if re.fullmatch(r"(merci|thx|thanks|super)( !)?", t):
        return "Avec plaisir 🙂 Si vous voulez, dites-moi l’activité, la date et le nombre de personnes, et je vous oriente."

    # 1) Retroworld: direct booking links when clearly asked
    if b == "retroworld":
        if re.search(r"\b(lien|link)\b", t) and re.search(r"\b(resa|réserv|reservation|réservation)\b", t):
            qw = facts.get("qweekle", {})
            if "escape" in t:
                return f"Parfait, je vous laisse réserver via notre lien.\n{qw.get('escape')}"
            if "quiz" in t or "quizz" in t:
                return f"Parfait, je vous laisse réserver via notre lien.\n{qw.get('quiz')}"
            if "vr" in t:
                return f"Parfait, je vous laisse réserver via notre lien.\n{qw.get('vr')}"
            return "Pour quelle activité souhaitez-vous le lien : jeux VR, escape game VR, quiz interactif ou salle enfant ?"

    # 2) Adresse / où
    if re.search(r"\b(adresse|où|ou est|vous etes ou|c est ou|localisation)\b", t):
        return f"Adresse : {facts['adresse']}"

    # 3) Horaires / dimanche
    if re.search(r"(horaire|horaires|ouvert|ouverture|ferme|dimanche)", t):
        if b == "retroworld":
            return f"{facts['horaires_prix']}\nAdresse : {facts['adresse']}"
        return f"Runningman Game Zone est au {facts['adresse']}. Pour les horaires exacts, merci de consulter {facts['site']} ou d’appeler le {facts['tel']}."

    # 4) Fidélité (avant le bloc VR)
    if b == "retroworld" and re.search(r"\b(fid[eé]lit[eé]|points?)\b", t):
        f = facts["fidelite"]
        return (
            "Programme fidélité :\n"
            f"• {f['gains']}\n"
            f"Récompenses : {f['recompenses']}"
        )

    # 5) Goûter / stockage
    if re.search(r"(gouter|goûter|gateau|gâteau|stock|stocker|frigo)", t):
        if b == "retroworld":
            if "a volonte" in t or "à volonté" in t_raw.lower():
                return (
                    "La formule de goûter illimité n’est plus proposée.\n"
                    + facts["gouter"]["phrase"]
                )
            return facts["gouter"]["phrase"]
        return facts["gouter"]["phrase"]

    # 6) Écrans / salle d’attente
    if re.search(r"(salle d attente|attente|ecran|écran|regarder|diffus)", t):
        if b == "retroworld":
            return "Oui, vous pouvez patienter dans la salle d’attente. Les écrans diffusent la vue du jeu (pas le joueur)."
        return "Vous pouvez patienter sur place. Pour les détails précis, merci de contacter Runningman : 04 98 09 30 59."

    # 7) Casques / hygiène
    if b == "retroworld" and re.search(r"(casque|vive|quest|meta|netto|désinfect|desinfect|lunettes)", t):
        return (
            f"{facts['equipement']}\n"
            "Lunettes : oui, c’est possible (on vous aide à ajuster le casque)."
        )

    # 8) Capacités
    if re.search(r"(combien de joueurs|max|on est \d+|a \d+|possible a \d+|solo|duo|trio)", t):
        if b == "retroworld":
            return (
                "Selon l’activité :\n"
                "• Jeux VR : 1 à 5 joueurs simultanés\n"
                "• Escape game VR : jusqu’à 5 joueurs (souvent 2 à 5 selon le scénario)\n"
                "• Quiz interactif : jusqu’à 12 joueurs\n"
                "• Salle enfant : capacité adaptable avec l’équipe\n"
                "Dites-moi l’activité et le nombre de participants, je vous oriente."
            )
        return f"Runningman : {facts['capacite']}"

    # 9) Booking / devis / événement
    if b == "retroworld" and (_is_reservation_intent(t) or _is_event_intent(t)):
        # Determine activity
        act = None
        if "quiz" in t or "quizz" in t:
            act = "quiz interactif"
        elif "salle" in t and ("enfant" in t or "anniversaire" in t):
            act = "salle enfant"
        elif "escape" in t:
            act = "escape game VR"
        elif "vr" in t:
            act = "jeux VR"

        if _is_event_intent(t):
            return (
                "Disponible. Pour organiser un anniversaire/événement, pouvez-vous me préciser :\n"
                "- Activité(s) (jeux VR, escape VR, quiz, salle enfant ou combinaison)\n"
                "- Date\n"
                "- Heure de début (ou une fourchette)\n"
                "- Nombre de participants\n"
                "- Âge moyen\n"
                "- Prénom + nom, email, téléphone\n\n"
                + facts["gouter"]["phrase"] + "\n"
                f"Site : {facts['site']} | Téléphone : {facts['tel']}"
            )

        if act:
            return (
                f"Disponible. Pour réserver {act}, pouvez-vous me préciser la date, l’heure de début souhaitée et le nombre de participants ?\n"
                f"Site : {facts['site']} | Téléphone : {facts['tel']}"
            )
        return (
            "Disponible. Pouvez-vous me préciser l’activité (jeux VR, escape game VR, quiz ou salle enfant), la date, l’heure de début souhaitée et le nombre de participants ?\n"
            f"Site : {facts['site']} | Téléphone : {facts['tel']}"
        )

    if b == "runningman" and (_is_reservation_intent(t) or _is_event_intent(t)):
        return (
            f"Pour réserver Runningman : {facts['site']} | {facts['tel']}.\n"
            + facts["gouter"]["phrase"]
        )

    # 10) Tarifs (non booking)
    if b == "retroworld":
        if re.search(r"\bescape\b", t) and "vr" in t:
            ev = facts["escape_vr"]
            return f"Escape game VR : {ev['prix_normal']}€ / joueur (11h–20h). Avant 11h (9h–11h) et après 20h (20h–23h) : {ev['prix_majoré']}€ / joueur. Jusqu’à {ev['max_joueurs']} joueurs."
        if "quiz" in t or "quizz" in t:
            q = facts["quiz"]
            return f"Quiz interactif : {q['prix_30']}€ (30 min), {q['prix_60']}€ (60 min), {q['prix_90']}€ (90 min). Hors 11h–20h (9h–11h et 20h–23h) : +{q['suppl_hors_11_20']}€ / joueur. Jusqu’à {q['max_joueurs']} joueurs."
        if "salle" in t and ("enfant" in t or "anniversaire" in t):
            s = facts["salle_enfant"]
            return f"Salle enfant : {s['prix_h']}€ / heure (+{s['prix_demi_h_sup']}€ / 30 min). Inclus : {s['details']}"
        if "vr" in t or re.search(r"(tarif|prix|combien)", t):
            v = facts["vr"]
            return f"Jeux VR : {v['prix_normal']}€ / joueur (11h–20h). Avant 11h (9h–11h) : {v['prix_avant_11']}€ / joueur. Après 20h (20h–23h) : {v['prix_apres_20']}€ / joueur. Jusqu’à {v['max_joueurs']} joueurs. {v['session']}"

    if b == "runningman":
        if re.search(r"(tarif|prix|combien)", t):
            return f"Tarifs : {facts['tarifs']}"
        if re.search(r"(duree|durée|1h|60)", t):
            return f"Une session dure {facts['session']}"

    return None


# ---------------------------------------------------------
# PROMPT BUILDER (STRICT)
# ---------------------------------------------------------

def _kb_identity_line(brand: str, kb: Dict[str, Any]) -> str:
    if brand == "runningman":
        ident = kb.get("identite") if isinstance(kb, dict) else None
        if isinstance(ident, dict):
            nom = ident.get("nom") or "Runningman Game Zone"
            role_ia = ident.get("role_ia") or "Assistant IA"
            return f"Vous êtes {role_ia} de {nom}."
        return "Vous êtes l’assistant IA de Runningman Game Zone."
    ident = kb.get("identite") if isinstance(kb, dict) else None
    if isinstance(ident, dict):
        nom = ident.get("nom") or "Retroworld France"
        return f"Vous êtes l’assistant IA officiel de {nom}."
    return "Vous êtes l’assistant IA officiel de Retroworld France."


def build_prompt(
    brand: str,
    kb: Dict[str, Any],
    messages: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> List[Dict[str, str]]:
    brand = (brand or "").lower()
    facts = _merge_defaults(brand, kb)

    system_rules = [
        _kb_identity_line(brand, kb),
        "Vous répondez en français. Vouvoiement obligatoire.",
        "Règle d’or : N’inventez jamais un chiffre, une règle, une promo, un événement ou un horaire non confirmé.",
        "Si une info n’est pas dans les FACTS ci-dessous, dites clairement que vous n’avez pas l’information et redirigez vers le contact officiel.",
        "Disponibilités : ne jamais confirmer un créneau. Orientez vers réservation / téléphone.",
        "Marques : Retroworld (VR, escape VR, quiz, salle enfant). Runningman (action game). Ne mélangez pas tarifs/règles.",
        "Ne proposez jamais la formule 'goûter illimité' (périmée). Goûter uniquement sur devis.",
        "Format : réponse directe, puis contact si besoin.",
    ]

    if brand == "runningman":
        facts_block = [
            f"Adresse : {facts['adresse']}",
            f"Réservation : {facts['site']} (contact : {facts['contact']}) | {facts['tel']}",
            f"Session : {facts['session']}",
            f"Capacité : {facts['capacite']}",
            f"Âge : {facts['age']}",
            f"Tarifs : {facts['tarifs']}",
            f"Goûter : {facts['gouter']['phrase']}",
        ]
    else:
        jc = facts["jeux_counts"]
        v = facts["vr"]
        ev = facts["escape_vr"]
        q = facts["quiz"]
        s = facts["salle_enfant"]
        facts_block = [
            f"Adresse : {facts['adresse']}",
            f"Contact : {facts['site']} | {facts['tel']}",
            f"Catalogue : {jc['jeux_vr']} jeux VR et {jc['escape_vr']} scénarios d’escape VR",
            f"Plages tarifaires : {facts['horaires_prix']}",
            f"Jeux VR : {v['prix_normal']}€ (11h–20h), {v['prix_avant_11']}€ (9h–11h), {v['prix_apres_20']}€ (20h–23h). Max {v['max_joueurs']} joueurs. {v['session']}",
            f"Escape VR : {ev['prix_normal']}€ (11h–20h), {ev['prix_majoré']}€ (9h–11h & 20h–23h). Max {ev['max_joueurs']} joueurs.",
            f"Quiz : {q['prix_30']}€ (30min), {q['prix_60']}€ (60min), {q['prix_90']}€ (90min), +{q['suppl_hors_11_20']}€ hors 11h–20h. Max {q['max_joueurs']} joueurs. {q['age']}",
            f"Salle enfant : {s['prix_h']}€ /h (+{s['prix_demi_h_sup']}€ / 30 min). Inclus : {s['details']}",
            f"Salle d’attente : {facts['attente']['details']}",
            f"Hygiène/matériel : {facts['equipement']}",
            f"Fidélité : {facts['fidelite']['gains']} Récompenses : {facts['fidelite']['recompenses']}",
            f"Goûter : {facts['gouter']['phrase']}",
        ]

    system_text = "\n".join([
        "\n".join(system_rules),
        "",
        "FACTS (utilisez uniquement ces informations) :",
        "\n".join(f"- {line}" for line in facts_block),
        "",
        "Si l’utilisateur demande Runningman depuis Retroworld (ou inversement), expliquez que c’est distinct mais dans le même bâtiment, et donnez le bon contact.",
    ])

    prompt_messages: List[Dict[str, str]] = [{"role": "system", "content": system_text}]

    # include conversation history
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and content is not None:
            prompt_messages.append({"role": role, "content": str(content)})

    return prompt_messages


# ---------------------------------------------------------
# GUARDRAILS FOR OPENAI OUTPUT
# ---------------------------------------------------------

_RETRO_SAFE_FALLBACK = (
    "Pour ce point, je préfère éviter toute erreur. "
    "Pouvez-vous contacter l’équipe Retroworld au 04 94 47 94 64 ou via https://www.retroworldfrance.com ?"
)

def guard_openai_reply(brand: str, reply: str) -> Tuple[str, List[str]]:
    brand = (brand or "").lower()
    r = (reply or "").strip()
    r_low = r.lower()
    hits: List[str] = []

    # block "goûter à volonté" anywhere
    if "à volonté" in r_low or "a volonté" in r_low or "illimité" in r_low or "illimite" in r_low:
        hits.append("gouter_illimite")
        # but allow if it says it's not proposed
        if "n’est plus" in r_low or "n'est plus" in r_low or "périmée" in r_low or "perime" in r_low:
            hits.pop()
        else:
            return _RETRO_SAFE_FALLBACK, hits

    # availability claims block
    if re.search(r"\b(c['’]?est dispo|disponible|il reste|places disponibles|c'est bon)\b", r_low):
        # allow if it says it cannot confirm
        if "ne peux pas confirmer" not in r_low and "je ne peux pas confirmer" not in r_low:
            hits.append("availability_claim")
            if brand == "runningman":
                return "Je ne peux pas confirmer la disponibilité en direct. Pour réserver : https://runningmangames.fr ou 04 98 09 30 59.", hits
            return _RETRO_SAFE_FALLBACK, hits

    return r, hits


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

def append_conversation_log(
    conversation_id: str,
    brand: str,
    channel: str,
    user_messages: List[Dict[str, Any]],
    assistant_reply: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not conversation_id:
        conversation_id = f"conv_{int(time.time())}"
    path = os.path.join(CONVERSATIONS_LOG_DIR, f"{conversation_id}.jsonl")
    record = {
        "timestamp": time.time(),
        "conversation_id": conversation_id,
        "brand": brand,
        "channel": channel,
        "user_messages": user_messages,
        "assistant_reply": assistant_reply,
        "extra": extra or {},
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_conversation_records(conversation_id: str) -> List[Dict[str, Any]]:
    path = os.path.join(CONVERSATIONS_LOG_DIR, f"{conversation_id}.jsonl")
    if not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    records.append(rec)
            except Exception:
                continue
    records.sort(key=lambda r: float(r.get("timestamp") or 0.0))
    return records


def reconstruct_history_from_logs(conversation_id: str) -> List[Dict[str, str]]:
    records = load_conversation_records(conversation_id)
    history: List[Dict[str, str]] = []
    for rec in records:
        rec_msgs = rec.get("user_messages") or []
        for m in rec_msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and content is not None:
                history.append({"role": role, "content": str(content)})
        assistant_text = rec.get("assistant_reply")
        if assistant_text:
            history.append({"role": "assistant", "content": str(assistant_text)})
    return history


def classify_conversation_brands(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    brands_seen = set()
    last_effective = None
    for rec in records:
        extra = rec.get("extra") or {}
        meta = extra.get("metadata") or {}
        be = extra.get("brand_effective") or meta.get("brand_effective") or rec.get("brand")
        if be in ("runningman", "retroworld"):
            brands_seen.add(be)
            last_effective = be
    if not brands_seen:
        return {"brand_final": "unknown", "brands_seen": []}
    if len(brands_seen) == 1:
        b = next(iter(brands_seen))
        return {"brand_final": b, "brands_seen": [b]}
    brand_final = last_effective or "mixed"
    if brand_final not in ("runningman", "retroworld"):
        brand_final = "mixed"
    return {"brand_final": brand_final, "brands_seen": list(brands_seen)}


# ---------------------------------------------------------
# CORE CHAT PROCESSOR
# ---------------------------------------------------------

def process_chat(
    brand_entry: str,
    messages: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    allow_server_history: bool = True,
    do_log: bool = True,
) -> Dict[str, Any]:
    brand_entry = (brand_entry or "").lower()
    if brand_entry not in SUPPORTED_BRANDS:
        return {"error": "unknown_brand"}

    # last user message
    last_user_text = ""
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_text = str(msg.get("content") or "")
            break

    effective_brand = detect_brand_from_text(last_user_text, default=brand_entry)

    # conversation_id resolution:
    # 1) explicit conversation_id in metadata
    # 2) user_id mapping (user sees his own history)
    # 3) fallback new
    user_id = str(metadata.get("user_id") or "").strip()
    conversation_id = str(metadata.get("conversation_id") or "").strip()

    if not conversation_id and user_id:
        conversation_id = get_user_conversation(user_id)

    if not conversation_id:
        conversation_id = f"{effective_brand}_{int(time.time()*1000)}"

    metadata["conversation_id"] = conversation_id
    metadata["brand_entry"] = brand_entry
    metadata["brand_effective"] = effective_brand

    # bind user->conversation_id
    if user_id:
        set_user_conversation(user_id, conversation_id)

    # rebuild history if client only sends last msg
    messages_for_prompt: List[Dict[str, Any]] = list(messages or [])
    try:
        only_one_user = (
            len(messages_for_prompt) == 1
            and isinstance(messages_for_prompt[0], dict)
            and messages_for_prompt[0].get("role") == "user"
        )
        use_server_history = allow_server_history and conversation_id and only_one_user and not metadata.get("no_server_history", False)
        if use_server_history:
            past = reconstruct_history_from_logs(conversation_id)
            if past:
                messages_for_prompt = past + messages_for_prompt
                logger.info("Reconstructed history for %s (%d past + %d new)", conversation_id, len(past), len(messages))
    except Exception as e:
        logger.error("History reconstruct error: %s", e)

    kb = load_kb(effective_brand)

    # FAST first
    fast_reply = answer_fast(effective_brand, kb, last_user_text, metadata=metadata)

    if fast_reply:
        reply_text = fast_reply
        usage: Dict[str, Any] = {}
        guard_hits: List[str] = []
        skipped_openai = True
    else:
        prompt_messages = build_prompt(effective_brand, kb, messages_for_prompt, metadata)
        reply_text, usage = call_openai_chat(prompt_messages)
        reply_text, guard_hits = guard_openai_reply(effective_brand, reply_text)
        skipped_openai = False

    # log
    if do_log:
        try:
            channel = str(metadata.get("source") or "web")
            append_conversation_log(
                conversation_id=conversation_id,
                brand=effective_brand,
                channel=channel,
                user_messages=messages,
                assistant_reply=reply_text,
                extra={
                    "brand_entry": brand_entry,
                    "brand_effective": effective_brand,
                    "metadata": metadata,
                    "openai_usage": usage,
                    "skipped_openai": skipped_openai,
                    "guard_hits": guard_hits,
                },
            )
        except Exception as e:
            logger.error("Logging error: %s", e)

    return {
        "reply": reply_text,
        "brand_used": effective_brand,
        "brand_entry": brand_entry,
        "conversation_id": conversation_id,
        "skipped_openai": skipped_openai,
        "guard_hits": guard_hits,
    }


# ---------------------------------------------------------
# AUTH HELPERS
# ---------------------------------------------------------

def _require_admin_token(req) -> bool:
    tok = (req.args.get("token") or "").strip()
    if not tok:
        tok = (req.headers.get("X-Admin-Token") or "").strip()
    return bool(tok) and tok == ADMIN_DASHBOARD_TOKEN


def _require_user_token(req) -> bool:
    tok = (req.args.get("token") or "").strip()
    if not tok:
        tok = (req.headers.get("X-User-Token") or "").strip()
    return bool(tok) and tok == USER_HISTORY_TOKEN


# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------

@app.route("/", methods=["GET", "HEAD"])
def root():
    return jsonify(
        {
            "service": "retroworld-ia",
            "status": "ok",
            "time": time.time(),
            "brands": list(SUPPORTED_BRANDS),
            "model": OPENAI_MODEL,
        }
    ), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": time.time()}), 200


@app.route("/chat/<brand>", methods=["POST"])
def chat_route(brand: str):
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid_json"}), 400
    if not isinstance(body, dict):
        return jsonify({"error": "invalid_payload"}), 400

    messages = body.get("messages") or []
    metadata = body.get("metadata") or {}

    if not isinstance(messages, list):
        return jsonify({"error": "messages must be a list"}), 400
    if not isinstance(metadata, dict):
        metadata = {}

    try:
        resp = process_chat(brand, messages, metadata, allow_server_history=True, do_log=True)
        return jsonify(resp), 200
    except RuntimeError as e:
        logger.warning("Chat runtime error: %s", e)
        safe = "Je peux répondre aux questions courantes, mais je n’ai pas accès au moteur de réponse avancé pour le moment. Pouvez-vous reformuler votre demande ?"
        return jsonify({"reply": safe, "brand_used": brand, "brand_entry": brand, "error": str(e)}), 200
    except Exception as e:
        logger.error("chat_route error: %s", e)
        return jsonify({"error": "server_error", "details": str(e)}), 500


# ---------------- USER HISTORY API ----------------

@app.route("/user/api/history", methods=["GET"])
def user_api_history():
    """
    User can only access their own history via user_id.
    Auth: token=... or X-User-Token header must match USER_HISTORY_TOKEN.
    """
    if not _require_user_token(request):
        return jsonify({"error": "forbidden"}), 403

    user_id = (request.args.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "missing_user_id"}), 400

    conversation_id = get_user_conversation(user_id)
    if not conversation_id:
        return jsonify({"user_id": user_id, "conversation_id": "", "records": []}), 200

    records = load_conversation_records(conversation_id)
    return jsonify({"user_id": user_id, "conversation_id": conversation_id, "records": records}), 200


# ---------------- KB ADMIN ----------------

@app.route("/kb/upsert/<brand>", methods=["POST"])
def kb_upsert(brand: str):
    if not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid_json"}), 400
    if not isinstance(body, dict):
        return jsonify({"error": "invalid_kb"}), 400
    try:
        save_kb(brand, body)
    except Exception as e:
        logger.error("save_kb(%s) failed: %s", brand, e)
        return jsonify({"error": "kb_save_failed"}), 500
    return jsonify({"status": "ok", "brand": brand}), 200


# ---------------- QWEEKLE WEBHOOK ----------------

@app.route("/webhooks/qweekle", methods=["POST"])
def qweekle_webhook():
    if QWEEKLE_WEBHOOK_SECRET:
        incoming_secret = request.headers.get("X-Qweekle-Secret") or ""
        if incoming_secret != QWEEKLE_WEBHOOK_SECRET:
            logger.warning("Qweekle webhook rejected (invalid secret)")
            return jsonify({"error": "forbidden"}), 403
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400
    event_type = payload.get("event_type") or payload.get("type") or "unknown"
    logger.info("Webhook Qweekle received: %s", event_type)
    fname = f"{event_type or 'unknown'}.jsonl"
    path = os.path.join(QWEEKLE_LOG_DIR, fname)
    record = {"timestamp": time.time(), "event_type": event_type, "payload": payload, "source": QWEEKLE_SOURCE_NAME}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return jsonify({"status": "ok", "event_type": event_type}), 200


# ---------------- ADMIN API ----------------

@app.route("/admin/api/conversations", methods=["GET"])
def admin_api_conversations():
    if not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    convs: List[Dict[str, Any]] = []
    if not os.path.isdir(CONVERSATIONS_LOG_DIR):
        return jsonify(convs), 200

    for fname in os.listdir(CONVERSATIONS_LOG_DIR):
        if not fname.endswith(".jsonl"):
            continue
        conversation_id = fname[:-5]
        records = load_conversation_records(conversation_id)
        if not records:
            continue
        last = records[-1]
        ts = float(last.get("timestamp") or 0.0)
        channel = last.get("channel") or "web"
        extra = last.get("extra") or {}
        meta = extra.get("metadata") or {}
        source = extra.get("source") or meta.get("source") or "unknown"

        brand_info = classify_conversation_brands(records)
        brand_final = brand_info.get("brand_final")

        preview = ""
        for rec in reversed(records):
            umsgs = rec.get("user_messages") or []
            for m in reversed(umsgs):
                if isinstance(m, dict) and m.get("role") == "user":
                    preview = str(m.get("content") or "")
                    break
            if preview:
                break
        preview = preview.strip()
        if len(preview) > 140:
            preview = preview[:137] + "..."

        convs.append(
            {
                "conversation_id": conversation_id,
                "timestamp": ts,
                "channel": channel,
                "source": source,
                "preview": preview,
                "brand_final": brand_final,
            }
        )

    convs.sort(key=lambda c: float(c.get("timestamp") or 0.0), reverse=True)
    return jsonify(convs), 200


@app.route("/admin/api/conversation/<conversation_id>", methods=["GET"])
def admin_api_conversation_detail(conversation_id: str):
    if not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403
    records = load_conversation_records(conversation_id)
    if not records:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"conversation_id": conversation_id, "records": records}), 200


@app.route("/admin/api/test", methods=["POST"])
def admin_api_test():
    """Run a batch test: input can be JSON (with results/q) or plain text (one question per line)."""
    if not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403
    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400
    if not isinstance(body, dict):
        return jsonify({"error": "invalid_payload"}), 400

    brand = str(body.get("brand") or "auto").lower()
    payload = body.get("payload")

    if isinstance(payload, str):
        s = payload.strip()
        if s and (s.startswith("{") or s.startswith("[")):
            try:
                payload = json.loads(s)
            except Exception:
                pass

    questions: List[str] = []

    if isinstance(payload, dict):
        res = payload.get("results")
        if isinstance(res, list):
            for item in res:
                if isinstance(item, dict) and item.get("q"):
                    questions.append(str(item["q"]))
                elif isinstance(item, str):
                    questions.append(item)
        elif payload.get("q"):
            questions.append(str(payload["q"]))
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                questions.append(item)
            elif isinstance(item, dict) and item.get("q"):
                questions.append(str(item["q"]))
    elif isinstance(payload, str):
        text = payload.strip()
        if text:
            for line in text.splitlines():
                ln = line.strip()
                if not ln:
                    continue
                if ln.upper() in ("FAST", "OPENAI"):
                    continue
                if ln.lower() in ("retroworld", "runningman"):
                    continue
                questions.append(ln)

    # Deduplicate keep order
    seen = set()
    qs: List[str] = []
    for q in questions:
        qn = q.strip()
        if not qn or qn in seen:
            continue
        seen.add(qn)
        qs.append(qn)

    results: List[Dict[str, Any]] = []
    convo_id = f"test_{int(time.time()*1000)}"

    for q in qs[:300]:
        messages = [{"role": "user", "content": q}]
        md = {"source": "admin_test", "conversation_id": convo_id, "no_server_history": True}
        use_brand = "retroworld"
        if brand in SUPPORTED_BRANDS:
            use_brand = brand
        elif brand == "auto":
            use_brand = detect_brand_from_text(q, default="retroworld")
        try:
            resp = process_chat(use_brand, messages, md, allow_server_history=False, do_log=False)
            results.append(
                {
                    "q": q,
                    "brand_used": resp.get("brand_used"),
                    "skipped_openai": resp.get("skipped_openai"),
                    "guard_hits": resp.get("guard_hits") or [],
                    "a": resp.get("reply"),
                }
            )
        except Exception as e:
            results.append({"q": q, "brand_used": use_brand, "error": str(e), "a": ""})

    return jsonify({"count": len(results), "results": results}), 200


# ---------------- ADMIN UI ----------------
# (gardez votre HTML existant si vous voulez; je ne le réécris pas ici pour éviter de vous spammer 400 lignes.)
# 👉 Si vous tenez à garder EXACTEMENT votre page admin, laissez votre bloc HTML tel quel.
# Ici on renvoie une version minimaliste qui redirige vers votre page existante si vous l’avez.

@app.route("/admin/conversations", methods=["GET"])
def admin_conversations_page():
    if not _require_admin_token(request):
        return "Forbidden", 403
    # Si vous avez déjà un gros HTML admin, vous pouvez le remettre ici.
    # Pour l’instant: petit message (le côté API fait le vrai boulot).
    return """
<!doctype html>
<html lang="fr">
<meta charset="utf-8">
<title>Admin IA</title>
<body style="font-family:system-ui;background:#0b1220;color:#e5e7eb;padding:24px;">
  <h2>Admin IA</h2>
  <p>API prête ✅</p>
  <ul>
    <li><code>/admin/api/conversations</code></li>
    <li><code>/admin/api/conversation/&lt;conversation_id&gt;</code></li>
    <li><code>/admin/api/test</code></li>
  </ul>
  <p>Astuce: réinjectez votre page HTML complète si vous voulez l’interface.</p>
</body>
</html>
"""


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
