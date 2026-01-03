"""
Flask application for the Retroworld / Runningman conversational assistant.

Goals (hard requirements):
- Zero hallucination on business rules/prices/capacities: if uncertain -> say so and redirect to official contact.
- Never confirm availability in-chat.
- Avoid brand mixing: detect intent (Retroworld vs Runningman) and answer with the correct rules.
- Provide fast, deterministic answers for the common questions (address, prices, duration, capacity, booking, events…).
- Offer a professional admin dashboard + an integrated test console to debug multi-question payloads quickly.

Runtime notes:
- KB JSON files are loaded from /mnt/data/kb_<brand>.json (overrides) or /app/kb_<brand>.json (embedded).
- Conversation logs are stored as JSONL files in /mnt/data/logs/conversations/.
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

for d in (BASE_LOG_DIR, CONVERSATIONS_LOG_DIR, QWEEKLE_LOG_DIR):
    os.makedirs(d, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
ADMIN_DASHBOARD_TOKEN = os.getenv("ADMIN_DASHBOARD_TOKEN", "changeme_admin_token")

QWEEKLE_WEBHOOK_SECRET = os.getenv("QWEEKLE_WEBHOOK_SECRET", "")
QWEEKLE_SOURCE_NAME = os.getenv("QWEEKLE_SOURCE_NAME", "retroworld-qweekle")

SUPPORTED_BRANDS: set[str] = {"retroworld", "runningman"}


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
    # bust cache
    if brand in _KB_CACHE:
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
    return str(content or ""), usage


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
    "mini jeux", "défis", "defis", "physique", "capteur", "gilet",
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
# FAST ANSWERS (DETERMINISTIC)
# ---------------------------------------------------------

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _retroworld_defaults() -> Dict[str, Any]:
    # Hard facts, to stop hallucinations even if KB is empty or outdated.
    return {
        "adresse": "815 avenue Pierre Brossolette, 83300 Draguignan, France",
        "site": "https://www.retroworldfrance.com",
        "tel": "04 94 47 94 64",
        "vr": {"prix_normal": 15, "prix_majoré": 17, "max_joueurs": 5, "session": "Une session = 1 jeu (au choix dans le catalogue)."},
        "escape_vr": {"prix_normal": 30, "prix_majoré": 35, "max_joueurs": 5},
        "quiz": {"prix_30": 8, "prix_60": 15, "prix_90": 20, "max_joueurs": 12, "age": "Dès 10 ans avec accompagnant."},
        "salle_enfant": {"prix_h": 50, "prix_demi_h_sup": 20, "details": "Jeux en bois, mur interactif, ballayeuse, stockage goûter."},
        "attente": {"details": "Canapés, boissons/snacks, baby-foot, air hockey, borne de basketball, billard (10€/h), écrans pour suivre les sessions.", "billard": "10€ / heure"},
        "equipement": "Casques VR professionnels : Vive Pro 2 et Meta Quest 3. Équipements nettoyés entre chaque session.",
        "jeux_counts": {"jeux_vr": 31, "escape_vr": 28},
        "horaires_prix": "Tarifs standard de 11h à 20h. Tarifs majorés de 9h à 11h et de 20h à 23h.",
        "fidelite": {
            "gains": "1 partie VR = 1 point. 1 escape game VR = 2 points. Pas de points sur les formules anniversaire.",
            "recompenses": "5 points = 1 quiz 30 min offert. 10 points = 1 partie VR offerte. 20 points = 1 escape game VR offert. Goodies échangeables contre des points.",
            "utilisation": "Pour cumuler des points, le client doit présenter son QR code ou informer l’équipe avant de jouer. Pour utiliser ses points, il suffit d’en informer un agent/gamemaster lors de la visite.",
            "consultation": "Points consultables via l’application Retroworld (Android) ou sur le site en se connectant à son compte.",
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
        "events_reply": "Je n’ai pas les informations précises à propos de cet événement. Je vous invite à contacter l’équipe via la page contact : https://runningmangames.fr/contact ou par téléphone au 04 98 09 30 59.",
        "reservation_reply": "Pour réserver, vous pouvez utiliser le site officiel : https://runningmangames.fr. En cas de besoin, vous pouvez aussi appeler le 04 98 09 30 59.",
        "dispo_reply": "Je ne peux pas confirmer la disponibilité en direct. Pour réserver (et confirmer un créneau), utilisez : https://runningmangames.fr. Sinon, appelez le 04 98 09 30 59.",
    }


def _merge_defaults(brand: str, kb: Dict[str, Any]) -> Dict[str, Any]:
    base = _retroworld_defaults() if brand == "retroworld" else _runningman_defaults()
    if not isinstance(kb, dict):
        return base

    # overlay a few known KB fields if present (non-breaking)
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

        # tariffs
        prix_enfant = _get_nested(kb, "tarification.action_game.enfant.prix")
        prix_adulte = _get_nested(kb, "tarification.action_game.adulte_accompagnateur.prix")
        if isinstance(prix_enfant, (int, float)) and isinstance(prix_adulte, (int, float)):
            base["tarifs"] = f"{int(prix_enfant)}€ / personne (moins de 12 ans) et {int(prix_adulte)}€ / personne (12 ans et + / adulte)."

        # session duration
        duree = _get_nested(kb, "horaires_et_creneaux.duree_session")
        if isinstance(duree, str) and duree.strip():
            base["session"] = duree.strip()

        cap = _get_nested(kb, "horaires_et_creneaux.capacite.maximum_par_heure")
        if isinstance(cap, (int, float)):
            base["capacite"] = f"Jusqu’à {int(cap)} personnes par heure (organisation selon réservation)."

    else:
        # Retroworld: use memory/confirmed facts; if KB provides more, we can overlay carefully.
        site = _get_nested(kb, "liens_officiels.site")
        if isinstance(site, str) and site.strip():
            base["site"] = site.strip()
        tel = _get_nested(kb, "contact.telephone")
        if isinstance(tel, str) and tel.strip():
            base["tel"] = tel.strip()

        # counts
        n_vr = kb.get("nombre_jeux_vr")
        n_escape = kb.get("nombre_escape_vr")
        if isinstance(n_vr, int):
            base["jeux_counts"]["jeux_vr"] = n_vr
        if isinstance(n_escape, int):
            base["jeux_counts"]["escape_vr"] = n_escape

        # pricing blocks if available
        tarifs = kb.get("tarifs")
        if isinstance(tarifs, dict):
            vr = tarifs.get("vr") if isinstance(tarifs.get("vr"), dict) else {}
            if isinstance(vr, dict):
                pn = vr.get("horaire_normal")
                pm = vr.get("horaire_supplement")
                if isinstance(pn, (int, float)):
                    base["vr"]["prix_normal"] = int(pn)
                if isinstance(pm, (int, float)):
                    base["vr"]["prix_majoré"] = int(pm)

            escape = tarifs.get("escape_vr") if isinstance(tarifs.get("escape_vr"), dict) else {}
            if isinstance(escape, dict):
                pn = escape.get("horaire_normal")
                pm = escape.get("horaire_supplement")
                if isinstance(pn, (int, float)):
                    base["escape_vr"]["prix_normal"] = int(pn)
                if isinstance(pm, (int, float)):
                    base["escape_vr"]["prix_majoré"] = int(pm)

            quiz = tarifs.get("quiz") if isinstance(tarifs.get("quiz"), dict) else {}
            if isinstance(quiz, dict):
                if isinstance(quiz.get("30min"), (int, float)):
                    base["quiz"]["prix_30"] = int(quiz["30min"])
                if isinstance(quiz.get("60min"), (int, float)):
                    base["quiz"]["prix_60"] = int(quiz["60min"])
                if isinstance(quiz.get("90min"), (int, float)):
                    base["quiz"]["prix_90"] = int(quiz["90min"])

            salle = tarifs.get("salle_enfant") if isinstance(tarifs.get("salle_enfant"), dict) else {}
            if isinstance(salle, dict):
                if isinstance(salle.get("heure"), (int, float)):
                    base["salle_enfant"]["prix_h"] = int(salle["heure"])
                if isinstance(salle.get("demi_heure_supplement"), (int, float)):
                    base["salle_enfant"]["prix_demi_h_sup"] = int(salle["demi_heure_supplement"])

    return base


def _is_reservation_intent(t: str) -> bool:
    t = _norm(t)
    return any(k in t for k in [
        "réserver", "reserver", "reservation", "réservation", "bloquer", "creneau", "créneau",
        "lien", "dispo", "disponible", "place", "places", "complet", "complets",
    ])


def _is_event_intent(t: str) -> bool:
    t = _norm(t)
    return any(k in t for k in [
        "halloween", "saint-sylvestre", "saint sylvestre", "nouvel an", "noël", "noel",
        "ramadan", "aïd", "aid", "eid", "pâques", "paques", "toussaint", "hanouka",
        "kippour", "diwali", "jour férié", "jour ferie", "week-end férié", "week end ferie",
        "événement", "evenement", "soirée spéciale", "soiree speciale",
    ])


def answer_fast(brand: str, user_text: str) -> str | None:
    """Heuristic answers (no OpenAI call) for common questions.

    IMPORTANT:
    - For booking intent, always start with "Disponible." (Retroworld rules).
    - Never claim a slot is confirmed.
    - Never exceed 5 players simultaneously for VR / Escape VR.
    """
    b = (brand or "").lower().strip()
    t = _norm(user_text or "")

    # KB / defaults
    if b == "runningman":
        base = _merge_defaults(KB_RUNNINGMAN, _DEFAULTS_RUNNINGMAN)
    else:
        b = "retroworld"
        base = _merge_defaults(KB_RETROWORLD, _DEFAULTS)

    # ----------------------------
    # 0) Direct booking links (Retroworld only)
    # ----------------------------
    if b == "retroworld":
        if re.search(r"\b(lien|link)\b", t) and re.search(r"\b(resa|réserv|reservation|réservation|book)\b", t):
            # Activity already specified?
            if "escape" in t:
                return "Parfait, je vous laisse réserver via notre lien.\nhttps://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=escape%20game&lang=fr"
            if "quiz" in t or "quizz" in t:
                return "Parfait, je vous laisse réserver via notre lien.\nhttps://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=quizz&lang=fr"
            if "vr" in t:
                return "Parfait, je vous laisse réserver via notre lien.\nhttps://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=Jeu%20%C3%A0%20la%20partie&lang=fr"
            return "Pour quelle activité souhaitez-vous le lien : jeux VR, escape game VR, quiz interactif ou salle enfant ?"

    # ----------------------------
    # 1) Adresse / localisation
    # ----------------------------
    if re.search(r"\b(adresse|où|ou\s+etes\s+vous|vous\s+etes\s+ou|c\s*est\s+ou|localisation)\b", t):
        # Both are in the same building (address always identical)
        return f"Adresse : {base['adresse']}"

    # ----------------------------
    # 2) Horaires (and price bands)
    # ----------------------------
    if re.search(r"(horaire|horaires|ouvert|ouverture|ferme|fermé|fermee|dimanche)", t):
        if b == "retroworld":
            return f"{base.get('horaires_prix','Horaires : 11h–20h en tarif standard. Avant 11h et après 20h : tarif majoré.')}\nAdresse : {base['adresse']}"
        # Runningman: we avoid inventing exact daily schedules
        return f"Runningman Game Zone est au {base['adresse']}. Pour les horaires exacts et les disponibilités, merci de consulter {base.get('site_web','https://www.runningmangames.fr')} ou d’appeler le {base.get('telephone','04 98 09 30 59')}."

    # ----------------------------
    # 3) Fidélité (must be checked BEFORE "VR" keywords)
    # ----------------------------
    if b == "retroworld" and re.search(r"\b(fid[eé]lit[eé]|points?)\b", t):
        return (
            "Programme fidélité :\n"
            "• 1 partie de jeux VR = 1 point\n"
            "• 1 escape game VR = 2 points\n"
            "• Pas de points sur les formules anniversaire\n"
            "Récompenses : 5 points = 1 quiz 30 min offert, 10 points = 1 partie VR offerte, 20 points = 1 escape game VR offert."
        )

    # ----------------------------
    # 4) Goûter / stockage / anniversaire (info)
    # ----------------------------
    if re.search(r"(gouter|goûter|gateau|gâteau|frigo|stock|stocker)", t):
        if b == "retroworld":
            # Never mention an unlimited formula as available
            if "a volonte" in t or "à volonté" in user_text.lower():
                return (
                    "La formule de goûter illimité n’est plus proposée.\n"
                    "En revanche, vous pouvez stocker un gâteau/goûter sur place, et un goûter peut être proposé sur devis (préparé par Runningman et géré/commercialisé par Retroworld)."
                )
            return "Oui, vous pouvez stocker un gâteau/goûter sur place. Et si vous le souhaitez, un goûter peut être proposé sur devis (préparé par Runningman et géré/commercialisé par Retroworld)."
        # Runningman: goûter managed by Retroworld
        return "Oui, vous pouvez stocker un gâteau/goûter sur place. Pour un goûter sur devis (préparé par Runningman), c’est géré par Retroworld : 04 94 47 94 64."

    # ----------------------------
    # 5) Paiement
    # ----------------------------
    if re.search(r"(ticket\s*resto|tickets\s*resto|restaurant|cheque\s*vac|ch[eè]ques\s*vac|ancv|especes|esp[eè]ces|carte\b)", t):
        # We only answer what we know for sure from KB defaults
        if "ticket" in t:
            return "Tickets resto : non."
        if re.search(r"(cheque\s*vac|ch[eè]ques\s*vac|ancv)", t):
            return "Chèques vacances : oui."
        if re.search(r"(especes|esp[eè]ces)", t):
            return "Oui, vous pouvez payer en espèces sur place."
        if "carte" in t:
            return "Oui, vous pouvez payer par carte bancaire sur place."

    # ----------------------------
    # 6) Écrans / salle d’attente
    # ----------------------------
    if re.search(r"(salle\s*d\s*attente|attente|ecran|écran|regarder|diffus)", t):
        if b == "retroworld":
            return "Oui, vous pouvez patienter dans la salle d’attente (canapés, snacks/boissons, baby-foot, air hockey, borne de basketball, billard). Les écrans diffusent la vue du jeu (pas le joueur)."
        return "Oui, vous pouvez patienter sur place. Pour les détails sur l’accueil/salle d’attente, merci de contacter Runningman : 04 98 09 30 59."

    # ----------------------------
    # 7) Casques / hygiène
    # ----------------------------
    if b == "retroworld" and re.search(r"(casque|headset|vive|quest|meta|netto|d[eé]sinfect|lunettes)", t):
        return (
            "Casques VR : Vive Pro 2 et Meta Quest 3 (ou équivalent selon l’expérience).\n"
            "Hygiène : les casques et équipements sont nettoyés entre chaque session.\n"
            "Lunettes : oui, c’est possible (on vous aide à ajuster le casque)."
        )

    # ----------------------------
    # 8) Questions capacité / joueurs (generic)
    # ----------------------------
    if re.search(r"(combien\s+de\s+joueurs|max|on\s+peut\s+[eê]tre|a\s+\d+|on\s+est\s+\d+|solo|duo|trio)", t):
        if b == "retroworld":
            return (
                "Selon l’activité :\n"
                "• Jeux VR : 1 à 5 joueurs simultanés\n"
                "• Escape game VR : jusqu’à 5 joueurs (souvent 2 à 5 selon le scénario)\n"
                "• Quiz interactif : jusqu’à 12 joueurs\n"
                "• Salle enfant : capacité adaptable avec l’équipe\n"
                "Dites-moi l’activité et le nombre de participants, je vous oriente."
            )
        return "Runningman Game Zone : jusqu’à 25 personnes par session (60 minutes). Au-delà, c’est sur demande/devis."

    # ----------------------------
    # 9) Booking / devis / créneau (Retroworld rules)
    # ----------------------------
    if b == "retroworld" and (_is_reservation_intent(t) or _is_event_intent(t)):
        # Try to detect the activity to avoid re-asking it
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
                "Info goûter : vous pouvez stocker un gâteau/goûter sur place, et un goûter peut être proposé sur devis (préparé par Runningman et géré/commercialisé par Retroworld).\n"
                "Site : https://www.retroworldfrance.com | Téléphone : 04 94 47 94 64"
            )

        # Simple reservation (no event)
        if act:
            return (
                f"Disponible. Pour réserver {act}, pouvez-vous me préciser la date, l’heure de début souhaitée, le nombre de participants et l’âge des enfants (s’il y en a) ?\n"
                "Site : https://www.retroworldfrance.com | Téléphone : 04 94 47 94 64."
            )
        return "Disponible. Pouvez-vous me préciser l’activité (jeux VR, escape game VR, quiz ou salle enfant), la date, l’heure de début souhaitée et le nombre de participants ?\nSite : https://www.retroworldfrance.com | Téléphone : 04 94 47 94 64."

    if b == "runningman" and (_is_reservation_intent(t) or _is_event_intent(t)):
        return f"Pour réserver Runningman : {base.get('site_web','https://www.runningmangames.fr')} | {base.get('telephone','04 98 09 30 59')}.\nPour le goûter sur devis (préparé par Runningman), c’est géré par Retroworld : 04 94 47 94 64."

    # ----------------------------
    # 10) Activity details (non-booking)
    # ----------------------------
    if b == "retroworld":
        # Escape VR must be checked BEFORE generic VR
        if re.search(r"\bescape\b", t) and "vr" in t:
            return "Escape game VR : 30€ / joueur (tarif standard 11h–20h). Avant 11h (9h–11h) et après 20h (20h–23h) : 35€ / joueur. Jusqu’à 5 joueurs."
        if "quiz" in t or "quizz" in t:
            return "Quiz interactif : 8€ (30 min), 15€ (60 min), 20€ (90 min). Hors 11h–20h (9h–11h et 20h–23h) : +5€ / joueur. Jusqu’à 12 joueurs."
        if "salle" in t and ("enfant" in t or "anniversaire" in t):
            return "Salle enfant : 50€ / heure (+20€ / demi-heure supplémentaire). Inclus : jeux en bois, mur interactif, balayeuse, espace goûter."
        if "vr" in t:
            return "Jeux VR : 15€ / joueur (tarif standard 11h–20h). Avant 11h (9h–11h) : 20€ / joueur. Après 20h (20h–23h) : 17€ / joueur. Jusqu’à 5 joueurs. Une session = 1 jeu (au choix dans le catalogue)."


    # Runningman basics
    if b == "runningman":
        if re.search(r"(tarif|prix|combien)", t):
            return "Tarifs : 15€ / personne (moins de 12 ans) et 20€ / personne (12 ans et +)."
        if re.search(r"(duree|durée|1h|60)", t):
            return "Une session dure 60 minutes de jeu."

    return None

def _kb_identity_line(brand: str, kb: Dict[str, Any]) -> str:
    if brand == "runningman":
        ident = kb.get("identite") if isinstance(kb, dict) else None
        if isinstance(ident, dict):
            nom = ident.get("nom") or "Runningman Game Zone"
            role_ia = ident.get("role_ia") or "Assistant IA"
            return f"Vous êtes {role_ia} de {nom}."
        return "Vous êtes l’assistant IA de Runningman Game Zone."
    else:
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
    """
    Build a strict system prompt:
    - Use ONLY the facts below. If a fact isn't present, say you don't know and redirect to official contact.
    - Never confirm availability in chat.
    - Avoid mixing brands.
    """
    brand = (brand or "").lower()
    facts = _merge_defaults(brand, kb)

    system_rules = [
        _kb_identity_line(brand, kb),
        "Vous répondez en français. Vouvoiement obligatoire.",
        "Règle d’or : N’inventez jamais un chiffre, une règle, une promo, un événement ou un horaire non confirmé.",
        "Si une info n’est pas dans les FACTS ci-dessous, dites clairement que vous n’avez pas l’information et redirigez vers le contact officiel.",
        "Disponibilités : ne jamais confirmer un créneau ou dire 'il reste de la place'. Orientez vers réservation / téléphone.",
        "Marques : Retroworld (VR, escape VR, quiz, salle enfant). Runningman (action game, mini-jeux physiques). Ne mélangez pas tarifs/règles.",
        "Format : 1) réponse directe. 2) si réservation demandée -> lien officiel. 3) téléphone en secours.",
        "Ne pas ajouter d’informations non demandées (sauf le téléphone quand on parle réservation).",
    ]

    # Brand-specific contacts + key facts
    if brand == "runningman":
        facts_block = [
            f"Adresse : {facts['adresse']}",
            f"Réservation : {facts['site']} (contact : {facts['contact']}) | {facts['tel']}",
            f"Session : {facts['session']}",
            f"Capacité : {facts['capacite']}",
            f"Âge : {facts['age']}",
            f"Tarifs : {facts['tarifs']}",
            "Événements/horaires fériés : si demandé, répondre que vous n’avez pas l’info précise et rediriger vers la page contact ou téléphone.",
        ]
    else:
        jc = facts["jeux_counts"]
        facts_block = [
            f"Adresse : {facts['adresse']}",
            f"Contact : {facts['site']} | {facts['tel']}",
            f"Catalogue : {jc['jeux_vr']} jeux VR et {jc['escape_vr']} scénarios d’escape VR",
            f"Plages tarifaires : {facts['horaires_prix']}",
            f"Jeux VR : {facts['vr']['prix_normal']}€ (standard) / {facts['vr']['prix_majoré']}€ (majoré) par joueur, jusqu’à {facts['vr']['max_joueurs']} joueurs. {facts['vr']['session']}",
            f"Escape VR : {facts['escape_vr']['prix_normal']}€ (standard) / {facts['escape_vr']['prix_majoré']}€ (majoré) par joueur, jusqu’à {facts['escape_vr']['max_joueurs']} joueurs.",
            f"Quiz : {facts['quiz']['prix_30']}€ (30min), {facts['quiz']['prix_60']}€ (60min), {facts['quiz']['prix_90']}€ (90min), jusqu’à {facts['quiz']['max_joueurs']} joueurs. {facts['quiz']['age']}",
            f"Salle enfant : {facts['salle_enfant']['prix_h']}€ /h (+{facts['salle_enfant']['prix_demi_h_sup']}€ / 30 min), {facts['salle_enfant']['details']} (goûter possible sur devis).",
            f"Salle d’attente : {facts['attente']['details']}",
            f"Hygiène/matériel : {facts['equipement']}",
            f"Fidélité : {facts['fidelite']['gains']} Récompenses : {facts['fidelite']['recompenses']} {facts['fidelite']['utilisation']} {facts['fidelite']['consultation']}",
            "Événements/horaires fériés : si demandé, répondre que vous n’avez pas l’info précise et rediriger vers Retroworld (site/téléphone).",
            "IMPORTANT : la formule 'goûter illimité' est périmée -> ne jamais la proposer. Goûter uniquement sur devis.",
        ]

    convo_id = str(metadata.get("conversation_id") or "").strip()
    meta_lines: List[str] = []
    if metadata.get("source"):
        meta_lines.append(f"Source : {metadata['source']}.")
    if metadata.get("page_url"):
        meta_lines.append(f"Page : {metadata['page_url']}.")
    if convo_id:
        meta_lines.append(f"Conversation ID : {convo_id}.")

    system_text = "\n".join([
        "\n".join(system_rules),
        "",
        "FACTS (utilisez uniquement ces informations) :",
        "\n".join(f"- {line}" for line in facts_block),
        "",
        "Si l’utilisateur demande Runningman depuis Retroworld (ou inversement), expliquez que c’est une activité distincte dans le même bâtiment et donnez le bon contact.",
    ])

    prompt_messages: List[Dict[str, str]] = [{"role": "system", "content": system_text}]
    if meta_lines:
        prompt_messages.append({"role": "system", "content": " ".join(meta_lines)})

    # Conversation history
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant", "system") and content is not None:
            prompt_messages.append({"role": role, "content": str(content)})

    return prompt_messages


# ---------------------------------------------------------
# GUARDRAILS FOR OPENAI OUTPUT
# ---------------------------------------------------------

_RETRO_BANNED = [
    "meta quest 2", "quest 2",
]
_RETRO_SAFE_FALLBACK = (
    "Pour ce point, je préfère éviter toute erreur. "
    "Pouvez-vous contacter l’équipe Retroworld au 04 94 47 94 64 ou via https://www.retroworldfrance.com ?"
)

_RUNNING_BANNED = ["tournoi", "promotion", "promo", "réduction", "reduction", "événement confirmé", "evenement confirme"]


def guard_openai_reply(brand: str, reply: str) -> Tuple[str, List[str]]:
    """If the model output contains banned/known-wrong claims, replace with a safe fallback."""
    brand = (brand or "").lower()
    r = reply or ""
    r_low = r.lower()
    hits: List[str] = []

    if brand == "retroworld":
        for b in _RETRO_BANNED:
            if b in r_low:
                hits.append(b)
        if hits:
            return _RETRO_SAFE_FALLBACK, hits

    if brand == "runningman":
        for b in _RUNNING_BANNED:
            if b in r_low:
                hits.append(b)
        # If it mentions availability as certain, block
        if re.search(r"\b(dispo|disponible|il reste|places disponibles|c['’]est bon)\b", r_low) and "ne peux pas confirmer" not in r_low:
            hits.append("availability_claim")
        if hits:
            return (
                "Je ne peux pas confirmer la disponibilité ou des offres en direct. "
                "Pour une réponse fiable, merci de réserver via https://runningmangames.fr "
                "ou d’appeler le 04 98 09 30 59.",
                hits,
            )

    return r, hits


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

def append_conversation_log(
    conversation_id: Optional[str],
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
            if role in ("user", "assistant", "system") and content is not None:
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
        return {"error": "unknown_brand"}, 404  # type: ignore[return-value]

    # find last user message
    last_user_text = ""
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_text = str(msg.get("content") or "")
            break

    effective_brand = detect_brand_from_text(last_user_text, default=brand_entry)

    conversation_id = str(metadata.get("conversation_id") or "").strip()
    if not conversation_id:
        conversation_id = f"{effective_brand}_{int(time.time() * 1000)}"
        metadata["conversation_id"] = conversation_id

    metadata["brand_entry"] = brand_entry
    metadata["brand_effective"] = effective_brand

    # Optionally rebuild history if client sends only the latest message
    messages_for_prompt: List[Dict[str, Any]] = list(messages or [])
    try:
        only_user_simple = (
            len(messages_for_prompt) == 1
            and isinstance(messages_for_prompt[0], dict)
            and messages_for_prompt[0].get("role") == "user"
        )
        no_assistant_msgs = all(
            (isinstance(m, dict) and m.get("role") != "assistant") for m in messages_for_prompt
        )
        use_server_history = allow_server_history and conversation_id and (only_user_simple or no_assistant_msgs)
        if metadata.get("no_server_history") is True:
            use_server_history = False
        if use_server_history:
            past = reconstruct_history_from_logs(conversation_id)
            if past:
                messages_for_prompt = past + messages_for_prompt
                logger.info("Reconstructed history for %s (%d past + %d new)", conversation_id, len(past), len(messages))
    except Exception as e:
        logger.error("History reconstruct error: %s", e)

    kb = load_kb(effective_brand)

    # FAST path first
    fast_reply = answer_fast(effective_brand, kb, last_user_text, metadata=metadata)
    if fast_reply:
        reply_text = fast_reply
        usage: Dict[str, Any] = {}
        guard_hits: List[str] = []
    else:
        prompt_messages = build_prompt(effective_brand, kb, messages_for_prompt, metadata)
        reply_text, usage = call_openai_chat(prompt_messages)
        reply_text, guard_hits = guard_openai_reply(effective_brand, reply_text)

    # log
    if do_log:
        try:
            channel = metadata.get("source") or "web"
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
                    "skipped_openai": bool(fast_reply),
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
        "skipped_openai": bool(fast_reply),
        "guard_hits": guard_hits,
    }


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


@app.route("/favicon.ico", methods=["GET"])
def favicon():  # type: ignore[override]
    return "", 204


@app.route("/health", methods=["GET"])
def health():  # type: ignore[override]
    return jsonify({"status": "ok", "time": time.time()}), 200


@app.route("/chat/<brand>", methods=["POST"])
def chat_route(brand: str):  # type: ignore[override]
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
        logger.warning("Chat fallback due to runtime error: %s", e)
        safe = "Je peux répondre aux questions courantes, mais je n’ai pas accès au moteur de réponse avancé pour le moment. Pouvez-vous reformuler votre demande ?"
        return jsonify({"reply": safe, "brand_used": brand, "brand_entry": brand, "error": str(e)}), 200
    except Exception as e:
        logger.error("chat_route error: %s", e)
        return jsonify({"error": "server_error", "details": str(e)}), 500


def _require_admin_token(req) -> bool:
    tok = (req.args.get("token") or "").strip()
    if not tok:
        tok = (req.headers.get("X-Admin-Token") or "").strip()
    return bool(tok) and tok == ADMIN_DASHBOARD_TOKEN


@app.route("/kb/upsert/<brand>", methods=["POST"])
def kb_upsert(brand: str):  # type: ignore[override]
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


@app.route("/webhooks/qweekle", methods=["POST"])
def qweekle_webhook():  # type: ignore[override]
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
def admin_api_conversations():  # type: ignore[override]
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
def admin_api_conversation_detail(conversation_id: str):  # type: ignore[override]
    if not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403
    records = load_conversation_records(conversation_id)
    if not records:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"conversation_id": conversation_id, "records": records}), 200


@app.route("/admin/api/test", methods=["POST"])
def admin_api_test():  # type: ignore[override]
    """Run a batch test: input can be JSON (with results/q) or plain text (one question per line)."""
    if not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403
    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400
    if not isinstance(body, dict):
        return jsonify({"error": "invalid_payload"}), 400

    mode = str(body.get("mode") or "auto")
    brand = str(body.get("brand") or "auto").lower()
    payload = body.get("payload")

    # If payload is a JSON string, try to parse it once.
    if isinstance(payload, str):
        s = payload.strip()
        if s and (s.startswith("{") or s.startswith("[")):
            try:
                payload = json.loads(s)
            except Exception:
                pass

    questions: List[str] = []

    # payload could be a dict (test file), list, or string
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
                if ln.lower().startswith("réponse") or ln.lower().startswith("reponse"):
                    continue
                if ln.lower().startswith("brand") or ln.lower().startswith("mode"):
                    continue
                questions.append(ln)

    # Deduplicate while keeping order
    seen = set()
    qs: List[str] = []
    for q in questions:
        qn = q.strip()
        if not qn:
            continue
        if qn in seen:
            continue
        seen.add(qn)
        qs.append(qn)

    # run
    results: List[Dict[str, Any]] = []
    convo_id = f"test_{int(time.time()*1000)}"

    for q in qs[:300]:
        messages = [{"role": "user", "content": q}]
        md = {"source": "admin_test", "conversation_id": convo_id, "no_server_history": True}
        use_brand = brand if brand in SUPPORTED_BRANDS else "retroworld"
        if brand == "auto":
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

    return jsonify({"count": len(results), "mode": mode, "results": results}), 200


# ---------------- ADMIN UI ----------------

@app.route("/admin/conversations", methods=["GET"])
def admin_conversations_page():  # type: ignore[override]
    if not _require_admin_token(request):
        return "Forbidden", 403

    return """
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <title>Admin IA – Retroworld / Runningman</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --bg: #0b1220;
      --card: #111b2e;
      --card2: #0f172a;
      --border: #22314f;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;

      --ok: #22c55e;
      --warn: #f59e0b;
      --bad: #ef4444;

      --brand-retro: #6366f1;
      --brand-run: #22c55e;
      --brand-mix: #f97316;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
      background: radial-gradient(1000px 700px at 20% -10%, rgba(56,189,248,0.18), transparent 55%),
                  radial-gradient(900px 600px at 110% 30%, rgba(99,102,241,0.18), transparent 50%),
                  var(--bg);
      color: var(--text);
    }
    .container { max-width: 1250px; margin: 0 auto; padding: 22px 18px 40px; }
    header {
      display:flex; align-items:flex-end; justify-content:space-between; gap:16px; flex-wrap:wrap;
      margin-bottom: 18px;
    }
    h1 { margin:0; font-size: 26px; font-weight: 650; letter-spacing: 0.2px; }
    .subtitle { margin-top:6px; font-size: 13px; color: var(--muted); }
    .tabs { display:flex; gap:8px; flex-wrap:wrap; }
    .tab {
      border-radius: 999px; border: 1px solid var(--border);
      background: rgba(17,27,46,0.7);
      color: var(--muted);
      padding: 8px 12px;
      font-size: 12px; cursor:pointer;
    }
    .tab.active { color: var(--bg); background: var(--accent); border-color: var(--accent); }
    .grid { display:grid; grid-template-columns: 1fr 420px; gap: 14px; }
    @media (max-width: 1000px) { .grid { grid-template-columns: 1fr; } }

    .panel {
      border: 1px solid var(--border);
      border-radius: 18px;
      background: rgba(17,27,46,0.86);
      backdrop-filter: blur(8px);
      overflow: hidden;
      box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    }
    .panel-head {
      display:flex; align-items:center; justify-content:space-between;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      background: rgba(15,23,42,0.55);
    }
    .panel-head h2 { margin:0; font-size: 13px; color: var(--muted); font-weight: 650; letter-spacing: 0.06em; text-transform: uppercase; }
    .toolbar { display:flex; gap:10px; flex-wrap:wrap; align-items:center; padding: 12px 14px; border-bottom:1px solid var(--border); }
    .search { flex:1; min-width: 220px; }
    input[type="text"], select, textarea {
      width:100%;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(15,23,42,0.6);
      color: var(--text);
      padding: 10px 12px;
      font-size: 13px;
      outline: none;
    }
    textarea { min-height: 190px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; line-height: 1.35; }
    .btn {
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(15,23,42,0.6);
      color: var(--muted);
      padding: 10px 12px;
      font-size: 12px;
      cursor:pointer;
      transition: 120ms ease;
      user-select:none;
    }
    .btn:hover { border-color: var(--accent); color: var(--accent); }
    .btn.primary { background: var(--accent); border-color: var(--accent); color: var(--bg); font-weight: 650; }
    .btn.primary:hover { filter: brightness(1.03); color: var(--bg); }
    .chips { display:flex; gap:8px; flex-wrap:wrap; }
    .chip {
      border-radius: 999px; border: 1px solid var(--border);
      padding: 7px 11px;
      font-size: 12px;
      cursor:pointer;
      background: rgba(15,23,42,0.55);
      color: var(--muted);
    }
    .chip.active { background: var(--accent); border-color: var(--accent); color: var(--bg); }
    .chip[data-brand="runningman"].active { background: var(--brand-run); border-color: var(--brand-run); }
    .chip[data-brand="retroworld"].active { background: var(--brand-retro); border-color: var(--brand-retro); }
    .chip[data-brand="mixed"].active { background: var(--brand-mix); border-color: var(--brand-mix); }

    table { width:100%; border-collapse: collapse; font-size: 13px; }
    thead { background: rgba(15,23,42,0.7); }
    th, td { padding: 10px 12px; border-bottom: 1px solid rgba(34,49,79,0.7); text-align: left; vertical-align: top; }
    th { font-size: 11px; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; }
    tr:hover td { background: rgba(15,23,42,0.35); }
    .badge { display:inline-flex; align-items:center; border-radius: 999px; padding: 3px 9px; font-size: 11px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; }
    .badge-run { background: rgba(34,197,94,0.18); color: #22c55e; }
    .badge-retro { background: rgba(99,102,241,0.18); color: #a5b4fc; }
    .badge-mix { background: rgba(249,115,22,0.18); color: #fdba74; }
    .badge-unknown { background: rgba(148,163,184,0.18); color: var(--muted); }
    .pill { display:inline-flex; align-items:center; border-radius: 999px; padding: 3px 9px; font-size: 11px; color: var(--muted); border: 1px solid rgba(148,163,184,0.35); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }

    .detail { padding: 14px; max-height: 520px; overflow-y: auto; }
    .bubble { max-width: 86%; margin: 10px 0; padding: 10px 12px; border-radius: 16px; white-space: pre-wrap; line-height: 1.35; }
    .bubble-user { margin-left:auto; background: rgba(51,65,85,0.65); border-bottom-right-radius: 6px; }
    .bubble-bot { margin-right:auto; background: rgba(15,23,42,0.75); border-bottom-left-radius: 6px; border: 1px solid rgba(34,49,79,0.65); }
    .meta { font-size: 11px; color: var(--muted); margin-top: 6px; }
    .muted { color: var(--muted); }

    .test-grid { display:grid; grid-template-columns: 1fr; gap: 12px; padding: 14px; }
    .test-actions { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    .results { border-top: 1px solid var(--border); padding-top: 12px; }
    .res-row { border: 1px solid rgba(34,49,79,0.7); border-radius: 16px; padding: 10px 12px; margin-bottom: 10px; background: rgba(15,23,42,0.45); }
    .res-row .top { display:flex; gap:10px; flex-wrap:wrap; align-items:center; justify-content:space-between; margin-bottom: 8px; }
    .flag { font-size: 11px; border-radius: 999px; padding: 3px 9px; border: 1px solid rgba(148,163,184,0.35); color: var(--muted); }
    .flag.ok { border-color: rgba(34,197,94,0.45); color: #86efac; }
    .flag.warn { border-color: rgba(245,158,11,0.45); color: #fbbf24; }
    .flag.bad { border-color: rgba(239,68,68,0.55); color: #fca5a5; }
    .small { font-size: 12px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Admin IA</h1>
        <div class="subtitle">Retroworld / Runningman • logs + console de test</div>
      </div>
      <div class="tabs">
        <button class="tab active" data-tab="convs">Conversations</button>
        <button class="tab" data-tab="test">Console de test</button>
      </div>
    </header>

    <!-- CONVERSATIONS TAB -->
    <section id="tab-convs">
      <div class="panel" style="margin-bottom: 14px;">
        <div class="panel-head">
          <h2>Filtres</h2>
          <button class="btn" id="btn-refresh">Rafraîchir</button>
        </div>
        <div class="toolbar">
          <div class="search">
            <input type="text" id="search" placeholder="Rechercher (question, source, conversation_id…)" />
          </div>
          <div class="chips">
            <button class="chip active" data-filter="all">Tout</button>
            <button class="chip" data-filter="runningman" data-brand="runningman">Runningman</button>
            <button class="chip" data-filter="retroworld" data-brand="retroworld">Retroworld</button>
            <button class="chip" data-filter="mixed" data-brand="mixed">Mix</button>
          </div>
        </div>
      </div>

      <div class="grid">
        <div class="panel">
          <div class="panel-head"><h2>Dernières conversations</h2></div>
          <div style="overflow:auto;">
            <table>
              <thead>
                <tr>
                  <th style="width: 170px;">Date</th>
                  <th style="width: 90px;">Canal</th>
                  <th style="width: 120px;">Marque</th>
                  <th style="width: 160px;">Source</th>
                  <th>Dernier message</th>
                </tr>
              </thead>
              <tbody id="rows">
                <tr><td colspan="5" class="muted">Chargement…</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head"><h2>Détail</h2></div>
          <div class="detail" id="convDetail">
            <div class="muted">Sélectionnez une conversation.</div>
          </div>
        </div>
      </div>
    </section>

    <!-- TEST TAB -->
    <section id="tab-test" style="display:none;">
      <div class="panel">
        <div class="panel-head"><h2>Console de test</h2></div>
        <div class="test-grid">
          <div class="small muted">
            Collez un JSON (format “results/q”) ou un texte (1 question par ligne). La console renvoie les réponses en mode FAST/OpenAI, la marque détectée et les blocages (guard).
          </div>
          <div class="test-actions">
            <div style="min-width:220px; flex:1;">
              <select id="testBrand">
                <option value="auto">Marque : auto</option>
                <option value="retroworld">Marque : Retroworld</option>
                <option value="runningman">Marque : Runningman</option>
              </select>
            </div>
            <button class="btn primary" id="btn-run">Lancer le test</button>
            <button class="btn" id="btn-sample">Exemple</button>
            <span class="muted small" id="testStatus"></span>
          </div>
          <textarea id="testInput" placeholder="Exemples :
- Adresse ?
- Je veux réserver une salle enfant samedi dans 2 semaines
OU collez votre JSON complet ici"></textarea>

          <div class="results" id="testResults"></div>
        </div>
      </div>
    </section>

  </div>

<script>
(function() {
  const params = new URLSearchParams(window.location.search);
  const token = params.get("token") || "";

  // Tabs
  const tabs = Array.from(document.querySelectorAll(".tab"));
  const tabConvs = document.getElementById("tab-convs");
  const tabTest = document.getElementById("tab-test");
  tabs.forEach(t => t.addEventListener("click", () => {
    tabs.forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    const which = t.getAttribute("data-tab");
    tabConvs.style.display = which === "convs" ? "" : "none";
    tabTest.style.display  = which === "test"  ? "" : "none";
  }));

  // Conversations
  const rowsEl = document.getElementById("rows");
  const searchInput = document.getElementById("search");
  const btnRefresh = document.getElementById("btn-refresh");
  const chips = Array.from(document.querySelectorAll(".chip"));
  const convDetail = document.getElementById("convDetail");
  let allData = [];
  let currentFilter = "all";
  let searchTerm = "";

  function formatDate(ts) {
    if (!ts) return "";
    try {
      return new Date(ts * 1000).toLocaleString("fr-FR", { hour12: false });
    } catch(e) { return ""; }
  }
  function brandBadge(b) {
    if (b === "runningman") return '<span class="badge badge-run">Runningman</span>';
    if (b === "retroworld") return '<span class="badge badge-retro">Retroworld</span>';
    if (b === "mixed") return '<span class="badge badge-mix">Mix</span>';
    return '<span class="badge badge-unknown">Inconnu</span>';
  }
  function channelPill(ch) {
    const txt = (ch || "web").toUpperCase();
    return '<span class="pill">' + txt + '</span>';
  }
  function sourcePill(s) {
    return '<span class="pill">' + (s || "n/a") + '</span>';
  }

  function renderConvs() {
    const term = searchTerm.trim().toLowerCase();
    let filtered = allData.slice();
    if (currentFilter !== "all") {
      filtered = filtered.filter(c => {
        if (currentFilter === "mixed") return c.brand_final === "mixed";
        return c.brand_final === currentFilter;
      });
    }
    if (term) {
      filtered = filtered.filter(c =>
        (c.preview && c.preview.toLowerCase().includes(term)) ||
        (c.source && c.source.toLowerCase().includes(term)) ||
        (c.conversation_id && c.conversation_id.toLowerCase().includes(term))
      );
    }
    if (!filtered.length) {
      rowsEl.innerHTML = '<tr><td colspan="5" class="muted">Aucune conversation trouvée.</td></tr>';
      return;
    }
    rowsEl.innerHTML = filtered.map(c => `
      <tr onclick="viewConversation('${c.conversation_id}')">
        <td>
          <div>${formatDate(c.timestamp)}</div>
          <div class="muted mono" style="font-size:11px;">${c.conversation_id}</div>
        </td>
        <td>${channelPill(c.channel)}</td>
        <td>${brandBadge(c.brand_final)}</td>
        <td>${sourcePill(c.source)}</td>
        <td><div style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:560px;">${c.preview || '<span class="muted">(vide)</span>'}</div></td>
      </tr>
    `).join("");
  }

  async function loadConvs() {
    rowsEl.innerHTML = '<tr><td colspan="5" class="muted">Chargement…</td></tr>';
    try {
      const res = await fetch(`/admin/api/conversations?token=${encodeURIComponent(token)}`);
      if (!res.ok) {
        rowsEl.innerHTML = `<tr><td colspan="5" class="muted">Erreur ${res.status}</td></tr>`;
        return;
      }
      allData = await res.json();
      renderConvs();
    } catch (e) {
      console.error(e);
      rowsEl.innerHTML = '<tr><td colspan="5" class="muted">Erreur réseau.</td></tr>';
    }
  }

  window.viewConversation = async function(id) {
    convDetail.innerHTML = `<div class="muted">Chargement ${id}…</div>`;
    try {
      const res = await fetch(`/admin/api/conversation/${encodeURIComponent(id)}?token=${encodeURIComponent(token)}`);
      if (!res.ok) {
        convDetail.innerHTML = `<div class="muted">Erreur ${res.status}</div>`;
        return;
      }
      const data = await res.json();
      const records = data.records || [];
      if (!records.length) {
        convDetail.innerHTML = `<div class="muted">Aucun enregistrement.</div>`;
        return;
      }
      let html = `<div class="mono muted" style="margin-bottom:10px;">${id}</div>`;
      records.forEach(rec => {
        const userMsgs = rec.user_messages || [];
        const reply = rec.assistant_reply || "";
        const extra = rec.extra || {};
        const guard = (extra.guard_hits || []).join(", ");
        const skipped = extra.skipped_openai ? "FAST" : "OPENAI";
        userMsgs.filter(m => m.role === "user").forEach(m => {
          html += `<div class="bubble bubble-user">${(m.content || "")}</div>`;
        });
        if (reply) {
          html += `<div class="bubble bubble-bot">${reply}</div>`;
          html += `<div class="meta">${skipped}${guard ? " • guard: " + guard : ""}</div>`;
        }
        if (rec.timestamp) {
          html += `<div class="meta">${new Date(rec.timestamp*1000).toLocaleString("fr-FR", {hour12:false})}</div>`;
        }
      });
      convDetail.innerHTML = html;
      convDetail.scrollTop = convDetail.scrollHeight;
    } catch (e) {
      console.error(e);
      convDetail.innerHTML = `<div class="muted">Erreur réseau.</div>`;
    }
  };

  searchInput.addEventListener("input", function() { searchTerm = this.value; renderConvs(); });
  btnRefresh.addEventListener("click", loadConvs);
  chips.forEach(chip => chip.addEventListener("click", () => {
    chips.forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
    currentFilter = chip.getAttribute("data-filter") || "all";
    renderConvs();
  }));

  loadConvs();

  // Test console
  const testInput = document.getElementById("testInput");
  const testBrand = document.getElementById("testBrand");
  const btnRun = document.getElementById("btn-run");
  const btnSample = document.getElementById("btn-sample");
  const testResults = document.getElementById("testResults");
  const testStatus = document.getElementById("testStatus");

  btnSample.addEventListener("click", () => {
    testInput.value = `{
  "results": [
    {"q": "Adresse ?"},
    {"q": "Je veux réserver une salle enfant samedi dans 2 semaines"},
    {"q": "Vous nettoyez les casques ?"},
    {"q": "Vous faites quelque chose pour Halloween ?"}
  ]
}`;
  });

  function flag(skipped, guardHits) {
    if (guardHits && guardHits.length) return `<span class="flag bad">GUARD</span>`;
    return skipped ? `<span class="flag ok">FAST</span>` : `<span class="flag warn">OPENAI</span>`;
  }

  function escapeHtml(s) {
    return (s || "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
  }

  btnRun.addEventListener("click", async () => {
    testResults.innerHTML = "";
    testStatus.textContent = "Test en cours…";
    const payloadText = testInput.value || "";
    let payload = payloadText;
    // Try JSON parse to send as object when possible
    try { payload = JSON.parse(payloadText); } catch(e) {}
    try {
      const res = await fetch(`/admin/api/test?token=${encodeURIComponent(token)}`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ brand: testBrand.value, mode: "auto", payload })
      });
      if (!res.ok) {
        testStatus.textContent = `Erreur ${res.status}`;
        return;
      }
      const data = await res.json();
      const results = data.results || [];
      testStatus.textContent = `${results.length} réponses`;
      if (!results.length) {
        testResults.innerHTML = `<div class="muted">Aucun résultat.</div>`;
        return;
      }
      testResults.innerHTML = results.map(r => {
        const b = r.brand_used || "auto";
        const skipped = !!r.skipped_openai;
        const guardHits = r.guard_hits || [];
        return `
          <div class="res-row">
            <div class="top">
              <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
                ${flag(skipped, guardHits)}
                <span class="pill">${b}</span>
                ${guardHits.length ? `<span class="flag bad">hits: ${escapeHtml(guardHits.join(", "))}</span>` : ""}
              </div>
              <div class="muted mono small">${escapeHtml((r.q || "").slice(0, 90))}${(r.q||"").length>90?"…":""}</div>
            </div>
            <div class="small"><b>Q:</b> ${escapeHtml(r.q || "")}</div>
            <div class="small" style="margin-top:6px;"><b>R:</b> ${escapeHtml(r.a || "")}</div>
          </div>
        `;
      }).join("");
    } catch(e) {
      console.error(e);
      testStatus.textContent = "Erreur réseau";
    }
  });

})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
