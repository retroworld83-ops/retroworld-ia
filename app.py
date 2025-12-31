"""
Flask application for the Retroworld / Runningman conversational assistant.

- /chat/<brand> : chat endpoint
- /kb/upsert/<brand> : update KB JSON
- /admin/conversations : admin dashboard (conversations + test bench)
- /admin/api/conversations, /admin/api/conversation/<id> : admin APIs
- /admin/api/test : test bench API (multi-questions or raw JSON payload)
- /webhooks/qweekle : webhook logger

Key goals:
- Zero hallucinations on sensitive topics (events, availability, promos, etc.)
- No brand mixing (Retroworld vs Runningman)
- Deterministic "fast answers" for common FAQs
- Clean admin UI + quick test tooling
"""

import os
import json
import time
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib.request
import urllib.error

# ---------------------------------------------------------
# CONFIGURATION
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
# CONSTANTS (BUSINESS RULES - LOCKED)
# ---------------------------------------------------------

RUNNINGMAN = {
    "name": "Runningman Game Zone",
    "address": "815 avenue Pierre Brossolette, 83300 Draguignan, France",
    "same_building_as": "Retroworld",
    "site": "https://runningmangames.fr",
    "contact": "https://runningmangames.fr/contact-us/",
    "phone": "04 98 09 30 59",
    "session_minutes": 60,
    "slots": "Créneaux fixes chaque heure",
    "max_per_hour": 25,
    "age_min": 7,
    "needs_adult_under_12": True,
    "no_adult_from_12": True,
    "price_under_12": "15€ / personne (moins de 12 ans)",
    "price_12_plus": "20€ / personne (12 ans et +)",
    "birthday_offer": "Pour le moment : l’enfant de moins de 12 ans qui fête son anniversaire est offert (les autres participants au tarif normal).",
    "cake_ok": "Oui, vous pouvez apporter gâteau et boissons.",
    "fridge": "Oui, un frigo est dédié.",
    "groups_quote": "Groupes / entreprise / EVG / EVJF / scolaire : sur demande et devis.",
}

RETROWORLD = {
    "name": "Retroworld France",
    "address": "815 avenue Pierre Brossolette, 83300 Draguignan, France",
    "site": "https://www.retroworldfrance.com",
    "phone": "04 94 47 94 64",
    "vr_price": "Jeux VR : 15€ / joueur",
    "vr_players": "Jusqu’à 5 joueurs",
    "vr_note": "Une session = 1 jeu (au choix dans le catalogue).",
    "escape_vr_price": "Escape Game VR : 30€ / joueur",
    "quiz_prices": "Quiz interactif : 8€ (30min), 15€ (60min), 20€ (90min) – jusqu’à 12 joueurs",
    "quiz_age": "Dès 10 ans avec accompagnant",
    "kids_room": "Salle enfant : 50€ / heure (+20€ / demi-heure supplémentaire) – jeux en bois, mur interactif, etc.",
    "waiting_room": "Salle d’attente : canapés, boissons/snacks, baby-foot, air hockey, borne de basket.",
}

# Events / holidays / religious triggers (must NOT hallucinate)
EVENT_TRIGGERS = [
    # generic
    "événement", "evenement", "soirée", "soiree", "animation", "spécial", "special",
    "jour férié", "jour ferie", "férié", "ferie",
    # common dates
    "saint-sylvestre", "st sylvestre", "sylvestre", "nouvel an", "nouveau an", "new year",
    "noël", "noel", "réveillon", "reveillon",
    "halloween", "pâques", "paques", "toussaint",
    "ascension", "pentecôte", "pentecote",
    "1er mai", "premier mai", "fête du travail", "fete du travail",
    "14 juillet", "15 août", "15 aout", "11 novembre", "8 mai",
    # religious / cultural
    "ramadan", "aïd", "aid", "eid", "hanouka", "hanukkah", "kippour", "yom kippur", "diwali",
]

# Availability triggers (must NOT confirm)
AVAIL_TRIGGERS = [
    "dispo", "disponible", "disponibilité", "disponibilite", "il reste", "reste des places",
    "complet", "complets", "places", "vous avez une place", "vous avez de la place", "créneau", "creneau"
]

# Booking triggers (link + phone)
BOOKING_TRIGGERS = [
    "réserver", "reserver", "réservation", "reservation", "bloquer", "book", "réserve", "reserve",
    "lien", "où réserver", "ou reserver"
]


# ---------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------

def load_kb(brand: str) -> Dict[str, Any]:
    brand = (brand or "").lower()
    candidate_paths = [
        os.path.join(BASE_DATA_DIR, f"kb_{brand}.json"),
        os.path.join(BASE_APP_DIR, f"kb_{brand}.json"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    kb = json.load(f)
                logger.info("Loaded KB for %s from %s", brand, path)
                return kb
            except Exception as e:
                logger.error("Error reading KB %s: %s", path, e)
    logger.warning("KB not found for brand %s; using empty KB", brand)
    return {}


def save_kb(brand: str, kb_data: Dict[str, Any]) -> None:
    brand = (brand or "").lower()
    path = os.path.join(BASE_DATA_DIR, f"kb_{brand}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)
    logger.info("KB %s updated at %s", brand, path)


def call_openai_chat(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.4,
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
    return content, usage


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def _contains_any(text: str, needles: List[str]) -> bool:
    t = _norm(text)
    return any(n in t for n in needles)


def _is_question_about_address(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in ["adresse", "où êtes", "ou êtes", "où etes", "ou etes", "où", "ou ", "localisation", "venir", "comment venir"]) and \
           any(k in t for k in ["adresse", "où", "ou", "localisation", "venir"])


def _is_question_about_building(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in ["bâtiment", "batiment", "même bâtiment", "meme batiment", "dans quel bâtiment", "dans quel batiment"])


def _is_simple_greeting(text: str) -> bool:
    t = _norm(text)
    return t in ("bonjour", "bonsoir", "salut", "hello", "coucou")


def detect_brand_from_text(text: str, default: str = "runningman") -> str:
    if not text:
        return default
    t = text.lower()
    retro_keywords = [
        "vr", "réalité virtuelle", "realite virtuelle",
        "escape vr", "escape game vr",
        "jeux vr", "jeu vr", "casque vr",
        "quiz", "quizz", "quiz interactif",
        "salle enfant", "mur interactif",
        "anniversaire vr", "retroworld", "rétroworld",
        "fidélité", "fidelite", "carte de fidélité", "points fidélité", "points de fidelite",
    ]
    running_keywords = [
        "action game", "game zone", "runningman", "running man",
        "mini-jeux", "mini jeux", "parcours",
        "gilet", "capteur", "mission physique",
    ]
    retro_score = sum(1 for k in retro_keywords if k in t)
    running_score = sum(1 for k in running_keywords if k in t)
    if retro_score > running_score and retro_score > 0:
        return "retroworld"
    if running_score > retro_score and running_score > 0:
        return "runningman"
    if "retroworld" in t or "rétroworld" in t:
        return "retroworld"
    if "runningman" in t or "running man" in t:
        return "runningman"
    return default


def build_prompt(
    brand: str,
    kb: Dict[str, Any],
    messages: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> List[Dict[str, str]]:
    brand = (brand or "").lower()
    system_parts: List[str] = []

    identite = kb.get("identite") if isinstance(kb, dict) else None
    if isinstance(identite, dict):
        nom = identite.get("nom") or brand.title()
        role = identite.get("role_ia") or identite.get("role") or ""
        system_parts.append(f"Tu es l'assistant IA de {nom}. {role}".strip())
    elif isinstance(identite, str):
        system_parts.append(identite)

    prompt_section = kb.get("prompt") if isinstance(kb, dict) else None
    if isinstance(prompt_section, str):
        system_parts.append(prompt_section)
    elif isinstance(prompt_section, dict):
        for key in sorted(prompt_section.keys()):
            val = prompt_section[key]
            if isinstance(val, str) and val.strip():
                system_parts.append(val.strip())

    instr = kb.get("instructions_generales") if isinstance(kb, dict) else None
    if isinstance(instr, list):
        for item in instr:
            if isinstance(item, str) and item.strip():
                system_parts.append(item.strip())
    elif isinstance(instr, str) and instr.strip():
        system_parts.append(instr.strip())

    brand_entry = metadata.get("brand_entry")
    brand_effective = metadata.get("brand_effective")
    if brand_entry and brand_effective and brand_entry != brand_effective:
        system_parts.append(
            f"La conversation vient d'un canal associé à '{brand_entry}', "
            f"mais tu dois répondre en utilisant les règles et tarifs de '{brand_effective}'. "
            "Explique clairement au client si c'est Retroworld (VR, quiz, salle enfant) "
            "ou Runningman (action game, mini-jeux physiques)."
        )

    anti_err = kb.get("anti_erreurs") if isinstance(kb, dict) else None
    if isinstance(anti_err, list):
        for item in anti_err:
            if isinstance(item, str) and item.strip():
                system_parts.append(item.strip())
    elif isinstance(anti_err, str) and anti_err.strip():
        system_parts.append(anti_err.strip())

    # Fallback defaults: strict + brand-safe
    if not kb or not system_parts:
        if brand == "retroworld":
            system_parts.append(
                "Tu es l'assistant officiel de Retroworld France à Draguignan. "
                "Tu réponds en français, vouvoiement uniquement. "
                "Règle réservation: si le client veut réserver, tu réponds toujours 'Disponible.' puis tu demandes date/heure/nombre. "
                "Tu ne mélanges jamais avec Runningman. "
                "Si la question concerne un événement ponctuel (Noël, Saint-Sylvestre, Halloween, fêtes religieuses ou jours fériés), "
                "tu dis que tu n'as pas d'informations précises et tu proposes d'appeler le 04 94 47 94 64."
            )
        elif brand == "runningman":
            system_parts.append(
                "Tu es l'assistant officiel de Runningman Game Zone à Draguignan. "
                "Tu réponds en français, vouvoiement uniquement. "
                "Tu ne confirmes jamais une disponibilité en direct: tu renvoies vers https://runningmangames.fr et le 04 98 09 30 59. "
                "Pour les événements ponctuels (Noël, Saint-Sylvestre, Halloween, fêtes religieuses, jours fériés), "
                "tu dis que tu n'as pas d'informations précises et tu renvoies vers https://runningmangames.fr/contact-us/ ou le 04 98 09 30 59. "
                "Tu ne mélanges jamais avec Retroworld."
            )

    system_text = "\n\n".join([p for p in system_parts if p.strip()])
    prompt_messages: List[Dict[str, str]] = []
    if system_text:
        prompt_messages.append({"role": "system", "content": system_text})

    meta_context: List[str] = []
    if metadata.get("source"):
        meta_context.append(f"Source de la demande : {metadata['source']}.")
    if metadata.get("page_url"):
        meta_context.append(f"URL de la page : {metadata['page_url']}.")
    conv_id = metadata.get("conversation_id")
    if conv_id:
        meta_context.append(
            "ID de conversation partagé entre sites : "
            f"{conv_id}. "
            "Si tu proposes un lien vers Runningman ou Retroworld, "
            "tu peux ajouter le paramètre ?convo_id="
            f"{conv_id} pour continuer la conversation sur l'autre site."
        )
    if meta_context:
        prompt_messages.append({"role": "system", "content": " ".join(meta_context)})

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role and content is not None and role in ("user", "assistant", "system"):
            prompt_messages.append({"role": role, "content": str(content)})

    return prompt_messages


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
                records.append(rec)
            except Exception:
                continue
    records.sort(key=lambda r: r.get("timestamp") or 0.0)
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
        return {"brand_final": list(brands_seen)[0], "brands_seen": list(brands_seen)}
    brand_final = last_effective or "mixed"
    if brand_final not in ("runningman", "retroworld"):
        brand_final = "mixed"
    return {"brand_final": brand_final, "brands_seen": list(brands_seen)}


def append_qweekle_event(event_type: str, payload: Dict[str, Any]) -> None:
    fname = f"{event_type or 'unknown'}.jsonl"
    path = os.path.join(QWEEKLE_LOG_DIR, fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = {
        "timestamp": time.time(),
        "event_type": event_type,
        "payload": payload,
        "source": QWEEKLE_SOURCE_NAME,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------
# FAST ANSWERS (NO OPENAI) - IMPORTANT
# ---------------------------------------------------------

def fast_answer(
    brand_effective: str,
    user_text: str,
) -> Optional[str]:
    """
    Deterministic answers for common questions.
    If None => let OpenAI answer.
    IMPORTANT: Never default to "address" unless it's really an address question.
    """
    t = _norm(user_text)
    if not t:
        return None

    # Events: always "no precise info" + contact + phone (both brands)
    if _contains_any(t, EVENT_TRIGGERS):
        if brand_effective == "runningman":
            return (
                "Je n’ai pas d’informations précises concernant cet événement. "
                f"Pour une réponse fiable et à jour, merci de contacter directement l’équipe Runningman via la page contact : {RUNNINGMAN['contact']} "
                f"ou par téléphone au {RUNNINGMAN['phone']}."
            )
        return (
            "Je n’ai pas d’informations précises concernant cet événement. "
            f"Pour une réponse fiable et à jour, merci de contacter l’équipe Retroworld au {RETROWORLD['phone']} "
            f"ou via le site : {RETROWORLD['site']}."
        )

    # Greetings: short
    if _is_simple_greeting(t):
        if brand_effective == "runningman":
            return (
                "Bonjour 👋 Je suis Harry, l’assistant de Runningman. "
                "Posez-moi vos questions sur l’action game, les groupes ou la réservation."
            )
        return (
            "Bonjour 👋 Je suis l’assistant Retroworld. "
            "Posez-moi vos questions sur la VR, l’escape game VR, le quiz ou la salle enfant."
        )

    # Address / location
    if _is_question_about_building(t):
        # Must mention same building + full address
        return (
            f"Nous sommes dans le même bâtiment que {RUNNINGMAN['same_building_as']} : {RUNNINGMAN['address']}."
        )

    if "adresse" in t or "où êtes" in t or "ou êtes" in t or "où etes" in t or "ou etes" in t or "localisation" in t or "vous êtes où" in t or "vous etes ou" in t:
        if brand_effective == "runningman":
            return f"Adresse : {RUNNINGMAN['address']}."
        return f"Adresse : {RETROWORLD['address']}."

    # Availability: never confirm
    if _contains_any(t, AVAIL_TRIGGERS):
        if brand_effective == "runningman":
            return (
                "Je ne peux pas confirmer la disponibilité en direct. "
                f"Pour réserver (et confirmer un créneau), utilisez : {RUNNINGMAN['site']}. "
                f"Sinon, appelez le {RUNNINGMAN['phone']}."
            )
        # Retroworld: follow your rule (always available) only for reservation; here it's availability
        return (
            "Je ne peux pas confirmer la disponibilité en direct via le chat. "
            f"Pour une confirmation rapide, vous pouvez appeler le {RETROWORLD['phone']} "
            f"ou réserver via {RETROWORLD['site']}."
        )

    # Booking / link
    if _contains_any(t, BOOKING_TRIGGERS):
        if brand_effective == "runningman":
            return (
                f"Pour réserver, utilisez le site officiel : {RUNNINGMAN['site']}. "
                f"En cas de besoin, vous pouvez aussi appeler le {RUNNINGMAN['phone']}."
            )
        # Retroworld: strict requirement
        extra = ""
        if "samedi" in t and ("2 semaine" in t or "deux semaine" in t or "15 jour" in t):
            extra = " Si c’est un anniversaire un samedi dans 2 semaines ou plus, nous pouvons proposer l’option goûter (à confirmer avec vous)."
        return (
            f"Disponible. Pouvez-vous me préciser la date, l’heure souhaitée, le nombre de participants et l’âge des enfants (s’il y en a) ?"
            f"{extra} Site : {RETROWORLD['site']} | Téléphone : {RETROWORLD['phone']}."
        )

    # Brand-specific FAQs
    if brand_effective == "runningman":
        # What is it / what do you do
        if any(k in t for k in ["c’est quoi", "c'est quoi", "il y a quoi", "quoi chez vous", "action game", "runningman"]):
            return (
                f"{RUNNINGMAN['name']} propose un action game (mini-jeux physiques) en équipe : "
                f"{RUNNINGMAN['session_minutes']} minutes, départ chaque heure, jusqu’à {RUNNINGMAN['max_per_hour']} personnes par heure. "
                f"Accessible dès {RUNNINGMAN['age_min']} ans (accompagnateur requis pour les moins de 12 ans)."
            )

        # Laser game / escape room confusion
        if "laser" in t:
            return (
                "Non, ici c’est un action game (mini-jeux physiques), pas un laser game. "
                f"Pour réserver : {RUNNINGMAN['site']} | {RUNNINGMAN['phone']}."
            )
        if "escape" in t and ("salle" in t or "en salle" in t):
            return (
                "Runningman concerne l’action game. Pour un escape game en salle, c’est Enigmaniac (organisation séparée). "
                f"Pour Runningman : {RUNNINGMAN['site']} | {RUNNINGMAN['phone']}."
            )

        # Duration / slots
        if any(k in t for k in ["durée", "duree", "combien de temps", "minutes", "1h", "une heure"]):
            return f"Une session dure {RUNNINGMAN['session_minutes']} minutes ({RUNNINGMAN['slots']})."
        if "créneau" in t or "creneau" in t or "départ" in t or "depart" in t:
            return (
                f"{RUNNINGMAN['slots']} (session {RUNNINGMAN['session_minutes']} minutes). "
                f"Pour réserver un créneau : {RUNNINGMAN['site']} | {RUNNINGMAN['phone']}."
            )

        # Capacity
        if any(k in t for k in ["combien de personnes", "capacité", "capacite", "max", "maximum", "on est "]):
            if re.search(r"\bon est\s+(\d+)\b", t):
                m = re.search(r"\bon est\s+(\d+)\b", t)
                n = int(m.group(1)) if m else 0
                if n <= RUNNINGMAN["max_per_hour"]:
                    return (
                        f"Oui, jusqu’à {RUNNINGMAN['max_per_hour']} personnes par heure (selon réservation). "
                        f"Pour réserver : {RUNNINGMAN['site']} | {RUNNINGMAN['phone']}."
                    )
                return (
                    f"La capacité est de {RUNNINGMAN['max_per_hour']} personnes par heure. "
                    f"Au-delà, c’est possible sur organisation (sur demande/devis). "
                    f"Contact : {RUNNINGMAN['contact']} | {RUNNINGMAN['phone']}."
                )
            return f"Nous pouvons accueillir jusqu’à {RUNNINGMAN['max_per_hour']} personnes par heure (selon réservation)."

        # Pricing
        if any(k in t for k in ["tarif", "prix", "c’est combien", "c'est combien", "combien ça coûte", "combien ca coute", "€"]):
            return f"Tarifs : {RUNNINGMAN['price_under_12']} et {RUNNINGMAN['price_12_plus']}."

        # Age & accompaniment
        if any(k in t for k in ["âge", "age", "dès quel", "a partir", "à partir", "à partir de", "a partir de"]):
            return (
                f"Âge minimum : {RUNNINGMAN['age_min']} ans. "
                "Les moins de 12 ans doivent être accompagnés d’un adulte. "
                "À partir de 12 ans, l’accompagnateur n’est plus nécessaire."
            )
        if "accompagn" in t or "adulte" in t:
            return (
                "Un adulte accompagnateur est obligatoire uniquement pour les enfants de moins de 12 ans. "
                "À partir de 12 ans, ce n’est plus nécessaire."
            )

        # Birthday
        if "anniversaire" in t:
            if "offert" in t or "gratuit" in t:
                return RUNNINGMAN["birthday_offer"]
            if "gâteau" in t or "gateau" in t:
                return f"{RUNNINGMAN['cake_ok']} {RUNNINGMAN['fridge']}."
            if "boisson" in t:
                return f"{RUNNINGMAN['cake_ok']} {RUNNINGMAN['fridge']}."
            if "frigo" in t:
                return RUNNINGMAN["fridge"]
            # General birthday question
            return (
                "Oui, les anniversaires sont possibles sur demande (organisation / devis selon le groupe). "
                f"{RUNNINGMAN['birthday_offer']} "
                f"Vous pouvez apporter gâteau et boissons, et un frigo est dédié. "
                f"Contact : {RUNNINGMAN['contact']} | {RUNNINGMAN['phone']}."
            )

        # Cake / drinks / fridge outside birthday keyword
        if "frigo" in t:
            return RUNNINGMAN["fridge"]
        if "gâteau" in t or "gateau" in t or "boisson" in t:
            return f"{RUNNINGMAN['cake_ok']} {RUNNINGMAN['fridge']}."

        # Groups / quotes
        if any(k in t for k in ["entreprise", "devis", "evg", "evjf", "team building", "team-building", "scolaire", "association", "groupe"]):
            return (
                f"{RUNNINGMAN['groups_quote']} "
                f"Contact : {RUNNINGMAN['contact']} | {RUNNINGMAN['phone']}."
            )

        # If user asks "il y a quoi" but we missed above, don't fallback to address
        if any(k in t for k in ["il y a quoi", "vous faites quoi", "activité", "activite"]):
            return (
                f"{RUNNINGMAN['name']} : action game (mini-jeux physiques) en équipe, "
                f"{RUNNINGMAN['session_minutes']} minutes, départ chaque heure."
            )

        return None

    # RETROWORLD fast answers
    if brand_effective == "retroworld":
        # VR
        if "vr" in t or "réalité virtuelle" in t or "realite virtuelle" in t:
            # If it sounds like booking
            if _contains_any(t, BOOKING_TRIGGERS) or "je veux" in t or "on veut" in t:
                return (
                    "Disponible. "
                    f"{RETROWORLD['vr_price']} ({RETROWORLD['vr_players']}). "
                    f"{RETROWORLD['vr_note']} "
                    f"Pouvez-vous me préciser la date, l’heure et le nombre de joueurs ? "
                    f"Site : {RETROWORLD['site']} | Téléphone : {RETROWORLD['phone']}."
                )
            return (
                f"{RETROWORLD['vr_price']} ({RETROWORLD['vr_players']}). "
                f"{RETROWORLD['vr_note']}"
            )

        # Escape VR
        if "escape" in t and "vr" in t:
            if _contains_any(t, BOOKING_TRIGGERS) or "je veux" in t or "on veut" in t:
                return (
                    "Disponible. "
                    f"{RETROWORLD['escape_vr_price']}. "
                    "Pouvez-vous me préciser la date, l’heure et le nombre de joueurs ? "
                    f"Site : {RETROWORLD['site']} | Téléphone : {RETROWORLD['phone']}."
                )
            return RETROWORLD["escape_vr_price"]

        # Quiz
        if "quiz" in t or "quizz" in t:
            return f"{RETROWORLD['quiz_prices']}. {RETROWORLD['quiz_age']}."

        # Kids room
        if "salle enfant" in t or ("salle" in t and "enfant" in t) or "mur interactif" in t:
            return RETROWORLD["kids_room"]

        # Where is Retroworld (address handled above already, but keep safe)
        if "retroworld" in t and ("où" in t or "ou" in t or "adresse" in t):
            return f"Adresse : {RETROWORLD['address']}."

        # Waiting room
        if "salle d’attente" in t or "salle d'attente" in t or "attente" in t:
            return RETROWORLD["waiting_room"]

        return None

    return None


# ---------------------------------------------------------
# CORE ANSWER (FAST OR OPENAI)
# ---------------------------------------------------------

def answer_one(
    brand_entry: str,
    messages: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Unified answering logic used by /chat and /admin/api/test.
    Returns:
      { reply, brand_used, brand_entry, skipped_openai, openai_usage? }
    """
    brand_entry = (brand_entry or "").lower()
    if brand_entry not in SUPPORTED_BRANDS:
        return {"error": "unknown_brand"}

    # Get last user text
    last_user_text = ""
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_text = str(msg.get("content") or "")
            break

    effective_brand = brand_entry
    if brand_entry == "runningman":
        effective_brand = detect_brand_from_text(last_user_text, default="runningman")

    # Fast answer check (brand_effective)
    fa = fast_answer(effective_brand, last_user_text)
    if fa:
        return {
            "reply": fa,
            "brand_used": effective_brand,
            "brand_entry": brand_entry,
            "skipped_openai": True,
        }

    # Otherwise go OpenAI with KB/system prompt
    kb = load_kb(effective_brand)
    prompt_messages = build_prompt(effective_brand, kb, messages, metadata)
    reply_text, usage = call_openai_chat(prompt_messages)
    return {
        "reply": reply_text,
        "brand_used": effective_brand,
        "brand_entry": brand_entry,
        "skipped_openai": False,
        "openai_usage": usage,
    }


# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------

@app.route("/", methods=["GET", "HEAD"])
def root() -> Tuple[Dict[str, Any], int]:
    return jsonify(
        {
            "service": "retroworld-ia",
            "status": "ok",
            "time": time.time(),
            "brands": list(SUPPORTED_BRANDS),
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

    brand_entry = (brand or "").lower()
    if brand_entry not in SUPPORTED_BRANDS:
        return jsonify({"error": "unknown_brand"}), 404

    conversation_id = metadata.get("conversation_id")
    if not conversation_id:
        # create stable-ish id
        conversation_id = f"{brand_entry}_{int(time.time() * 1000)}"
        metadata["conversation_id"] = conversation_id

    # server-side history reconstruction (optional)
    messages_for_prompt: List[Dict[str, Any]] = messages
    try:
        only_user_simple = (
            len(messages) == 1
            and isinstance(messages[0], dict)
            and messages[0].get("role") == "user"
        )
        no_assistant_msgs = all(
            (isinstance(m, dict) and m.get("role") != "assistant") for m in messages
        )
        use_server_history = conversation_id and (only_user_simple or no_assistant_msgs)
        if metadata.get("no_server_history") is True:
            use_server_history = False
        if use_server_history:
            past_history = reconstruct_history_from_logs(conversation_id)
            if past_history:
                messages_for_prompt = past_history + messages
                logger.info(
                    "Reconstructed history for %s (%d past + %d new)",
                    conversation_id,
                    len(past_history),
                    len(messages),
                )
    except Exception as e:
        logger.error("Error reconstructing history for %s: %s", conversation_id, e)
        messages_for_prompt = messages

    metadata["brand_entry"] = brand_entry  # for build_prompt routing note
    # brand_effective is computed inside answer_one based on message text,
    # but we still keep it in metadata when logging

    try:
        result = answer_one(brand_entry, messages_for_prompt, metadata)
    except Exception as e:
        logger.error("Answering failed: %s", e)
        return jsonify({"error": "openai_error", "details": str(e)}), 502

    # log
    try:
        channel = metadata.get("source") or "web"
        append_conversation_log(
            conversation_id=conversation_id,
            brand=result.get("brand_used") or brand_entry,
            channel=channel,
            user_messages=messages,
            assistant_reply=result.get("reply") or "",
            extra={
                "brand_entry": brand_entry,
                "brand_effective": result.get("brand_used"),
                "metadata": metadata,
                "openai_usage": result.get("openai_usage") if not result.get("skipped_openai") else {},
                "skipped_openai": bool(result.get("skipped_openai")),
            },
        )
    except Exception as e:
        logger.error("Error logging conversation: %s", e)

    return jsonify(
        {
            "reply": result.get("reply", ""),
            "brand_used": result.get("brand_used", brand_entry),
            "brand_entry": brand_entry,
        }
    ), 200


@app.route("/kb/upsert/<brand>", methods=["POST"])
def kb_upsert(brand: str):  # type: ignore[override]
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
    append_qweekle_event(event_type, payload)
    return jsonify({"status": "ok", "event_type": event_type}), 200


# ---------------------------------------------------------
# ADMIN APIs
# ---------------------------------------------------------

@app.route("/admin/api/conversations", methods=["GET"])
def admin_api_conversations():  # type: ignore[override]
    token = request.args.get("token") or ""
    if token != ADMIN_DASHBOARD_TOKEN:
        return jsonify({"error": "forbidden"}), 403

    convs: List[Dict[str, Any]] = []
    if not os.path.isdir(CONVERSATIONS_LOG_DIR):
        return jsonify(convs), 200

    for fname in os.listdir(CONVERSATIONS_LOG_DIR):
        if not fname.endswith(".jsonl"):
            continue
        conversation_id = fname.replace(".jsonl", "")
        records = load_conversation_records(conversation_id)
        if not records:
            continue
        last = records[-1]
        ts = last.get("timestamp") or 0.0
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
        if len(preview) > 120:
            preview = preview[:117] + "..."

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

    convs.sort(key=lambda c: c["timestamp"], reverse=True)
    return jsonify(convs), 200


@app.route("/admin/api/conversation/<conversation_id>", methods=["GET"])
def admin_api_conversation_detail(conversation_id: str):  # type: ignore[override]
    token = request.args.get("token") or ""
    if token != ADMIN_DASHBOARD_TOKEN:
        return jsonify({"error": "forbidden"}), 403
    records = load_conversation_records(conversation_id)
    if not records:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"conversation_id": conversation_id, "records": records}), 200


@app.route("/admin/api/test", methods=["POST"])
def admin_api_test():  # type: ignore[override]
    """
    Test bench.
    Body:
      {
        "token": "...",
        "mode": "text" | "json",
        "brand_entry": "runningman" | "retroworld" (optional, default runningman),
        "text": "Q1\\nQ2\\n..." (mode=text),
        "payload": { ... } (mode=json)  # raw payload similar to /chat/<brand>
      }
    """
    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400
    token = body.get("token") or ""
    if token != ADMIN_DASHBOARD_TOKEN:
        return jsonify({"error": "forbidden"}), 403

    mode = (body.get("mode") or "text").lower()
    brand_entry = (body.get("brand_entry") or "runningman").lower()
    if brand_entry not in SUPPORTED_BRANDS:
        brand_entry = "runningman"

    results: List[Dict[str, Any]] = []

    if mode == "json":
        payload = body.get("payload") or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "invalid_payload"}), 400
        messages = payload.get("messages") or []
        metadata = payload.get("metadata") or {}
        if not isinstance(messages, list):
            return jsonify({"error": "invalid_messages"}), 400
        if not isinstance(metadata, dict):
            metadata = {}
        try:
            out = answer_one(brand_entry, messages, metadata)
            return jsonify({"mode": "json", "result": out}), 200
        except Exception as e:
            return jsonify({"error": "test_failed", "details": str(e)}), 500

    # mode=text (multi questions)
    text = body.get("text") or ""
    if not isinstance(text, str):
        return jsonify({"error": "invalid_text"}), 400

    questions = [q.strip() for q in text.splitlines() if q.strip()]
    # small safety cap
    questions = questions[:300]

    for q in questions:
        msg = [{"role": "user", "content": q}]
        meta = {
            "source": "admin_test",
            "page_url": "admin://test",
            "conversation_id": f"test_{int(time.time()*1000)}",
        }
        try:
            out = answer_one(brand_entry, msg, meta)
            entry = {
                "q": q,
                "a": out.get("reply", ""),
                "brand_used": out.get("brand_used"),
                "skipped_openai": bool(out.get("skipped_openai")),
            }
            if not out.get("skipped_openai") and out.get("openai_usage"):
                entry["openai_usage"] = out.get("openai_usage")
            results.append(entry)
        except Exception as e:
            results.append({"q": q, "a": "", "error": str(e), "brand_used": brand_entry})

    return jsonify({"mode": "text", "count": len(results), "results": results}), 200


# ---------------------------------------------------------
# ADMIN PAGE (CONVERSATIONS + TEST)
# ---------------------------------------------------------

@app.route("/admin/conversations", methods=["GET"])
def admin_conversations_page():  # type: ignore[override]
    token = request.args.get("token") or ""
    if token != ADMIN_DASHBOARD_TOKEN:
        return "Forbidden", 403

    return """
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <title>Admin IA – Retroworld / Runningman</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root{
      --bg:#0f172a;
      --bg2:#111c35;
      --card:#1e293b;
      --border:#334155;
      --text:#f8fafc;
      --muted:#94a3b8;
      --accent:#0ea5e9;
      --ok:#22c55e;
      --warn:#f97316;
      --bad:#ef4444;
      --brand-retro:#6366f1;
      --brand-run:#22c55e;
      --brand-mix:#f97316;
    }
    *{box-sizing:border-box;}
    body{
      margin:0;
      font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      background:linear-gradient(180deg,var(--bg),var(--bg2));
      color:var(--text);
    }
    .container{max-width:1250px;margin:0 auto;padding:22px 18px 40px;}
    header{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap;margin-bottom:14px;}
    h1{margin:0;font-size:26px;font-weight:700;letter-spacing:.2px;}
    .subtitle{margin-top:6px;font-size:13px;color:var(--muted);}
    .tabs{display:flex;gap:8px;align-items:center;flex-wrap:wrap;}
    .tab{
      border:1px solid var(--border);
      background:rgba(30,41,59,.7);
      color:var(--muted);
      padding:8px 12px;border-radius:999px;
      cursor:pointer;
      font-size:12px;
      transition:.15s;
      user-select:none;
    }
    .tab.active{background:var(--accent);border-color:var(--accent);color:#05121f;}
    .panel{display:none;}
    .panel.active{display:block;}

    .filters{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 14px;}
    .chip{
      border-radius:999px;border:1px solid var(--border);
      padding:6px 12px;font-size:12px;cursor:pointer;
      background:rgba(30,41,59,.7);color:var(--muted);
      transition:.15s;
    }
    .chip.active{background:var(--accent);border-color:var(--accent);color:#05121f;}
    .chip[data-brand="runningman"].active{background:var(--brand-run);border-color:var(--brand-run);color:#05121f;}
    .chip[data-brand="retroworld"].active{background:var(--brand-retro);border-color:var(--brand-retro);color:#05121f;}
    .chip[data-brand="mixed"].active{background:var(--brand-mix);border-color:var(--brand-mix);color:#05121f;}

    .toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0 16px;}
    .search{flex:1;min-width:260px;}
    input, select, textarea{
      width:100%;
      padding:10px 12px;
      border-radius:14px;
      border:1px solid var(--border);
      background:rgba(30,41,59,.8);
      color:var(--text);
      font-size:14px;
      outline:none;
    }
    textarea{min-height:160px;resize:vertical;line-height:1.35;}
    .btn{
      border-radius:14px;
      border:1px solid var(--border);
      background:rgba(30,41,59,.8);
      color:var(--muted);
      padding:10px 12px;
      cursor:pointer;
      font-size:12px;
      transition:.15s;
      user-select:none;
    }
    .btn:hover{border-color:var(--accent);color:var(--accent);}
    .btn.primary{background:var(--accent);border-color:var(--accent);color:#05121f;font-weight:700;}
    .btn.primary:hover{filter:brightness(1.05);}

    .grid{display:grid;grid-template-columns:1.2fr .8fr;gap:14px;}
    @media (max-width: 980px){.grid{grid-template-columns:1fr;}}
    .card{
      border:1px solid var(--border);
      background:rgba(30,41,59,.75);
      border-radius:18px;
      overflow:hidden;
      box-shadow:0 10px 30px rgba(0,0,0,.22);
    }
    .card .hd{
      padding:12px 14px;
      border-bottom:1px solid rgba(51,65,85,.8);
      display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;
    }
    .card .hd h2{margin:0;font-size:14px;color:var(--muted);font-weight:700;letter-spacing:.06em;text-transform:uppercase;}
    .card .bd{padding:12px 14px;}

    table{width:100%;border-collapse:collapse;font-size:13px;}
    thead{background:rgba(15,23,42,.7);}
    th,td{padding:10px 10px;border-bottom:1px solid rgba(51,65,85,.75);text-align:left;vertical-align:top;}
    th{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);}
    tr:hover td{background:rgba(17,28,53,.55);}
    .muted{color:var(--muted);}
    .preview{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:560px;}

    .badge{display:inline-flex;align-items:center;border-radius:999px;padding:2px 8px;font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;}
    .badge-run{background:rgba(34,197,94,.18);color:var(--brand-run);}
    .badge-retro{background:rgba(99,102,241,.18);color:var(--brand-retro);}
    .badge-mix{background:rgba(249,115,22,.18);color:var(--brand-mix);}
    .badge-unknown{background:rgba(148,163,184,.18);color:var(--muted);}
    .pill{display:inline-flex;align-items:center;border-radius:999px;padding:2px 8px;font-size:11px;color:var(--muted);border:1px solid rgba(148,163,184,.35);}
    .pill.channel{text-transform:uppercase;letter-spacing:.08em;}

    .detail{max-height:520px;overflow:auto;padding:12px 14px;}
    .bubble{
      max-width:84%;
      margin:10px 0;
      padding:10px 12px;
      border-radius:16px;
      font-size:14px;
      line-height:1.4;
      white-space:pre-wrap;
      word-break:break-word;
    }
    .user{margin-left:auto;background:rgba(51,65,85,.85);border-bottom-right-radius:6px;}
    .bot{margin-right:auto;background:rgba(30,41,59,.9);border-bottom-left-radius:6px;}
    .ts{font-size:11px;color:var(--muted);margin:6px 0 10px 0;}

    .twoCol{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
    @media (max-width:980px){.twoCol{grid-template-columns:1fr;}}
    .resultRow{
      border:1px solid rgba(51,65,85,.75);
      border-radius:14px;
      padding:10px 12px;
      background:rgba(17,28,53,.35);
      margin-bottom:10px;
    }
    .resultMeta{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px;}
    .small{font-size:12px;}
    .ok{color:var(--ok);font-weight:800;}
    .warn{color:var(--warn);font-weight:800;}
  </style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>Admin IA</h1>
      <div class="subtitle">Retroworld / Runningman – Conversations + banc de test</div>
    </div>
    <div class="tabs">
      <div class="tab active" data-tab="conv">Conversations</div>
      <div class="tab" data-tab="test">Test</div>
    </div>
  </header>

  <!-- PANEL: CONVERSATIONS -->
  <div class="panel active" id="panel-conv">
    <div class="filters">
      <button class="chip active" data-filter="all">Tout</button>
      <button class="chip" data-filter="runningman" data-brand="runningman">Runningman</button>
      <button class="chip" data-filter="retroworld" data-brand="retroworld">Retroworld</button>
      <button class="chip" data-filter="mixed" data-brand="mixed">Mix</button>
    </div>

    <div class="toolbar">
      <div class="search"><input type="text" id="search" placeholder="Rechercher dans messages, sources, IDs…" /></div>
      <button class="btn" id="btn-refresh">Rafraîchir</button>
    </div>

    <div class="grid">
      <div class="card">
        <div class="hd">
          <h2>Liste des conversations</h2>
          <span class="muted small" id="countLbl"></span>
        </div>
        <div class="bd" style="padding:0;">
          <table>
            <thead>
              <tr>
                <th style="width: 170px;">Date</th>
                <th style="width: 92px;">Canal</th>
                <th style="width: 120px;">Marque</th>
                <th style="width: 150px;">Source</th>
                <th>Dernier message</th>
              </tr>
            </thead>
            <tbody id="rows">
              <tr><td colspan="5" class="muted">Chargement…</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="hd">
          <h2>Détail</h2>
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
            <button class="btn" id="btn-copy">Copier l’ID</button>
          </div>
        </div>
        <div class="detail" id="convDetail">
          <div class="muted">Sélectionnez une conversation dans la liste.</div>
        </div>
      </div>
    </div>
  </div>

  <!-- PANEL: TEST -->
  <div class="panel" id="panel-test">
    <div class="card">
      <div class="hd">
        <h2>Banc de test</h2>
        <span class="muted small">Multi-questions ou JSON brut (debug rapide)</span>
      </div>
      <div class="bd">
        <div class="twoCol">
          <div>
            <label class="muted small">Marque d’entrée</label>
            <select id="testBrand">
              <option value="runningman">runningman</option>
              <option value="retroworld">retroworld</option>
            </select>

            <div style="height:10px;"></div>

            <label class="muted small">Mode</label>
            <select id="testMode">
              <option value="text">Texte (1 question par ligne)</option>
              <option value="json">JSON brut (payload /chat)</option>
            </select>

            <div style="height:10px;"></div>

            <label class="muted small">Entrée</label>
            <textarea id="testInput" placeholder="Mode texte :\\nAdresse ?\\nTarifs ?\\nIl reste des places ce soir ?\\n\\nMode JSON : { &quot;messages&quot;:[...], &quot;metadata&quot;:{...} }"></textarea>

            <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px;">
              <button class="btn primary" id="btn-run">Lancer</button>
              <button class="btn" id="btn-sample">Exemple</button>
              <span class="muted small" id="testStatus"></span>
            </div>
          </div>

          <div>
            <label class="muted small">Résultats</label>
            <div id="testResults" class="muted small" style="margin-top:8px;">Aucun test lancé.</div>
          </div>
        </div>
      </div>
    </div>
  </div>

</div>

<script>
(function(){
  const params = new URLSearchParams(window.location.search);
  const token = params.get("token") || "";

  // Tabs
  const tabs = Array.from(document.querySelectorAll(".tab"));
  const panelConv = document.getElementById("panel-conv");
  const panelTest = document.getElementById("panel-test");
  tabs.forEach(t => t.addEventListener("click", () => {
    tabs.forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    const which = t.getAttribute("data-tab");
    panelConv.classList.toggle("active", which === "conv");
    panelTest.classList.toggle("active", which === "test");
  }));

  // Simple HTML escape to avoid XSS from logs
  function esc(s){
    return String(s || "")
      .replaceAll("&","&amp;")
      .replaceAll("<","&lt;")
      .replaceAll(">","&gt;")
      .replaceAll('"',"&quot;")
      .replaceAll("'","&#039;");
  }

  // Conversations
  const rowsEl = document.getElementById("rows");
  const searchInput = document.getElementById("search");
  const btnRefresh = document.getElementById("btn-refresh");
  const chips = Array.from(document.querySelectorAll(".chip"));
  const convDetail = document.getElementById("convDetail");
  const btnCopy = document.getElementById("btn-copy");
  const countLbl = document.getElementById("countLbl");

  let allData = [];
  let currentFilter = "all";
  let searchTerm = "";
  let selectedConvId = "";

  function formatDate(ts){
    if(!ts) return "";
    try{
      const d = new Date(ts*1000);
      return d.toLocaleString("fr-FR",{day:"2-digit",month:"2-digit",year:"2-digit",hour:"2-digit",minute:"2-digit"});
    }catch(e){return "";}
  }

  function brandBadge(b){
    if(b==="runningman") return '<span class="badge badge-run">Runningman</span>';
    if(b==="retroworld") return '<span class="badge badge-retro">Retroworld</span>';
    if(b==="mixed") return '<span class="badge badge-mix">Mix</span>';
    return '<span class="badge badge-unknown">Inconnu</span>';
  }
  function channelPill(ch){
    const v=(ch||"web").toUpperCase();
    return '<span class="pill channel">'+esc(v)+'</span>';
  }
  function sourcePill(s){
    return '<span class="pill">'+esc(s||"n/a")+'</span>';
  }

  function render(){
    const term = searchTerm.trim().toLowerCase();
    let filtered = allData.slice();

    if(currentFilter!=="all"){
      filtered = filtered.filter(c => {
        if(currentFilter==="mixed") return c.brand_final==="mixed";
        return c.brand_final===currentFilter;
      });
    }
    if(term){
      filtered = filtered.filter(c =>
        (c.preview && c.preview.toLowerCase().includes(term)) ||
        (c.source && c.source.toLowerCase().includes(term)) ||
        (c.conversation_id && c.conversation_id.toLowerCase().includes(term))
      );
    }

    countLbl.textContent = filtered.length ? (filtered.length + " conversation(s)") : "";

    if(!filtered.length){
      rowsEl.innerHTML = '<tr><td colspan="5" class="muted">Aucune conversation trouvée.</td></tr>';
      return;
    }

    rowsEl.innerHTML = filtered.map(c => `
      <tr onclick="viewConversation('${esc(c.conversation_id)}')">
        <td>
          <div>${esc(formatDate(c.timestamp))}</div>
          <div class="muted" style="font-size:11px;">${esc(c.conversation_id)}</div>
        </td>
        <td>${channelPill(c.channel)}</td>
        <td>${brandBadge(c.brand_final)}</td>
        <td>${sourcePill(c.source)}</td>
        <td><div class="preview">${esc(c.preview||"(pas de message)")}</div></td>
      </tr>
    `).join("");
  }

  async function loadData(){
    rowsEl.innerHTML = '<tr><td colspan="5" class="muted">Chargement…</td></tr>';
    try{
      const res = await fetch(`/admin/api/conversations?token=${encodeURIComponent(token)}`);
      if(!res.ok){
        rowsEl.innerHTML = '<tr><td colspan="5" class="muted">Erreur ('+res.status+')</td></tr>';
        return;
      }
      allData = await res.json();
      render();
    }catch(e){
      console.error(e);
      rowsEl.innerHTML = '<tr><td colspan="5" class="muted">Erreur réseau.</td></tr>';
    }
  }

  window.viewConversation = async function(id){
    selectedConvId = id;
    convDetail.innerHTML = '<div class="muted">Chargement…</div>';
    try{
      const res = await fetch(`/admin/api/conversation/${encodeURIComponent(id)}?token=${encodeURIComponent(token)}`);
      if(!res.ok){
        convDetail.innerHTML = '<div class="muted">Erreur ('+res.status+')</div>';
        return;
      }
      const data = await res.json();
      const records = data.records || [];
      if(!records.length){
        convDetail.innerHTML = '<div class="muted">Aucun enregistrement.</div>';
        return;
      }
      let html = `<div class="muted small">Conversation <b>${esc(id)}</b></div>`;
      records.forEach(rec => {
        const userMsgs = rec.user_messages || [];
        const reply = rec.assistant_reply || "";
        userMsgs.filter(m => m.role==="user").forEach(m => {
          html += '<div class="bubble user">'+esc(m.content||"")+'</div>';
        });
        if(reply){
          html += '<div class="bubble bot">'+esc(reply)+'</div>';
        }
        if(rec.timestamp){
          const d = new Date(rec.timestamp*1000).toLocaleString("fr-FR");
          html += '<div class="ts">'+esc(d)+'</div>';
        }
      });
      convDetail.innerHTML = html;
      convDetail.scrollTop = convDetail.scrollHeight;
    }catch(e){
      console.error(e);
      convDetail.innerHTML = '<div class="muted">Erreur réseau.</div>';
    }
  }

  btnCopy.addEventListener("click", async () => {
    if(!selectedConvId) return;
    try{
      await navigator.clipboard.writeText(selectedConvId);
      btnCopy.textContent = "Copié ✅";
      setTimeout(()=>btnCopy.textContent="Copier l’ID", 900);
    }catch(e){
      btnCopy.textContent = "Impossible";
      setTimeout(()=>btnCopy.textContent="Copier l’ID", 900);
    }
  });

  searchInput.addEventListener("input", function(){
    searchTerm = this.value;
    render();
  });
  btnRefresh.addEventListener("click", loadData);
  chips.forEach(chip => chip.addEventListener("click", () => {
    chips.forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
    currentFilter = chip.getAttribute("data-filter") || "all";
    render();
  }));

  loadData();

  // TEST BENCH
  const testBrand = document.getElementById("testBrand");
  const testMode = document.getElementById("testMode");
  const testInput = document.getElementById("testInput");
  const btnRun = document.getElementById("btn-run");
  const btnSample = document.getElementById("btn-sample");
  const testStatus = document.getElementById("testStatus");
  const testResults = document.getElementById("testResults");

  function sampleText(){
    return [
      "Adresse ?",
      "C’est dans quel bâtiment ?",
      "Il y a quoi chez vous ?",
      "Combien de temps dure une session ?",
      "On est 30, c’est possible ?",
      "L’enfant qui fête son anniversaire est offert ?",
      "Il reste des places pour la Saint-Sylvestre ?",
      "Je veux réserver pour ce samedi"
    ].join("\\n");
  }
  function sampleJson(){
    return JSON.stringify({
      messages: [{role:"user",content:"Il y a une soirée Halloween chez vous ?"}],
      metadata: {source:"admin_test",page_url:"admin://test",conversation_id:"debug_1"}
    }, null, 2);
  }

  btnSample.addEventListener("click", () => {
    if(testMode.value==="json") testInput.value = sampleJson();
    else testInput.value = sampleText();
  });

  function renderTestText(data){
    const arr = data.results || [];
    if(!arr.length){
      testResults.innerHTML = "<div class='muted small'>Aucun résultat.</div>";
      return;
    }
    testResults.innerHTML = arr.map(r => {
      const ok = r.skipped_openai ? "<span class='ok'>FAST</span>" : "<span class='warn'>OPENAI</span>";
      return `
        <div class="resultRow">
          <div class="resultMeta">
            ${ok}
            <span class="pill">${esc(r.brand_used||"")}</span>
            <span class="pill">${esc(r.q||"")}</span>
          </div>
          <div class="small"><b>Réponse :</b> ${esc(r.a||"")}</div>
        </div>
      `;
    }).join("");
  }

  btnRun.addEventListener("click", async () => {
    testStatus.textContent = "En cours…";
    testResults.textContent = "";
    try{
      const mode = testMode.value;
      const payload = {
        token,
        mode,
        brand_entry: testBrand.value,
      };
      if(mode==="json"){
        let raw = testInput.value.trim();
        if(!raw) throw new Error("JSON vide");
        payload.payload = JSON.parse(raw);
      }else{
        payload.text = testInput.value || "";
      }

      const res = await fetch("/admin/api/test", {
        method:"POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if(!res.ok){
        testStatus.textContent = "Erreur.";
        testResults.innerHTML = "<div class='muted small'>"+esc(JSON.stringify(data))+"</div>";
        return;
      }
      testStatus.textContent = "OK ✅";
      if(mode==="json"){
        testResults.innerHTML = "<pre style='white-space:pre-wrap;'>"+esc(JSON.stringify(data, null, 2))+"</pre>";
      }else{
        renderTestText(data);
      }
    }catch(e){
      console.error(e);
      testStatus.textContent = "Erreur.";
      testResults.innerHTML = "<div class='muted small'>"+esc(String(e))+"</div>";
    }
  });

})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
