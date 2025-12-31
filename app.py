import os
import json
import time
import re
import html
import logging
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib.request
import urllib.error

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

# ---------------------------------------------------------
# APP CONFIG
# ---------------------------------------------------------

app = Flask(__name__)
CORS(app)  # Vous avez dit allowlist inutile -> on laisse ouvert.

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

TZ_PARIS = "Europe/Paris"

# ---------------------------------------------------------
# HARD BUSINESS RULES (SERVER-SIDE) - ANTI VRILLE
# ---------------------------------------------------------

# Runningman: event cut-off keywords
RUNNINGMAN_EVENT_KEYWORDS = {
    "halloween",
    "saint-sylvestre",
    "saint sylvestre",
    "réveillon",
    "reveillon",
    "nouvel an",
    "nouvel-an",
    "noël",
    "noel",
    "soirée spéciale",
    "soiree speciale",
    "événement",
    "evenement",
    "événement spécial",
    "evenement special",
    "animation spéciale",
    "animation speciale",
    "jour férié",
    "jour ferie",
    "férié",
    "ferie",
    "fête",
    "fete",
    "ramadan",
    "aïd",
    "aid",
    "eid",
    "pâques",
    "paques",
    "ascension",
    "pentecôte",
    "pentecote",
    "toussaint",
    "hanouka",
    "hanukkah",
    "kippour",
    "yom kippur",
    "diwali",
}

RUNNINGMAN_EVENT_FIXED_REPLY = (
    "Je n’ai pas d’informations précises concernant cet événement. "
    "Pour une réponse fiable et à jour, merci de contacter directement l’équipe Runningman "
    "via la page contact : https://runningmangames.fr/contact-us/ "
    "ou par téléphone au 04 98 09 30 59."
)

RUNNINGMAN_BOOKING_REPLY = (
    "Pour réserver, utilisez le site officiel : https://runningmangames.fr. "
    "En cas de besoin, vous pouvez aussi appeler le 04 98 09 30 59."
)

RUNNINGMAN_AVAILABILITY_REPLY = (
    "Je ne peux pas confirmer la disponibilité en direct. "
    "Pour réserver (et confirmer un créneau), utilisez : https://runningmangames.fr. "
    "Sinon, appelez le 04 98 09 30 59."
)

RUNNINGMAN_CONTACT_REPLY = (
    "Pour une réponse fiable et à jour, contactez directement l’équipe Runningman : "
    "https://runningmangames.fr/contact-us/ ou 04 98 09 30 59."
)

RUNNINGMAN_INFO_PRICE_REPLY = (
    "Tarifs Runningman Game Zone : "
    "15€ par personne (moins de 12 ans) et 20€ par personne (12 ans et + / adulte)."
)

RUNNINGMAN_INFO_DURATION_REPLY = "Une session dure 60 minutes (créneaux fixes chaque heure)."

RUNNINGMAN_INFO_CAPA_REPLY = "Nous pouvons accueillir jusqu’à 25 personnes par heure (selon réservation)."

# Retroworld booking rule: always “disponible”
RETROWORLD_BOOKING_AVAILABLE_PREFIX = "Disponible."

RETROWORLD_PHONE = "04 94 47 94 64"
RETROWORLD_SITE = "https://www.retroworldfrance.com"

# Retroworld known activities (safe, factuels selon mémoire)
RETROWORLD_ACTIVITIES_FACTS = {
    "vr": "Jeux VR : 15€ / joueur, jusqu’à 5 joueurs (hors escape game VR).",
    "escape_vr": "Escape Game VR : 30€ / joueur (jeu de complétion si fin anticipée).",
    "quiz": "Quiz interactif : 8€ (30min), 15€ (60min), 20€ (90min), jusqu’à 12 joueurs, dès 10 ans avec accompagnant.",
    "salle_enfant": "Salle enfant : 50€ / heure, +20€ la demi-heure supplémentaire (jeux en bois, mur interactif, etc.).",
}

# Keywords
KW_BOOKING = {"réserver", "reservation", "réservation", "lien", "réserve", "bloquer", "creneau", "créneau"}
KW_AVAIL = {"dispo", "disponible", "disponibilité", "places", "place", "complet", "complète", "reste des places"}
KW_PRICE = {"tarif", "tarifs", "prix", "combien", "coute", "coûte", "€", "euro"}
KW_DURATION = {"durée", "duree", "combien de temps", "minutes", "1h", "heure"}
KW_CAPA = {"combien de personnes", "maximum", "max", "capacité", "capacite", "jusqu'à", "jusqu’a"}
KW_ADDRESS = {"adresse", "où", "ou", "situé", "situe", "localisation"}
KW_RETRO = {"retroworld", "rétroworld", "vr", "quiz", "escape vr", "escape game vr", "salle enfant", "mur interactif"}
KW_RUN = {"runningman", "running man", "action game", "game zone", "mini-jeux", "mini jeux"}

URL_REGEX = re.compile(r"(https?://[^\s\]\)\"\'<>]+)", re.IGNORECASE)


# ---------------------------------------------------------
# TIME HELPERS
# ---------------------------------------------------------

def now_paris() -> datetime:
    if ZoneInfo is None:
        return datetime.now()
    try:
        return datetime.now(ZoneInfo(TZ_PARIS))
    except Exception:
        return datetime.now()


def parse_date_from_text_fr(text: str) -> Optional[date]:
    """
    Tries to parse a date from common formats:
    - YYYY-MM-DD
    - DD/MM/YYYY
    - DD/MM (assumes current year)
    """
    if not text:
        return None
    t = text.strip()

    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d)
        except Exception:
            return None

    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", t)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d)
        except Exception:
            return None

    m = re.search(r"\b(\d{1,2})/(\d{1,2})\b", t)
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
        y = now_paris().date().year
        try:
            return date(y, mo, d)
        except Exception:
            return None

    return None


def should_offer_gouter_for_retroworld(user_text: str) -> bool:
    """
    Rule: propose goûter if it's Saturday + 2 weeks.
    Best-effort detection based on:
    - presence of "samedi"
    - and explicit date >= today + 14 days OR phrase like "dans 2 semaines"
    """
    if not user_text:
        return False
    t = user_text.lower()

    if "samedi" not in t:
        return False

    today = now_paris().date()
    d = parse_date_from_text_fr(t)
    if d:
        return d >= (today + timedelta(days=14))

    # Heuristic: "dans 2 semaines", "dans deux semaines", "dans 15 jours"
    if re.search(r"\bdans\s+(2|deux)\s+semaines\b", t):
        return True
    if re.search(r"\bdans\s+1[4-9]\s+jours\b", t):
        return True
    if re.search(r"\bdans\s+2[0-9]\s+jours\b", t):
        return True

    return False


# ---------------------------------------------------------
# KB LOAD/SAVE
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


# ---------------------------------------------------------
# BRAND DETECTION
# ---------------------------------------------------------

def detect_brand_from_text(text: str, default: str = "runningman") -> str:
    if not text:
        return default
    t = text.lower()

    retro_score = sum(1 for k in KW_RETRO if k in t)
    run_score = sum(1 for k in KW_RUN if k in t)

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
# PROMPT BUILDER (robust: supports old/new KB structures)
# ---------------------------------------------------------

def _kb_collect_text_blocks(kb: Dict[str, Any]) -> List[str]:
    """
    Collects instruction blocks from multiple possible KB schemas.
    Supports:
      - identite (dict/str)
      - prompt (str/dict)
      - instructions_generales (str/list)
      - anti_erreurs (str/list)
      - regles_fondamentales_ia (dict)
      - evenements_exceptionnels (dict)
      - reservation / tarification / etc. (dict) -> summarized lightly
    """
    parts: List[str] = []

    if not isinstance(kb, dict):
        return parts

    identite = kb.get("identite")
    if isinstance(identite, dict):
        nom = identite.get("nom") or ""
        role_ia = identite.get("role_ia") or identite.get("role") or ""
        if nom:
            parts.append(f"Vous êtes l'assistant officiel de {nom}.")
        if isinstance(role_ia, str) and role_ia.strip():
            parts.append(role_ia.strip())
    elif isinstance(identite, str) and identite.strip():
        parts.append(identite.strip())

    prompt_section = kb.get("prompt")
    if isinstance(prompt_section, str) and prompt_section.strip():
        parts.append(prompt_section.strip())
    elif isinstance(prompt_section, dict):
        for key in sorted(prompt_section.keys()):
            val = prompt_section[key]
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())

    instr = kb.get("instructions_generales")
    if isinstance(instr, list):
        for item in instr:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
    elif isinstance(instr, str) and instr.strip():
        parts.append(instr.strip())

    anti = kb.get("anti_erreurs")
    if isinstance(anti, list):
        for item in anti:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
    elif isinstance(anti, str) and anti.strip():
        parts.append(anti.strip())

    # Old schema support: regles_fondamentales_ia
    rfi = kb.get("regles_fondamentales_ia")
    if isinstance(rfi, dict):
        if rfi.get("vouvoiement_obligatoire") is True:
            parts.append("Vouvoiement obligatoire.")
        ton = rfi.get("ton")
        if isinstance(ton, list) and ton:
            parts.append("Ton: " + ", ".join([str(x) for x in ton if str(x).strip()]) + ".")
        interdits = rfi.get("interdictions_absolues")
        if isinstance(interdits, list) and interdits:
            parts.append("Interdictions absolues: " + "; ".join([str(x) for x in interdits if str(x).strip()]) + ".")
        rr = rfi.get("regle_reponse")
        if isinstance(rr, dict):
            p = rr.get("principe")
            if isinstance(p, str) and p.strip():
                parts.append(p.strip())

    # Old schema support: evenements_exceptionnels
    ev = kb.get("evenements_exceptionnels")
    if isinstance(ev, dict):
        ra = ev.get("regle_absolue") or ev.get("regle_de_coupure")
        if isinstance(ra, str) and ra.strip():
            parts.append("Règle événements: " + ra.strip())
        rep = ev.get("reponse_unique_non_generative")
        if isinstance(rep, dict):
            t = rep.get("texte")
            if isinstance(t, str) and t.strip():
                parts.append("Réponse événement (exacte): " + t.strip())

    return parts


def build_prompt(
    brand: str,
    kb: Dict[str, Any],
    messages: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> List[Dict[str, str]]:
    brand = (brand or "").lower()
    system_parts: List[str] = []

    system_parts.extend(_kb_collect_text_blocks(kb))

    # Always enforce non-mixing + safety in system prompt
    system_parts.append(
        "Règle anti-mélange: Répondez uniquement pour la marque effective. "
        "Si l'utilisateur mélange Runningman (action game) et Retroworld (VR/quiz/salle enfant), "
        "clarifiez en 1 phrase puis répondez séparément (max 2 puces)."
    )

    # Brand-specific safety defaults (server rules already handle Runningman events/availability,
    # but we keep reminders for the LLM anyway).
    if brand == "runningman":
        system_parts.append(
            "Runningman: Ne jamais inventer un événement, une promotion ou une disponibilité. "
            "Ne jamais dire 'il reste des places'. "
            "Si question événement: rediriger vers contact/téléphone. "
            "Si demande de réservation: donner https://runningmangames.fr + 04 98 09 30 59."
        )
    elif brand == "retroworld":
        system_parts.append(
            "Retroworld: Respecter strictement les règles anniversaires et capacités. "
            "Ne pas inventer (ex: crêpes par défaut, 15 personnes en même temps, etc.). "
            "Site officiel toujours en https://www.retroworldfrance.com."
        )

    # Metadata context (no tracking hints)
    meta_context: List[str] = []
    if metadata.get("source"):
        meta_context.append(f"Source: {metadata['source']}.")
    if metadata.get("page_url"):
        meta_context.append(f"Page: {metadata['page_url']}.")
    if metadata.get("conversation_id"):
        meta_context.append(f"Conversation (interne): {metadata['conversation_id']}.")

    prompt_messages: List[Dict[str, str]] = []
    system_text = "\n\n".join([p for p in system_parts if p.strip()])
    if system_text:
        prompt_messages.append({"role": "system", "content": system_text})
    if meta_context:
        prompt_messages.append({"role": "system", "content": " ".join(meta_context)})

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant", "system") and content is not None:
            prompt_messages.append({"role": role, "content": str(content)})

    return prompt_messages


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
                records.append(json.loads(line))
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
# SANITIZATION (anti liens foireux / tracking)
# ---------------------------------------------------------

def _strip_tracking(url: str) -> str:
    # On supprime totalement la querystring (évite convo_id/utm/fbclid etc.)
    return url.split("?", 1)[0].strip()


def sanitize_reply_text(brand: str, reply: str) -> str:
    if not reply:
        return reply

    brand = (brand or "").lower()

    def _replace_url(m: re.Match) -> str:
        u = m.group(1)
        base = _strip_tracking(u)

        # Runningman: éviter endpoints inventés
        if brand == "runningman":
            if "runningmangames.fr" in base and "/reservation" in base:
                return "https://runningmangames.fr"
        return base

    cleaned = URL_REGEX.sub(_replace_url, reply)
    cleaned = re.sub(r"\bconvo_id\s*=\s*[A-Za-z0-9\-_]+\b", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


# ---------------------------------------------------------
# REQUEST NORMALIZER (accepts legacy payloads)
# ---------------------------------------------------------

def normalize_incoming_messages(body: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Accepts:
      - {messages:[{role,content},...], metadata:{...}}
      - {message:"...", history:[...], metadata:{...}}
      - {message:"..."} (legacy)
    Returns (messages, metadata)
    """
    metadata = body.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    messages = body.get("messages")
    if isinstance(messages, list):
        # Ensure list of dicts
        out = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant", "system"):
                out.append({"role": m.get("role"), "content": m.get("content", "")})
        return out, metadata

    # Legacy
    msg = body.get("message")
    history = body.get("history")

    out: List[Dict[str, Any]] = []
    if isinstance(history, list):
        for h in history:
            if isinstance(h, dict) and h.get("role") in ("user", "assistant", "system"):
                out.append({"role": h.get("role"), "content": h.get("content", "")})
            elif isinstance(h, str) and h.strip():
                out.append({"role": "user", "content": h.strip()})

    if isinstance(msg, str) and msg.strip():
        out.append({"role": "user", "content": msg.strip()})

    return out, metadata


# ---------------------------------------------------------
# SERVER-SIDE ROUTER (templates to guarantee no vrille)
# ---------------------------------------------------------

def is_runningman_event_question(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in RUNNINGMAN_EVENT_KEYWORDS)


def route_pre_llm(brand_effective: str, last_user_text: str) -> Optional[str]:
    """
    Returns a fixed reply if we must not call LLM (safety/business rules),
    otherwise None.
    """
    b = (brand_effective or "").lower()
    t = (last_user_text or "").lower()

    # Runningman: EVENT cut-off (absolute)
    if b == "runningman" and is_runningman_event_question(t):
        return RUNNINGMAN_EVENT_FIXED_REPLY

    # Runningman: availability questions -> never confirm
    if b == "runningman" and any(k in t for k in KW_AVAIL):
        return RUNNINGMAN_AVAILABILITY_REPLY

    # Runningman: booking questions -> link + phone (only)
    if b == "runningman" and any(k in t for k in KW_BOOKING):
        return RUNNINGMAN_BOOKING_REPLY

    # Runningman: quick factual templates
    if b == "runningman":
        # address
        if any(k in t for k in KW_ADDRESS):
            return "Adresse : 815 avenue Pierre Brossolette, 83300 Draguignan, France."
        # duration
        if any(k in t for k in KW_DURATION):
            return RUNNINGMAN_INFO_DURATION_REPLY
        # capacity
        if any(k in t for k in KW_CAPA):
            return RUNNINGMAN_INFO_CAPA_REPLY
        # price
        if any(k in t for k in KW_PRICE):
            return RUNNINGMAN_INFO_PRICE_REPLY

    # Retroworld: booking rule "toujours disponible"
    if b == "retroworld" and any(k in t for k in KW_BOOKING.union(KW_AVAIL)):
        # Must be clear in first message + propose goûter if saturday+2 weeks
        parts = [RETROWORLD_BOOKING_AVAILABLE_PREFIX]
        # Mandatory info request kept minimal but useful
        parts.append("Pouvez-vous me préciser la date, l’heure souhaitée, le nombre de participants et l’âge des enfants (s’il y en a) ?")
        if should_offer_gouter_for_retroworld(t):
            parts.append("Si c’est un anniversaire un samedi dans 2 semaines ou plus, nous pouvons proposer l’option goûter (à confirmer avec vous).")
        parts.append(f"Site : {RETROWORLD_SITE} | Téléphone : {RETROWORLD_PHONE}.")
        return " ".join(parts)

    # Retroworld: factual templates for common questions (avoid hallucinations)
    if b == "retroworld":
        if any(k in t for k in KW_ADDRESS):
            return "Adresse : 815 avenue Pierre Brossolette, 83300 Draguignan, France."
        if "vr" in t and any(k in t for k in KW_PRICE.union(KW_DURATION)):
            return RETROWORLD_ACTIVITIES_FACTS["vr"]
        if "escape" in t and "vr" in t and any(k in t for k in KW_PRICE.union(KW_DURATION)):
            return RETROWORLD_ACTIVITIES_FACTS["escape_vr"]
        if "quiz" in t or "quizz" in t:
            if any(k in t for k in KW_PRICE.union(KW_DURATION).union(KW_CAPA)):
                return RETROWORLD_ACTIVITIES_FACTS["quiz"]
        if "salle enfant" in t or "salle d'enfant" in t:
            if any(k in t for k in KW_PRICE.union(KW_DURATION)):
                return RETROWORLD_ACTIVITIES_FACTS["salle_enfant"]

    return None


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
        }
    ), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": time.time()}), 200


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    return "", 204


def _admin_token_ok(req) -> bool:
    # Accept query ?token= OR header X-Admin-Token
    token_q = req.args.get("token") or ""
    token_h = req.headers.get("X-Admin-Token") or ""
    return (token_q and token_q == ADMIN_DASHBOARD_TOKEN) or (token_h and token_h == ADMIN_DASHBOARD_TOKEN)


@app.route("/chat", methods=["POST"])
def chat_legacy():
    """
    Legacy endpoint:
      - If body has brand -> routes to /chat/<brand>
      - Else defaults to runningman and auto-detect effective brand from user text
    """
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid_json"}), 400
    if not isinstance(body, dict):
        return jsonify({"error": "invalid_payload"}), 400

    brand = (body.get("brand") or body.get("metadata", {}).get("brand") or "runningman").lower()
    if brand not in SUPPORTED_BRANDS:
        brand = "runningman"
    return chat_route(brand)


@app.route("/chat/<brand>", methods=["POST"])
def chat_route(brand: str):
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid_json"}), 400
    if not isinstance(body, dict):
        return jsonify({"error": "invalid_payload"}), 400

    brand_entry = (brand or "").lower()
    if brand_entry not in SUPPORTED_BRANDS:
        return jsonify({"error": "unknown_brand"}), 404

    messages, metadata = normalize_incoming_messages(body)

    # Find last user text
    last_user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_text = str(msg.get("content") or "")
            break

    # Determine effective brand
    effective_brand = brand_entry
    if brand_entry == "runningman":
        # runningman entry can route to retroworld if VR/quiz etc.
        effective_brand = detect_brand_from_text(last_user_text, default="runningman")

    # Conversation id
    conversation_id = metadata.get("conversation_id")
    if not conversation_id:
        conversation_id = f"{effective_brand}_{int(time.time() * 1000)}"
        metadata["conversation_id"] = conversation_id

    metadata["brand_entry"] = brand_entry
    metadata["brand_effective"] = effective_brand

    # If only last message is sent, reconstruct
    messages_for_prompt: List[Dict[str, Any]] = messages
    try:
        only_user_simple = (
            len(messages) == 1
            and isinstance(messages[0], dict)
            and messages[0].get("role") == "user"
        )
        no_assistant_msgs = all(isinstance(m, dict) and m.get("role") != "assistant" for m in messages)
        use_server_history = bool(conversation_id) and (only_user_simple or no_assistant_msgs)
        if metadata.get("no_server_history") is True:
            use_server_history = False
        if use_server_history:
            past = reconstruct_history_from_logs(conversation_id)
            if past:
                messages_for_prompt = past + messages
    except Exception as e:
        logger.error("History reconstruction error: %s", e)
        messages_for_prompt = messages

    # -----------------------------------------------------
    # SERVER-SIDE SAFETY/TEMPLATES (guarantee no vrille)
    # -----------------------------------------------------
    fixed = route_pre_llm(effective_brand, last_user_text)
    if fixed is not None:
        reply_text = fixed
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
                    "openai_usage": {"skipped": True, "reason": "server_template_or_cutoff"},
                },
            )
        except Exception as e:
            logger.error("Logging error: %s", e)

        return jsonify({"reply": reply_text, "brand_used": effective_brand, "brand_entry": brand_entry}), 200

    # Normal LLM path
    kb = load_kb(effective_brand)

    try:
        prompt_messages = build_prompt(effective_brand, kb, messages_for_prompt, metadata)
    except Exception as e:
        logger.error("build_prompt failed: %s", e)
        return jsonify({"error": "prompt_build_failed"}), 500

    try:
        reply_text, usage = call_openai_chat(prompt_messages)
    except Exception as e:
        logger.error("OpenAI error: %s", e)
        return jsonify({"error": "openai_error", "details": str(e)}), 502

    reply_text = sanitize_reply_text(effective_brand, reply_text)

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
            },
        )
    except Exception as e:
        logger.error("Error logging conversation: %s", e)

    return jsonify({"reply": reply_text, "brand_used": effective_brand, "brand_entry": brand_entry}), 200


@app.route("/kb/upsert/<brand>", methods=["POST"])
def kb_upsert(brand: str):
    # Security: protect upsert
    if not _admin_token_ok(request):
        return jsonify({"error": "forbidden"}), 403

    brand = (brand or "").lower()
    if brand not in SUPPORTED_BRANDS:
        return jsonify({"error": "unknown_brand"}), 404

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
    append_qweekle_event(event_type, payload)
    return jsonify({"status": "ok", "event_type": event_type}), 200


# ---------------------------------------------------------
# ADMIN API (conversations + test box)
# ---------------------------------------------------------

@app.route("/admin/api/conversations", methods=["GET"])
def admin_api_conversations():
    if not _admin_token_ok(request):
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

    convs.sort(key=lambda c: c["timestamp"], reverse=True)
    return jsonify(convs), 200


@app.route("/admin/api/conversation/<conversation_id>", methods=["GET"])
def admin_api_conversation_detail(conversation_id: str):
    if not _admin_token_ok(request):
        return jsonify({"error": "forbidden"}), 403

    records = load_conversation_records(conversation_id)
    if not records:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"conversation_id": conversation_id, "records": records}), 200


@app.route("/admin/api/test_chat", methods=["POST"])
def admin_api_test_chat():
    """
    Debug endpoint for the Admin 'Test' panel.
    Supports:
      - mode: "text" -> split lines into questions, returns array of Q/A
      - mode: "json" -> forwards payload to /chat/<brand> logic and returns single reply
    """
    if not _admin_token_ok(request):
        return jsonify({"error": "forbidden"}), 403

    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    if not isinstance(body, dict):
        return jsonify({"error": "invalid_payload"}), 400

    mode = (body.get("mode") or "text").lower()
    brand = (body.get("brand") or "runningman").lower()
    if brand not in SUPPORTED_BRANDS:
        brand = "runningman"

    if mode == "json":
        payload = body.get("payload")
        if not isinstance(payload, dict):
            return jsonify({"error": "payload_must_be_object"}), 400

        # Call internal chat logic by faking a request object is hard;
        # we re-run the same pipeline here directly.
        messages, metadata = normalize_incoming_messages(payload if isinstance(payload, dict) else {})
        last_user_text = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user_text = str(msg.get("content") or "")
                break

        brand_entry = brand
        effective_brand = brand_entry
        if brand_entry == "runningman":
            effective_brand = detect_brand_from_text(last_user_text, default="runningman")

        fixed = route_pre_llm(effective_brand, last_user_text)
        if fixed is not None:
            return jsonify({"mode": "json", "brand_used": effective_brand, "reply": fixed, "skipped_openai": True}), 200

        kb = load_kb(effective_brand)
        metadata = metadata if isinstance(metadata, dict) else {}
        metadata.setdefault("source", "admin_test")
        metadata.setdefault("conversation_id", f"admin_test_{int(time.time())}")
        metadata["brand_entry"] = brand_entry
        metadata["brand_effective"] = effective_brand

        prompt_messages = build_prompt(effective_brand, kb, messages, metadata)
        reply_text, usage = call_openai_chat(prompt_messages)
        reply_text = sanitize_reply_text(effective_brand, reply_text)
        return jsonify({"mode": "json", "brand_used": effective_brand, "reply": reply_text, "openai_usage": usage}), 200

    # mode text (multi questions)
    text = body.get("text") or ""
    if not isinstance(text, str):
        return jsonify({"error": "text_must_be_string"}), 400

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return jsonify({"error": "no_questions"}), 400

    results: List[Dict[str, Any]] = []
    for q in lines:
        brand_entry = brand
        effective_brand = brand_entry
        if brand_entry == "runningman":
            effective_brand = detect_brand_from_text(q, default="runningman")

        fixed = route_pre_llm(effective_brand, q)
        if fixed is not None:
            results.append({"q": q, "brand_used": effective_brand, "a": fixed, "skipped_openai": True})
            continue

        kb = load_kb(effective_brand)
        metadata = {"source": "admin_test", "conversation_id": f"admin_test_{int(time.time())}", "brand_entry": brand_entry, "brand_effective": effective_brand}
        prompt = build_prompt(effective_brand, kb, [{"role": "user", "content": q}], metadata)
        try:
            a, usage = call_openai_chat(prompt)
            a = sanitize_reply_text(effective_brand, a)
            results.append({"q": q, "brand_used": effective_brand, "a": a, "openai_usage": usage})
        except Exception as e:
            results.append({"q": q, "brand_used": effective_brand, "a": f"[ERREUR OPENAI] {e}", "error": True})

    return jsonify({"mode": "text", "count": len(results), "results": results}), 200


# ---------------------------------------------------------
# ADMIN PAGE (improved + test panel)
# ---------------------------------------------------------

@app.route("/admin/conversations", methods=["GET"])
def admin_conversations_page():
    if not _admin_token_ok(request):
        return "Forbidden", 403

    token = request.args.get("token") or ""

    return f"""
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <title>Admin IA – Retroworld / Runningman</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {{
      --bg: #0b1220;
      --panel: #111b2e;
      --panel2: #0f172a;
      --border: rgba(148,163,184,0.20);
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --good: #22c55e;
      --warn: #f59e0b;
      --bad: #ef4444;
      --retro: #6366f1;
      --run: #22c55e;
      --mix: #f97316;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: radial-gradient(1200px 600px at 20% 0%, rgba(56,189,248,0.08), transparent),
                  radial-gradient(1000px 600px at 90% 20%, rgba(99,102,241,0.10), transparent),
                  var(--bg);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 22px 18px 40px;
    }}
    header {{
      display:flex; gap:14px; align-items:flex-end; justify-content:space-between; flex-wrap:wrap;
      margin-bottom: 16px;
    }}
    h1 {{ margin:0; font-size: 22px; font-weight: 700; letter-spacing: .2px; }}
    .sub {{ margin:4px 0 0; color: var(--muted); font-size: 13px; }}
    .top-actions {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
    .pill {{
      display:inline-flex; align-items:center; gap:8px;
      padding: 8px 10px; border:1px solid var(--border); border-radius: 999px;
      background: rgba(15,23,42,0.55); color: var(--muted); font-size: 12px;
    }}
    .btn {{
      border:1px solid var(--border); background: rgba(17,27,46,0.85);
      color: var(--text); border-radius: 10px; padding: 10px 12px;
      cursor:pointer; font-weight: 600; font-size: 12px;
    }}
    .btn:hover {{ border-color: rgba(56,189,248,0.45); }}
    .btn.primary {{
      background: linear-gradient(180deg, rgba(56,189,248,0.20), rgba(56,189,248,0.08));
      border-color: rgba(56,189,248,0.35);
    }}
    .grid {{
      display:grid;
      grid-template-columns: 1.05fr 1.25fr;
      gap: 14px;
      align-items: stretch;
    }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
    .card {{
      border:1px solid var(--border);
      background: rgba(17,27,46,0.75);
      border-radius: 16px;
      overflow:hidden;
      backdrop-filter: blur(8px);
    }}
    .card .hd {{
      padding: 12px 14px;
      background: rgba(15,23,42,0.70);
      border-bottom:1px solid var(--border);
      display:flex; gap:10px; align-items:center; justify-content:space-between; flex-wrap:wrap;
    }}
    .card .hd h2 {{
      margin:0; font-size: 14px; letter-spacing:.2px;
      color: var(--muted); font-weight: 700;
    }}
    .card .bd {{ padding: 12px 14px; }}
    .filters {{
      display:flex; gap:8px; flex-wrap:wrap; align-items:center;
    }}
    .chip {{
      padding: 7px 10px; border:1px solid var(--border);
      border-radius: 999px; cursor:pointer;
      font-size: 12px; color: var(--muted);
      background: rgba(15,23,42,0.50);
      user-select:none;
    }}
    .chip.active {{
      border-color: rgba(56,189,248,0.50);
      color: var(--text);
      background: rgba(56,189,248,0.10);
    }}
    .search {{
      width: 280px; max-width: 100%;
      padding: 9px 12px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(15,23,42,0.55);
      color: var(--text);
      outline:none;
    }}
    .table {{
      width:100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .table th, .table td {{
      padding: 10px 10px;
      border-bottom: 1px solid rgba(148,163,184,0.12);
      vertical-align: top;
    }}
    .table th {{
      text-transform: uppercase;
      letter-spacing: .08em;
      font-size: 11px;
      color: var(--muted);
      background: rgba(15,23,42,0.60);
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .row {{
      cursor:pointer;
    }}
    .row:hover td {{
      background: rgba(56,189,248,0.06);
    }}
    .badge {{
      display:inline-flex; align-items:center;
      padding: 2px 8px; border-radius: 999px;
      font-size: 11px; font-weight: 800;
      letter-spacing: .05em; text-transform: uppercase;
      border: 1px solid rgba(148,163,184,0.25);
      background: rgba(15,23,42,0.50);
    }}
    .b-retro {{ color: #c7d2fe; border-color: rgba(99,102,241,0.35); background: rgba(99,102,241,0.12); }}
    .b-run {{ color: #86efac; border-color: rgba(34,197,94,0.35); background: rgba(34,197,94,0.12); }}
    .b-mix {{ color: #fed7aa; border-color: rgba(249,115,22,0.35); background: rgba(249,115,22,0.12); }}
    .b-unk {{ color: var(--muted); }}
    .muted {{ color: var(--muted); }}
    .detail {{
      max-height: 620px;
      overflow: auto;
      padding-right: 6px;
    }}
    .bubble {{
      max-width: 92%;
      padding: 10px 12px;
      border-radius: 14px;
      margin: 10px 0;
      white-space: pre-wrap;
      line-height: 1.35;
      border: 1px solid rgba(148,163,184,0.12);
    }}
    .u {{
      margin-left:auto;
      background: rgba(148,163,184,0.08);
    }}
    .a {{
      margin-right:auto;
      background: rgba(56,189,248,0.08);
      border-color: rgba(56,189,248,0.18);
    }}
    .ts {{
      font-size: 11px; color: var(--muted);
      margin: 0 0 8px 2px;
    }}
    .tabs {{
      display:flex; gap:8px; flex-wrap:wrap;
    }}
    .tab {{
      padding: 8px 10px;
      border-radius: 10px;
      border:1px solid var(--border);
      cursor:pointer;
      background: rgba(15,23,42,0.50);
      color: var(--muted);
      font-weight: 700;
      font-size: 12px;
    }}
    .tab.active {{
      color: var(--text);
      border-color: rgba(56,189,248,0.45);
      background: rgba(56,189,248,0.10);
    }}
    .testgrid {{
      display:grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }}
    .select {{
      border:1px solid var(--border);
      background: rgba(15,23,42,0.55);
      color: var(--text);
      border-radius: 10px;
      padding: 9px 10px;
    }}
    textarea {{
      width:100%;
      min-height: 170px;
      border: 1px solid var(--border);
      background: rgba(15,23,42,0.55);
      color: var(--text);
      border-radius: 14px;
      padding: 10px 12px;
      outline:none;
      resize: vertical;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.35;
    }}
    .out {{
      border: 1px solid rgba(148,163,184,0.16);
      background: rgba(15,23,42,0.35);
      border-radius: 14px;
      padding: 10px 12px;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.35;
      max-height: 260px;
      overflow:auto;
    }}
    .kbd {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 11px;
      padding: 1px 6px;
      border-radius: 6px;
      border: 1px solid rgba(148,163,184,0.25);
      background: rgba(15,23,42,0.45);
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>Admin IA</h1>
        <div class="sub">Retroworld / Runningman, logs + test rapide (JSON ou multi-questions)</div>
      </div>
      <div class="top-actions">
        <div class="pill"><span class="kbd">Token</span> {html.escape(token)}</div>
        <button class="btn primary" id="refreshBtn">Rafraîchir</button>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="hd">
          <h2>Conversations</h2>
          <div class="filters">
            <input class="search" id="search" placeholder="Recherche (message, source, ID…)" />
            <span class="chip active" data-filter="all">Tout</span>
            <span class="chip" data-filter="runningman">Runningman</span>
            <span class="chip" data-filter="retroworld">Retroworld</span>
            <span class="chip" data-filter="mixed">Mix</span>
          </div>
        </div>
        <div class="bd" style="padding:0;">
          <div style="max-height:620px; overflow:auto;">
            <table class="table">
              <thead>
                <tr>
                  <th style="width: 160px;">Date</th>
                  <th style="width: 90px;">Canal</th>
                  <th style="width: 120px;">Marque</th>
                  <th style="width: 140px;">Source</th>
                  <th>Dernier message</th>
                </tr>
              </thead>
              <tbody id="rows">
                <tr><td colspan="5" class="muted" style="padding:14px;">Chargement…</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="hd">
          <div class="tabs">
            <span class="tab active" data-tab="detail">Détail</span>
            <span class="tab" data-tab="test">Test</span>
          </div>
          <div class="muted" id="rightTitle">Sélectionnez une conversation</div>
        </div>

        <div class="bd" id="panelDetail">
          <div class="detail" id="convDetail">
            <div class="muted">Cliquez une conversation à gauche pour afficher le fil complet.</div>
          </div>
        </div>

        <div class="bd" id="panelTest" style="display:none;">
          <div class="testgrid">
            <div style="display:flex; gap:10px; flex-wrap:wrap;">
              <select class="select" id="testBrand">
                <option value="runningman">Runningman</option>
                <option value="retroworld">Retroworld</option>
              </select>
              <select class="select" id="testMode">
                <option value="text">Multi-questions (1 par ligne)</option>
                <option value="json">JSON brut (payload /chat)</option>
              </select>
              <button class="btn primary" id="runTest">Tester</button>
              <span class="muted" style="align-self:center;">Astuce : collez plusieurs questions ligne par ligne.</span>
            </div>

            <textarea id="testInput" placeholder="Mode multi-questions:
Adresse ?
Il reste des places pour Halloween ?
Je veux réserver ce samedi 15/02

Mode JSON:
{{&#10;  &quot;messages&quot;: [{{&quot;role&quot;:&quot;user&quot;,&quot;content&quot;:&quot;Bonjour&quot;}}],&#10;  &quot;metadata&quot;: {{&quot;source&quot;:&quot;admin_test&quot;}}&#10;}}"></textarea>

            <div class="out" id="testOutput">Résultats…</div>
          </div>
        </div>
      </div>

    </div>
  </div>

<script>
(function() {{
  const token = new URLSearchParams(window.location.search).get("token") || "{html.escape(token)}";
  const rowsEl = document.getElementById("rows");
  const searchEl = document.getElementById("search");
  const refreshBtn = document.getElementById("refreshBtn");
  const chips = Array.from(document.querySelectorAll(".chip"));
  const tabs = Array.from(document.querySelectorAll(".tab"));
  const panelDetail = document.getElementById("panelDetail");
  const panelTest = document.getElementById("panelTest");
  const rightTitle = document.getElementById("rightTitle");
  const convDetail = document.getElementById("convDetail");

  const testBrand = document.getElementById("testBrand");
  const testMode = document.getElementById("testMode");
  const testInput = document.getElementById("testInput");
  const runTest = document.getElementById("runTest");
  const testOutput = document.getElementById("testOutput");

  let allData = [];
  let currentFilter = "all";
  let searchTerm = "";

  function escapeHtml(s) {{
    return String(s || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }}

  function formatDate(ts) {{
    if (!ts) return "";
    try {{
      const d = new Date(ts * 1000);
      return d.toLocaleString("fr-FR", {{
        day: "2-digit", month: "2-digit", year: "2-digit",
        hour: "2-digit", minute: "2-digit"
      }});
    }} catch(e) {{ return ""; }}
  }}

  function brandBadge(b) {{
    if (b === "runningman") return '<span class="badge b-run">Runningman</span>';
    if (b === "retroworld") return '<span class="badge b-retro">Retroworld</span>';
    if (b === "mixed") return '<span class="badge b-mix">Mix</span>';
    return '<span class="badge b-unk">Inconnu</span>';
  }}

  async function loadData() {{
    rowsEl.innerHTML = '<tr><td colspan="5" class="muted" style="padding:14px;">Chargement…</td></tr>';
    try {{
      const res = await fetch(`/admin/api/conversations?token=${{encodeURIComponent(token)}}`);
      if (!res.ok) {{
        rowsEl.innerHTML = `<tr><td colspan="5" class="muted" style="padding:14px;">Erreur (${{res.status}})</td></tr>`;
        return;
      }}
      allData = await res.json();
      render();
    }} catch(e) {{
      rowsEl.innerHTML = '<tr><td colspan="5" class="muted" style="padding:14px;">Erreur réseau</td></tr>';
    }}
  }}

  function render() {{
    const term = searchTerm.trim().toLowerCase();
    let filtered = allData.slice();

    if (currentFilter !== "all") {{
      filtered = filtered.filter(c => {{
        if (currentFilter === "mixed") return c.brand_final === "mixed";
        return c.brand_final === currentFilter;
      }});
    }}

    if (term) {{
      filtered = filtered.filter(c =>
        (c.preview && c.preview.toLowerCase().includes(term)) ||
        (c.source && c.source.toLowerCase().includes(term)) ||
        (c.conversation_id && c.conversation_id.toLowerCase().includes(term))
      );
    }}

    if (!filtered.length) {{
      rowsEl.innerHTML = '<tr><td colspan="5" class="muted" style="padding:14px;">Aucune conversation</td></tr>';
      return;
    }}

    rowsEl.innerHTML = filtered.map(c => `
      <tr class="row" data-id="${{escapeHtml(c.conversation_id)}}">
        <td>
          <div>${{escapeHtml(formatDate(c.timestamp))}}</div>
          <div class="muted" style="font-size:11px;">${{escapeHtml(c.conversation_id)}}</div>
        </td>
        <td><span class="badge">${{escapeHtml((c.channel||"web").toUpperCase())}}</span></td>
        <td>${{brandBadge(c.brand_final)}}</td>
        <td><span class="badge">${{escapeHtml(c.source||"n/a")}}</span></td>
        <td>${{escapeHtml(c.preview||"")}}</td>
      </tr>
    `).join("");

    Array.from(document.querySelectorAll(".row")).forEach(r => {{
      r.addEventListener("click", () => viewConversation(r.getAttribute("data-id")));
    }});
  }}

  async function viewConversation(id) {{
    if (!id) return;
    rightTitle.textContent = `Conversation ${id}`;
    convDetail.innerHTML = '<div class="muted">Chargement…</div>';
    try {{
      const res = await fetch(`/admin/api/conversation/${{encodeURIComponent(id)}}?token=${{encodeURIComponent(token)}}`);
      if (!res.ok) {{
        convDetail.innerHTML = `<div class="muted">Erreur de chargement (${{res.status}})</div>`;
        return;
      }}
      const data = await res.json();
      const records = data.records || [];
      if (!records.length) {{
        convDetail.innerHTML = '<div class="muted">Aucun message</div>';
        return;
      }}
      let html = "";
      for (const rec of records) {{
        const userMsgs = rec.user_messages || [];
        for (const m of userMsgs) {{
          if (m.role === "user") {{
            html += `<div class="bubble u">${{escapeHtml(m.content||"")}}</div>`;
          }}
        }}
        if (rec.assistant_reply) {{
          html += `<div class="bubble a">${{escapeHtml(rec.assistant_reply)}}</div>`;
        }}
        if (rec.timestamp) {{
          const d = new Date(rec.timestamp * 1000).toLocaleString("fr-FR");
          html += `<div class="ts">${{escapeHtml(d)}}</div>`;
        }}
      }}
      convDetail.innerHTML = html;
      convDetail.scrollTop = convDetail.scrollHeight;
    }} catch(e) {{
      convDetail.innerHTML = '<div class="muted">Erreur réseau</div>';
    }}
  }}

  function setTab(name) {{
    tabs.forEach(t => t.classList.toggle("active", t.getAttribute("data-tab") === name));
    if (name === "detail") {{
      panelDetail.style.display = "";
      panelTest.style.display = "none";
      rightTitle.textContent = "Sélectionnez une conversation";
    }} else {{
      panelDetail.style.display = "none";
      panelTest.style.display = "";
      rightTitle.textContent = "Test rapide";
    }}
  }}

  tabs.forEach(t => {{
    t.addEventListener("click", () => setTab(t.getAttribute("data-tab")));
  }});

  chips.forEach(chip => {{
    chip.addEventListener("click", () => {{
      chips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      currentFilter = chip.getAttribute("data-filter") || "all";
      render();
    }});
  }});

  searchEl.addEventListener("input", () => {{
    searchTerm = searchEl.value;
    render();
  }});

  refreshBtn.addEventListener("click", loadData);

  runTest.addEventListener("click", async () => {{
    testOutput.textContent = "Exécution…";
    const mode = testMode.value;
    const brand = testBrand.value;
    const input = testInput.value || "";

    let payload = null;
    if (mode === "json") {{
      try {{
        payload = JSON.parse(input);
      }} catch(e) {{
        testOutput.textContent = "JSON invalide. Corrigez le JSON puis relancez.";
        return;
      }}
    }}

    try {{
      const res = await fetch(`/admin/api/test_chat?token=${{encodeURIComponent(token)}}`, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(mode === "json"
          ? {{ mode, brand, payload }}
          : {{ mode, brand, text: input }}
        )
      }});

      const data = await res.json();
      testOutput.textContent = JSON.stringify(data, null, 2);
    }} catch(e) {{
      testOutput.textContent = "Erreur réseau pendant le test.";
    }}
  }});

  loadData();
}})();
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
