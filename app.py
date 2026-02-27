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
import importlib
import smtplib
import traceback
from collections import deque
import importlib.util
import json
import os
import re
import time
import uuid
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional
from email.message import EmailMessage

from flask import Flask, jsonify, make_response, redirect, request, send_from_directory
from werkzeug.exceptions import HTTPException

from src.data.system_data import SYSTEM_PROMPT
from src.services.conversation_store import ConversationStore
from src.services.metrics import RuntimeMetrics
from src.services.rate_limit import InMemoryRateLimiter

requests = importlib.import_module("requests") if importlib.util.find_spec("requests") else None


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


def _env_float(key: str, default: float) -> float:
    raw = _env(key, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = _env(key, str(default))
    try:
        return int(raw)
    except ValueError:
        return default

# OpenAI
OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_FALLBACK_MODEL = _env("OPENAI_FALLBACK_MODEL", "gpt-4.1-mini")
OPENAI_API_MODE = _env("OPENAI_API_MODE", "auto").lower()
OPENAI_REASONING_EFFORT = _env("OPENAI_REASONING_EFFORT", "none")
OPENAI_TEMPERATURE = _env_float("OPENAI_TEMPERATURE", 0.3)
OPENAI_MAX_OUTPUT_TOKENS = _env_int("OPENAI_MAX_OUTPUT_TOKENS", 900)

# Sécurité admin / CORS
ADMIN_API_TOKEN = _env("ADMIN_API_TOKEN")
ADMIN_DASHBOARD_TOKEN = _env("ADMIN_DASHBOARD_TOKEN")
ALLOWED_ORIGINS = [o for o in _env("ALLOWED_ORIGINS").split(",") if o.strip()]

# Marque par défaut
BRAND_ID_DEFAULT = _env("BRAND_ID", "retroworld").lower() or "retroworld"

# FAQ publique : marques activées
FAQ_ENABLED_BRANDS = [b.strip() for b in _env("FAQ_ENABLED_BRANDS", "retroworld,runningman,enigmaniac").split(",") if b.strip()]
# Marques proposées dans le widget public
PUBLIC_BRANDS = [b.strip() for b in _env("PUBLIC_BRANDS", ",".join(FAQ_ENABLED_BRANDS)).split(",") if b.strip()]

# URL publique (utilisée dans /brands.json)
PUBLIC_BASE_URL = _env("PUBLIC_BASE_URL")

# Logs de debug
DEBUG_LOGS = _env("DEBUG_LOGS").lower() in ("1", "true", "yes", "on")

# Mode serveur (auto|flask|gunicorn)
SERVER_MODE = _env("SERVER_MODE", "auto").lower()
LOG_BUFFER_MAX = _env_int("ADMIN_LOG_BUFFER_MAX", 300)
APP_LOGS = deque(maxlen=max(LOG_BUFFER_MAX, 50))
CONV_BACKEND = _env("CONV_BACKEND", "json").lower()
CONV_SQLITE_PATH = Path(_env("CONV_SQLITE_PATH", str(DATA_DIR / "conversations.db")))
CHAT_RATE_LIMIT_PER_MIN = _env_int("CHAT_RATE_LIMIT_PER_MIN", 40)
LEAD_EMAIL_ENABLED = _env("LEAD_EMAIL_ENABLED", "false").lower() in ("1", "true", "yes", "on")
# Adresse de réception imposée pour toutes les demandes de lead.
LEAD_EMAIL_TO = "contact@retroworldfrance.com"
SMTP_HOST = _env("SMTP_HOST")
SMTP_PORT = _env_int("SMTP_PORT", 587)
SMTP_USER = _env("SMTP_USER")
SMTP_PASS = _env("SMTP_PASS")
SMTP_FROM = _env("SMTP_FROM", SMTP_USER)

CONV_STORE = ConversationStore(CONV_BACKEND, CONV_DIR, CONV_SQLITE_PATH)
CHAT_RATE_LIMITER = InMemoryRateLimiter(CHAT_RATE_LIMIT_PER_MIN, 60)
RUNTIME_METRICS = RuntimeMetrics()


def _record_log(level: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
    APP_LOGS.append({
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": level,
        "message": message,
        "context": context or {},
    })


def _running_on_render() -> bool:
    return _env("RENDER").lower() in ("1", "true", "yes", "on")


def _gunicorn_cmd(port: int) -> List[str]:
    return ["gunicorn", "-w", "2", "-k", "gthread", "-b", f"0.0.0.0:{port}", "app:app"]


def _should_use_gunicorn() -> bool:
    if SERVER_MODE == "gunicorn":
        return True
    if SERVER_MODE == "flask":
        return False
    # mode auto
    return _running_on_render()


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
        "domains": ["enigmaniac-escapegame.com", "www.enigmaniac-escapegame.com"],
    },
}

# Fusion avec un éventuel fichier YAML de configuration de marques (optionnel)
def _load_brands_from_yaml() -> Dict[str, Dict[str, Any]]:
    cfg_path = BASE_DIR / "config" / "brands.yaml"
    if not cfg_path.exists():
        return {}
    if not importlib.util.find_spec("yaml"):
        return {}
    yaml = importlib.import_module("yaml")
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
    message = " ".join([str(a) for a in args])
    _record_log("debug", message)
    if DEBUG_LOGS:
        print("[DBG]", *args, flush=True)


def log_error(message: str, err: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None) -> None:
    payload = context.copy() if context else {}
    if err is not None:
        payload["error"] = str(err)
        payload["traceback"] = traceback.format_exc(limit=5)
    _record_log("error", message, payload)
    print("[ERR]", message, payload.get("error", ""), flush=True)


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


def build_openai_history(conv: Dict[str, Any], max_items: int = 10) -> List[Dict[str, Any]]:
    msgs = conv.get("messages") or []
    out: List[Dict[str, Any]] = []
    for m in msgs[-max_items:]:
        role = (m.get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def openai_answer(system: str, user: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
    """Interroge OpenAI (Responses puis fallback Chat Completions) et renvoie le texte."""
    if not openai_ready():
        return ""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    convo_input: List[Dict[str, Any]] = [{"role": "system", "content": [{"type": "text", "text": system}]}]
    for h in history or []:
        convo_input.append({"role": h.get("role", "user"), "content": [{"type": "text", "text": h.get("content", "")}]} )
    if not any((h.get("role") == "user" and h.get("content", "").strip() == user.strip()) for h in (history or [])):
        convo_input.append({"role": "user", "content": [{"type": "text", "text": user}]})

    payload: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "input": convo_input,
        "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
    }
    # Mode reasoning effort
    allowed_reasoning_efforts = {"low", "medium", "high"}
    normalized_effort = (OPENAI_REASONING_EFFORT or "").lower().strip()
    if normalized_effort in allowed_reasoning_efforts:
        payload["reasoning"] = {"effort": normalized_effort}
    elif normalized_effort not in ("", "none"):
        log(f"Ignoring unsupported OPENAI_REASONING_EFFORT={OPENAI_REASONING_EFFORT!r}")

    if normalized_effort in ("", "none"):
        payload["temperature"] = OPENAI_TEMPERATURE
    def _perform_responses(payload2: Dict[str, Any]) -> str:
        started = time.perf_counter()
        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload2,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        out_texts: List[str] = []
        for item in data.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    out_texts.append(c.get("text", ""))
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        RUNTIME_METRICS.observe_openai(elapsed_ms, ok=True)
        return "\n".join(out_texts).strip()

    def _perform_chat_completions(model: str) -> str:
        started = time.perf_counter()
        msg_payload: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        for h in history or []:
            msg_payload.append({"role": h.get("role", "user"), "content": h.get("content", "")})
        if not any((h.get("role") == "user" and h.get("content", "").strip() == user.strip()) for h in (history or [])):
            msg_payload.append({"role": "user", "content": user})
        payload3: Dict[str, Any] = {
            "model": model,
            "messages": msg_payload,
            "max_tokens": OPENAI_MAX_OUTPUT_TOKENS,
            "temperature": OPENAI_TEMPERATURE,
        }
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload3,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        RUNTIME_METRICS.observe_openai(elapsed_ms, ok=True)
        return content

    prefer_chat_completions = OPENAI_API_MODE == "chat_completions" or (
        OPENAI_API_MODE == "auto" and OPENAI_MODEL.startswith("gpt-5")
    )
    if prefer_chat_completions:
        try:
            return _perform_chat_completions(OPENAI_MODEL)
        except Exception as e:
            _record_log("warning", "OpenAI chat.completions primary error", {"model": OPENAI_MODEL, "error": str(e)})
            fallback_model = OPENAI_FALLBACK_MODEL or OPENAI_MODEL
            if fallback_model != OPENAI_MODEL:
                try:
                    return _perform_chat_completions(fallback_model)
                except Exception as e2:
                    RUNTIME_METRICS.observe_openai(0, ok=False)
                    log_error("OpenAI chat.completions fallback error", e2, {"model": fallback_model})
                    return "Désolé, je rencontre un souci technique. Pouvez‑vous réessayer ou contacter l’équipe ?"
            RUNTIME_METRICS.observe_openai(0, ok=False)
            return "Désolé, je rencontre un souci technique. Pouvez‑vous réessayer ou contacter l’équipe ?"

    # Requête HTTP
    try:
        return _perform_responses(payload)
    except requests.exceptions.HTTPError as e:  # type: ignore[attr-defined]
        status = getattr(getattr(e, "response", None), "status_code", 0)
        if status != 400:
            RUNTIME_METRICS.observe_openai(0, ok=False)
            log_error("OpenAI error", e, {"model": OPENAI_MODEL})
            return "Désolé, je rencontre un souci technique. Pouvez‑vous réessayer ou contacter l’équipe ?"

        # Retry défensif: retire les champs optionnels et applique un modèle de fallback.
        retry_payload = {
            "model": OPENAI_FALLBACK_MODEL or OPENAI_MODEL,
            "input": payload.get("input", []),
            "max_output_tokens": payload.get("max_output_tokens", OPENAI_MAX_OUTPUT_TOKENS),
        }
        try:
            log("Retrying OpenAI call after 400", {"from_model": OPENAI_MODEL, "to_model": retry_payload["model"]})
            return _perform_responses(retry_payload)
        except Exception as e2:
            _record_log("warning", "OpenAI retry error", {"model": retry_payload.get("model", ""), "error": str(e2)})
            try:
                log("Falling back to chat.completions", {"model": retry_payload.get("model", "")})
                return _perform_chat_completions(str(retry_payload.get("model", OPENAI_FALLBACK_MODEL or OPENAI_MODEL)))
            except Exception as e3:
                RUNTIME_METRICS.observe_openai(0, ok=False)
                log_error("OpenAI chat.completions fallback error", e3, {"model": retry_payload.get("model", "")})
                return "Désolé, je rencontre un souci technique. Pouvez‑vous réessayer ou contacter l’équipe ?"
    except Exception as e:
        RUNTIME_METRICS.observe_openai(0, ok=False)
        log_error("OpenAI error", e, {"model": OPENAI_MODEL})
        return "Désolé, je rencontre un souci technique. Pouvez‑vous réessayer ou contacter l’équipe ?"


# -----------------------------------------------------------------------------
# Gestion des conversations (stockage JSON)
# -----------------------------------------------------------------------------

def conv_path(conv_id: str) -> Path:
    return CONV_DIR / f"{conv_id}.json"


def new_conv_id(prefix: str = "rw") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def load_conv(conv_id: str) -> Dict[str, Any]:
    try:
        return CONV_STORE.load(conv_id)
    except Exception:
        return {"id": conv_id, "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "messages": [], "meta": {}}


def save_conv(conv: Dict[str, Any]) -> None:
    try:
        CONV_STORE.save(conv)
    except Exception as e:
        log_error("save_conv error", e, {"conversation_id": conv.get("id", "unknown")})


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


def enforce_chat_rate_limit() -> Optional[Any]:
    ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown").split(",", 1)[0].strip()
    allowed, retry_after = CHAT_RATE_LIMITER.allow(ip)
    if allowed:
        RUNTIME_METRICS.observe_chat_request(rate_limited=False)
        return None
    RUNTIME_METRICS.observe_chat_request(rate_limited=True)
    return jsonify({"ok": False, "error": "rate_limited", "retry_after": retry_after}), 429


def _is_lead_request(text: str) -> bool:
    t = (text or "").lower()
    patterns = [
        r"\bdevis\b",
        r"\bentreprise\b",
        r"team\s*building",
        r"\bprivatis",
        r"\bcontact\b",
        r"rappel",
        r"appelez\s*moi",
        r"recontact",
    ]
    return any(re.search(p, t, flags=re.IGNORECASE) for p in patterns)


def _extract_emails(text: str) -> List[str]:
    raw = re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", (text or ""), flags=re.IGNORECASE)
    return [e.lower() for e in raw]


def _is_fraudulent_or_useless_lead(text: str) -> bool:
    t = (text or "").lower().strip()
    if len(t) < 8:
        return True

    blocked_patterns = [
        r"\btest\b",
        r"\bspam\b",
        r"\bhack\b",
        r"\barnaque\b",
        r"\bfake\b",
        r"\bscam\b",
        r"envoi\s+un\s+mail\s+a\s+tes\s+createur",
        r"\bbonjour\b$",
    ]
    if any(re.search(p, t, flags=re.IGNORECASE) for p in blocked_patterns):
        return True

    disposable_domains = {
        "mailinator.com",
        "yopmail.com",
        "temp-mail.org",
        "10minutemail.com",
        "guerrillamail.com",
        "trashmail.com",
    }
    for e in _extract_emails(t):
        domain = e.split("@", 1)[1]
        if domain in disposable_domains:
            return True
    return False


def _send_lead_email(brand_id: str, message: str, conversation_id: str, user_context: Optional[Dict[str, Any]] = None) -> bool:
    if not LEAD_EMAIL_ENABLED:
        return False
    if not (SMTP_HOST and SMTP_FROM and LEAD_EMAIL_TO):
        _record_log("warning", "lead_email skipped: missing SMTP config", {"brand_id": brand_id})
        return False

    ctx = user_context or {}
    msg = EmailMessage()
    msg["Subject"] = f"[Lead IA] {brand_id} - demande devis/contact"
    msg["From"] = SMTP_FROM
    msg["To"] = LEAD_EMAIL_TO
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    body = (
        f"Brand: {brand_id}\n"
        f"Conversation: {conversation_id}\n"
        f"Time: {now_str}\n"
        f"IP: {ctx.get('ip', '')}\n"
        f"Origin: {ctx.get('origin', '')}\n\n"
        f"Message utilisateur:\n{message}\n"
    )
    msg.set_content(body)
    safe_message = escape(message).replace("\n", "<br>")
    html_body = f"""
<html>
  <body style="margin:0;background:#f4f6fb;font-family:Arial,sans-serif;color:#1f2937;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="padding:24px 12px;">
      <tr>
        <td align="center">
          <table role="presentation" width="680" cellspacing="0" cellpadding="0" style="max-width:680px;background:#ffffff;border-radius:14px;overflow:hidden;border:1px solid #e5e7eb;box-shadow:0 8px 26px rgba(0,0,0,0.08);">
            <tr>
              <td style="background:linear-gradient(120deg,#7c3aed,#2563eb);padding:20px 24px;color:#fff;">
                <h1 style="margin:0;font-size:20px;">🎯 Nouveau lead IA</h1>
                <p style="margin:8px 0 0;font-size:14px;opacity:0.95;">Demande devis/contact détectée automatiquement.</p>
              </td>
            </tr>
            <tr>
              <td style="padding:22px 24px;">
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="font-size:14px;">
                  <tr><td style="padding:8px 0;color:#6b7280;width:160px;">Marque</td><td style="padding:8px 0;font-weight:600;">{escape(brand_id)}</td></tr>
                  <tr><td style="padding:8px 0;color:#6b7280;">Conversation</td><td style="padding:8px 0;font-weight:600;">{escape(conversation_id)}</td></tr>
                  <tr><td style="padding:8px 0;color:#6b7280;">Date</td><td style="padding:8px 0;">{escape(now_str)}</td></tr>
                  <tr><td style="padding:8px 0;color:#6b7280;">IP</td><td style="padding:8px 0;">{escape(ctx.get('ip', ''))}</td></tr>
                  <tr><td style="padding:8px 0;color:#6b7280;">Origine</td><td style="padding:8px 0;">{escape(ctx.get('origin', '') or '-')}</td></tr>
                </table>
                <div style="margin-top:18px;padding:16px;border-radius:10px;background:#f9fafb;border:1px solid #e5e7eb;">
                  <div style="font-size:13px;color:#6b7280;margin-bottom:8px;">Message utilisateur</div>
                  <div style="font-size:15px;line-height:1.55;">{safe_message}</div>
                </div>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
""".strip()
    msg.add_alternative(html_body, subtype="html")

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            try:
                server.starttls()
            except Exception:
                pass
            if SMTP_USER and SMTP_PASS:
                server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        _record_log("info", "lead_email sent", {"brand_id": brand_id, "conversation_id": conversation_id})
        return True
    except Exception as e:
        log_error("lead_email error", e, {"brand_id": brand_id, "conversation_id": conversation_id})
        return False


def _process_lead_email_if_needed(brand_id: str, msg: str, conv_id: str, conv: Dict[str, Any]) -> bool:
    if not _is_lead_request(msg):
        conv.setdefault("meta", {})
        conv["meta"]["lead_email_sent"] = False
        conv["meta"]["lead_email_reason"] = "not_lead"
        return False
    if _is_fraudulent_or_useless_lead(msg):
        conv.setdefault("meta", {})
        conv["meta"]["lead_email_sent"] = False
        conv["meta"]["lead_email_reason"] = "filtered"
        _record_log("warning", "lead_email filtered", {"brand_id": brand_id, "conversation_id": conv_id})
        return False
    sent = _send_lead_email(
        brand_id,
        msg,
        conv_id,
        user_context={
            "ip": (request.headers.get("X-Forwarded-For") or request.remote_addr or "").split(",", 1)[0].strip(),
            "origin": (request.headers.get("Origin") or "").strip(),
        },
    )
    conv.setdefault("meta", {})
    conv["meta"]["lead_email_sent"] = bool(sent)
    conv["meta"]["lead_email_reason"] = "sent" if sent else "failed"
    return bool(sent)


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


def _allowed_origin_candidates() -> set:
    allowed = {o.strip().rstrip("/") for o in ALLOWED_ORIGINS if o.strip()}
    for cfg in BRANDS.values():
        for d in cfg.get("domains", []) or []:
            dom = (d or "").strip().lower()
            if not dom:
                continue
            allowed.add(f"https://{dom}")
            allowed.add(f"http://{dom}")
    return allowed


@app.after_request
def after(resp):
    """Applique la politique CORS en fin de requête."""
    origin = (request.headers.get("Origin") or "").strip().rstrip("/")
    if ALLOWED_ORIGINS:
        if origin in _allowed_origin_candidates():
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
            "server_mode": SERVER_MODE,
            "running_on_render": _running_on_render(),
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


@app.route("/faq/<brand_id>", methods=["GET"], strict_slashes=False)
def faq_page_by_brand(brand_id: str):
    """Alias de compatibilité: FAQ publique via URL segmentée (/faq/<brand>)."""
    bid = normalize_brand(brand_id)
    if bid not in FAQ_ENABLED_BRANDS:
        return make_response("FAQ indisponible pour le moment.", 404)
    wants_json = "application/json" in (request.headers.get("Accept") or "").lower()
    if wants_json:
        p = STATIC_DIR / f"faq_{bid}.json"
        try:
            data = json.loads(p.read_text("utf-8"))
        except Exception:
            data = {"brand": bid, "items": []}
        return jsonify({"brand": bid, "updated": data.get("updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "items": data.get("items", [])})
    return redirect(f"/static/chat-widget.html?tab=faq&brand={bid}")


@app.route("/faq/<brand_id>.json", methods=["GET"], strict_slashes=False)
def faq_json_by_brand_alias(brand_id: str):
    """Alias legacy: FAQ JSON directe par marque (/faq/<brand>.json)."""
    bid = normalize_brand(brand_id)
    if bid not in FAQ_ENABLED_BRANDS:
        return jsonify({"brand": bid, "items": [], "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}), 404
    p = STATIC_DIR / f"faq_{bid}.json"
    try:
        data = json.loads(p.read_text("utf-8"))
    except Exception:
        data = {"brand": bid, "items": []}
    return jsonify({"brand": bid, "updated": data.get("updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "items": data.get("items", [])})


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


@app.route("/faq_enigmaniac.json", methods=["GET"])
def faq_enigmaniac_alias():
    return send_from_directory(str(STATIC_DIR), "faq_enigmaniac.json")


# ------------------ CHAT ------------------
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    """Endpoint principal pour le chatbot."""
    if request.method == "OPTIONS":
        return ("", 204)
    limited = enforce_chat_rate_limit()
    if limited is not None:
        return limited
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        payload = {}
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
    lead_sent = _process_lead_email_if_needed(brand_id, msg, conv_id, conv)
    # Appel OpenAI
    if not openai_ready():
        answer = "Le service IA n'est pas configuré (OPENAI_API_KEY manquante)."
        flags = ["openai_missing"] + (["lead_email"] if lead_sent else [])
        append_message(conv, "assistant", answer, extra={"brand_id": brand_id, "flags": flags})
        save_conv(conv)
        return jsonify({"ok": True, "conversation_id": conv_id, "brand_id": brand_id, "answer": answer})
    raw_answer = openai_answer(sys_prompt, user_prompt, history=build_openai_history(conv, max_items=12))
    safe_answer, promised = enforce_no_reservation_promises(raw_answer)
    safe_answer = add_disclaimer_if_needed(safe_answer, brand_id, msg)
    # Ajout des liens Qweekle pour Retroworld si pertinent
    if brand_id == "retroworld":
        safe_answer = _append_retroworld_links_if_missing(msg, safe_answer)
    flags: List[str] = []
    if promised:
        flags.append("promesse_resa")
    if lead_sent:
        flags.append("lead_email")
    append_message(conv, "assistant", safe_answer, extra={"brand_id": brand_id, "flags": flags})
    save_conv(conv)
    return jsonify({"ok": True, "conversation_id": conv_id, "brand_id": brand_id, "answer": safe_answer})


@app.route("/chat/<brand_id>", methods=["POST", "OPTIONS"], strict_slashes=False)
def chat_by_brand(brand_id: str):
    """Alias de compatibilité: endpoint brandé (/chat/<brand>)."""
    if request.method == "OPTIONS":
        return ("", 204)
    limited = enforce_chat_rate_limit()
    if limited is not None:
        return limited

    bid = normalize_brand(brand_id)
    if bid not in BRANDS:
        return jsonify({"ok": False, "error": "unknown brand"}), 404

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("brand_id", bid)

    msg = (payload.get("message") or "").strip()
    if not msg:
        return jsonify({"ok": False, "error": "message manquant"}), 400

    conv_id = (payload.get("conversation_id") or "").strip()
    if not conv_id:
        conv_id = new_conv_id(prefix=bid[:2] if bid else "rw")
    conv = load_conv(conv_id)
    conv.setdefault("meta", {})
    conv["meta"]["brand_id"] = bid
    conv["meta"]["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_message(conv, "user", msg)

    sys_prompt = build_system_prompt(bid, msg)
    user_prompt = msg
    lead_sent = _process_lead_email_if_needed(bid, msg, conv_id, conv)
    if not openai_ready():
        answer = "Le service IA n'est pas configuré (OPENAI_API_KEY manquante)."
        flags = ["openai_missing"] + (["lead_email"] if lead_sent else [])
        append_message(conv, "assistant", answer, extra={"brand_id": bid, "flags": flags})
        save_conv(conv)
        return jsonify({"ok": True, "conversation_id": conv_id, "brand_id": bid, "answer": answer})

    raw_answer = openai_answer(sys_prompt, user_prompt, history=build_openai_history(conv, max_items=12))
    safe_answer, promised = enforce_no_reservation_promises(raw_answer)
    safe_answer = add_disclaimer_if_needed(safe_answer, bid, msg)
    if bid == "retroworld":
        safe_answer = _append_retroworld_links_if_missing(msg, safe_answer)
    flags: List[str] = []
    if promised:
        flags.append("promesse_resa")
    if lead_sent:
        flags.append("lead_email")
    append_message(conv, "assistant", safe_answer, extra={"brand_id": bid, "flags": flags})
    save_conv(conv)
    return jsonify({"ok": True, "conversation_id": conv_id, "brand_id": bid, "answer": safe_answer})


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
    faq_files = []
    for bid in BRANDS.keys():
        fp = STATIC_DIR / f"faq_{bid}.json"
        try:
            file_display = str(fp.relative_to(BASE_DIR))
        except Exception:
            file_display = str(fp)
        faq_files.append(
            {
                "brand_id": bid,
                "file": file_display,
                "exists": fp.exists(),
                "size_bytes": fp.stat().st_size if fp.exists() else 0,
            }
        )
    return jsonify(
        {
            "ok": True,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "openai_configured": openai_ready(),
            "brands": list(BRANDS.keys()),
            "faq_enabled_brands": FAQ_ENABLED_BRANDS,
            "public_brands": PUBLIC_BRANDS,
            "server_mode": SERVER_MODE,
            "running_on_render": _running_on_render(),
            "allowed_origins": ALLOWED_ORIGINS,
            "conversations_count": CONV_STORE.count(),
            "conversation_backend": CONV_BACKEND,
            "chat_rate_limit_per_min": CHAT_RATE_LIMIT_PER_MIN,
            "runtime_metrics": RUNTIME_METRICS.snapshot(),
            "faq_files": faq_files,
            "recent_logs": list(APP_LOGS)[-80:],
        }
    )


@app.route("/admin/api/conversations", methods=["GET"])
def admin_list_conversations():
    if not require_admin_token():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    items: List[Dict[str, Any]] = []
    for conv in CONV_STORE.list_all():
        cid = conv.get("id", "")
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
    for conv in CONV_STORE.list_all():
        cid = conv.get("id", "")
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


@app.route("/admin/api/brands", methods=["GET"])
def admin_api_brands_compat():
    if not require_admin_token():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    faq_only = (request.args.get("faq_only") or "").strip().lower() in ("1", "true", "yes", "on")
    source = FAQ_ENABLED_BRANDS if faq_only else list(BRANDS.keys())
    brands = []
    for bid in source:
        cfg = BRANDS.get(bid, {})
        brands.append(
            {
                "id": bid,
                "name": cfg.get("name", bid),
                "display_name": cfg.get("name", bid),
            }
        )
    return jsonify({"ok": True, "brands": brands})


@app.route("/admin/api/faq/<brand_id>", methods=["GET", "PUT"])
def admin_api_faq_compat(brand_id: str):
    if not require_admin_token():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    bid = normalize_brand(brand_id)
    if bid not in BRANDS:
        return jsonify({"ok": False, "error": "unknown brand"}), 400

    fp = STATIC_DIR / f"faq_{bid}.json"
    if request.method == "GET":
        try:
            kb = json.loads(fp.read_text("utf-8"))
        except Exception:
            kb = _default_faq_payload(bid, include_default_items=True)
            fp.write_text(json.dumps(kb, ensure_ascii=False, indent=2), "utf-8")
        items = []
        for it in kb.get("items", []) or []:
            if not isinstance(it, dict):
                continue
            items.append(
                {
                    "q": it.get("question", ""),
                    "a": it.get("answer", ""),
                    "tags": it.get("tags", []) if isinstance(it.get("tags", []), list) else [],
                }
            )
        return jsonify({"brand": bid, "updated": kb.get("updated", ""), "items": items})

    payload = request.get_json(silent=True) or {}
    raw_items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(raw_items, list):
        return jsonify({"ok": False, "error": "items must be a list"}), 400
    cleaned = []
    for it in raw_items:
        if not isinstance(it, dict):
            continue
        q = (it.get("q") or it.get("question") or "").strip()
        a = (it.get("a") or it.get("answer") or "").strip()
        tags = it.get("tags") or []
        if not q or not a:
            continue
        cleaned.append({"question": q, "answer": a, "tags": tags if isinstance(tags, list) else []})
    kb_out = {"brand": bid, "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "items": cleaned}
    fp.write_text(json.dumps(kb_out, ensure_ascii=False, indent=2), "utf-8")
    if bid == "runningman":
        legacy_path = STATIC_DIR / "static" / "faq_runningman.json"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text(json.dumps(kb_out, ensure_ascii=False, indent=2), "utf-8")

    # Return in legacy UI shape (q/a)
    legacy_items = [{"q": x["question"], "a": x["answer"], "tags": x.get("tags", [])} for x in cleaned]
    return jsonify({"brand": bid, "updated": kb_out["updated"], "items": legacy_items})


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
            legacy_path = STATIC_DIR / "static" / "faq_runningman.json"
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path.write_text(
                json.dumps(kb_out, ensure_ascii=False, indent=2), "utf-8"
            )
    except Exception as e:
        log_error("faq_save error", e, {"brand_id": bid})
        return jsonify({"ok": False, "error": "save failed"}), 500
    return jsonify({"ok": True, "saved": True, "updated": kb_out["updated"], "count": len(cleaned)})


def _default_faq_payload(brand_id: str, include_default_items: bool = True) -> Dict[str, Any]:
    bid = normalize_brand(brand_id)
    cfg = BRANDS.get(bid, {})
    items: List[Dict[str, Any]] = []
    if include_default_items:
        contact = format_contact(bid) or "contact indisponible"
        items.append(
            {
                "question": f"Comment contacter {cfg.get('name', bid.title())} ?",
                "answer": f"Vous pouvez contacter l’équipe via {contact}.",
                "tags": ["contact"],
            }
        )
    return {
        "brand": bid,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "items": items,
    }


def ensure_faq_file(brand_id: str, force: bool = False, include_default_items: bool = True) -> Dict[str, Any]:
    bid = normalize_brand(brand_id)
    out_path = STATIC_DIR / f"faq_{bid}.json"
    created = False
    if force or (not out_path.exists()):
        payload = _default_faq_payload(bid, include_default_items=include_default_items)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        created = True
    if bid == "runningman":
        legacy_path = STATIC_DIR / "static" / "faq_runningman.json"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        if force or (not legacy_path.exists()):
            legacy_payload = json.loads(out_path.read_text("utf-8")) if out_path.exists() else _default_faq_payload(bid)
            legacy_path.write_text(json.dumps(legacy_payload, ensure_ascii=False, indent=2), "utf-8")
    try:
        file_display = str(out_path.relative_to(BASE_DIR))
    except Exception:
        file_display = str(out_path)
    return {"brand_id": bid, "file": file_display, "created": created, "exists": out_path.exists()}


@app.route("/admin/api/faq/generate", methods=["POST"])
def admin_faq_generate():
    if not require_admin_token():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    payload = request.get_json(silent=True) or {}
    force = bool(payload.get("force", False))
    include_default_items = bool(payload.get("include_default_items", True))
    target = normalize_brand(payload.get("brand_id") or payload.get("brand") or "")
    targets = [target] if target else [b for b in FAQ_ENABLED_BRANDS if b in BRANDS]
    results = []
    for bid in targets:
        try:
            results.append(ensure_faq_file(bid, force=force, include_default_items=include_default_items))
        except Exception as e:
            log_error("faq_generate error", e, {"brand_id": bid})
            return jsonify({"ok": False, "error": "generate failed", "brand_id": bid}), 500
    return jsonify({"ok": True, "items": results})


# -----------------------------------------------------------------------------
# Fichiers statiques de repli
# -----------------------------------------------------------------------------
@app.route("/static/<path:filename>", methods=["GET"])
def static_files(filename):
    return send_from_directory(str(STATIC_DIR), filename)


@app.route("/robots.txt", methods=["GET"])
def robots_txt():
    """Expose robots.txt without triggering noisy 404 logs from bots/crawlers."""
    path = STATIC_DIR / "robots.txt"
    if path.exists():
        return send_from_directory(str(STATIC_DIR), "robots.txt")
    return make_response("User-agent: *\nAllow: /\n", 200, {"Content-Type": "text/plain; charset=utf-8"})


def _http_error_payload(status_code: int) -> str:
    if status_code == 404:
        return "not_found"
    if status_code == 405:
        return "method_not_allowed"
    return "http_error"


@app.errorhandler(HTTPException)
def handle_http_exception(err: HTTPException):
    """Normalise les erreurs HTTP attendues sans les logger comme crash serveur."""
    status_code = err.code or 500
    _record_log(
        "warning",
        "HTTP exception",
        {
            "path": request.path,
            "method": request.method,
            "status": status_code,
            "error": err.name,
        },
    )
    if request.path.startswith("/admin/api/") or request.path.startswith("/chat"):
        return jsonify({"ok": False, "error": _http_error_payload(status_code)}), status_code
    if status_code == 404:
        return make_response("not found", 404)
    return make_response(err.description or "http error", status_code)


@app.errorhandler(Exception)
def handle_unexpected_error(err: Exception):
    log_error("Unhandled server exception", err, {"path": request.path, "method": request.method})
    if request.path.startswith("/admin/api/") or request.path.startswith("/chat"):
        return jsonify({"ok": False, "error": "internal_error"}), 500
    return make_response("internal server error", 500)


if __name__ == "__main__":
    port = int(_env("PORT", "5000"))
    if _should_use_gunicorn():
        cmd = _gunicorn_cmd(port)
        print("[BOOT] mode gunicorn auto:", " ".join(cmd), flush=True)
        os.execvp(cmd[0], cmd)
    print("[BOOT] mode flask dev", flush=True)
    app.run(host="0.0.0.0", port=port, debug=DEBUG_LOGS)
