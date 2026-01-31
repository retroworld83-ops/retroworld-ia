from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory, make_response
from flask_cors import CORS

# =========================================================
# CONFIG
# =========================================================

SERVICE_NAME = os.getenv("SERVICE_NAME", "retroworld-ia")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

KB_DIR = os.getenv("KB_DIR", BASE_DIR)

PORT = int(os.getenv("PORT", "10000"))

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

# Default updated to a current flagship model (can be overridden by env)
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-5.2").strip()

# GPT-5 family: temperature only supported when reasoning effort is "none".
OPENAI_REASONING_EFFORT = (os.getenv("OPENAI_REASONING_EFFORT") or "none").strip().lower()
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

# Responses API uses max_output_tokens; keep backward compatibility with OPENAI_MAX_TOKENS.
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS") or os.getenv("OPENAI_MAX_TOKENS") or "900")

# Admin token (legacy alias supported)
ADMIN_DASHBOARD_TOKEN = (os.getenv("ADMIN_DASHBOARD_TOKEN") or os.getenv("ADMIN_API_TOKEN") or "").strip()
USER_HISTORY_TOKEN = (os.getenv("USER_HISTORY_TOKEN") or "").strip()

LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(SERVICE_NAME)

app = Flask(__name__, static_folder="static")
CORS(app)

# Cookie fallback (peut être bloqué en iframe => on ne dépend PAS de ça)
CONV_COOKIE_NAME = os.getenv("CONV_COOKIE_NAME", "rw_conv_id")

# =========================================================
# OpenAI client
# =========================================================

_openai_client = None
_openai_init_error: Optional[str] = None
try:
    if OPENAI_API_KEY:
        from openai import OpenAI  # type: ignore

        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    _openai_init_error = str(e)
    logger.warning("OpenAI client init failed: %s", e)
    _openai_client = None

# =========================================================
# Helpers JSON / IDs / time
# =========================================================


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _new_conversation_id() -> str:
    return "conv_" + datetime.utcnow().strftime("%Y%m%d%H%M%S%f")


def _safe_read_json(path: str, default: Any) -> Any:
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _safe_write_json(path: str, data: Any) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.warning("write json failed (%s): %s", path, e)
        return False


# =========================================================
# User index: user_id -> conversation_id
# =========================================================

USER_INDEX_PATH = os.path.join(DATA_DIR, "user_index.json")


def get_user_index() -> Dict[str, str]:
    data = _safe_read_json(USER_INDEX_PATH, {}) or {}
    return data if isinstance(data, dict) else {}


def set_user_conversation(user_id: str, conversation_id: str) -> None:
    if not user_id or not conversation_id:
        return
    idx = get_user_index()
    idx[str(user_id)] = str(conversation_id)
    _safe_write_json(USER_INDEX_PATH, idx)


def get_user_conversation(user_id: str) -> Optional[str]:
    if not user_id:
        return None
    idx = get_user_index()
    return idx.get(str(user_id))


# =========================================================
# KB cache
# =========================================================


@dataclass
class _KBCacheEntry:
    ts: float
    kb: Dict[str, Any]


_KB_CACHE: Dict[str, _KBCacheEntry] = {}
_KB_CACHE_TTL_SEC = 30.0


def load_kb(brand: str) -> Dict[str, Any]:
    brand = (brand or "").lower().strip()
    if brand not in ("retroworld", "runningman"):
        brand = "retroworld"

    now = time.time()
    entry = _KB_CACHE.get(brand)
    if entry and (now - entry.ts) < _KB_CACHE_TTL_SEC:
        return entry.kb

    filename = f"kb_{brand}.json"
    path = os.path.join(KB_DIR, filename)
    kb = _safe_read_json(path, {"brand": brand})
    if not isinstance(kb, dict):
        kb = {"brand": brand}

    _KB_CACHE[brand] = _KBCacheEntry(ts=now, kb=kb)
    return kb


def save_kb(brand: str, kb: Dict[str, Any]) -> bool:
    brand = (brand or "").lower().strip()
    if brand not in ("retroworld", "runningman"):
        brand = "retroworld"
    filename = f"kb_{brand}.json"
    path = os.path.join(KB_DIR, filename)
    ok = _safe_write_json(path, kb)
    if ok:
        _KB_CACHE[brand] = _KBCacheEntry(ts=time.time(), kb=kb)
    return ok


# =========================================================
# Text utils + intents
# =========================================================


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def detect_owner_from_text(text: str) -> str:
    """Retourne 'runningman' ou 'retroworld' selon la demande.

    Règle voulue par vous:
    - Game Zone / action game / Runningman / salle VR => Runningman
    - Le reste => Retroworld

    ⚠️ Cas particulier: "escape game VR" (même si VR) est traité comme Retroworld
    car c'est une activité distincte (et évite les réponses contradictoires).
    """
    t = _norm(text)

    # Escape game VR => Retroworld
    if ("escape" in t and "vr" in t) or "escape game vr" in t or "escape vr" in t:
        return "retroworld"

    running_keys = [
        "runningman",
        "running man",
        "action game",
        "game zone",
        "gamezone",
        "mini-jeux",
        "mini jeux",
        "défis",
        "defis",
        "salle vr",
        "vr classique",
        "partie vr",
        "jeux vr",
        "jeu vr",
    ]
    if any(k in t for k in running_keys):
        return "runningman"

    retroworld_keys = [
        "retroworld",
        "quiz",
        "anniversaire",
        "goûter",
        "gouter",
        "salle enfant",
        "fidelite",
        "fidélité",
        "points",
        "carte fidelite",
        "carte fidélité",
        "escape",
    ]
    if any(k in t for k in retroworld_keys):
        return "retroworld"

    return "retroworld"  # défaut


def _is_reservation_intent(text: str) -> bool:
    t = _norm(text)
    keys = [
        "réserver",
        "reservation",
        "réservation",
        "dispo",
        "disponible",
        "créneau",
        "creneau",
        "aujourd",
        "demain",
        "samedi",
        "dimanche",
        "lundi",
        "mardi",
        "mercredi",
        "jeudi",
        "vendredi",
        "\b\d{1,2}h\b",
        "\b\d{1,2}h\d{2}\b",
    ]
    # note: include hour regex via search
    if any(k in t for k in keys if "\\b" not in k):
        return True
    if re.search(r"\b\d{1,2}h(\d{2})?\b", t):
        return True
    return False


def _is_price_intent(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in ["c'est combien", "cest combien", "tarif", "prix", "ça coute", "ca coute", "devis"])


def _is_hours_intent(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in ["horaire", "horaires", "ouvert", "ouvre", "ferme", "fermé"])


def _is_location_intent(text: str) -> bool:
    """IMPORTANT:
    - NE PAS tester "ou" (sans accent), sinon "gouter" contient "ou" => bug adresse.
    - On accepte "où" uniquement en mot entier + clés explicites.
    """
    t = _norm(text)
    if re.search(r"\boù\b", t):
        return True
    keys = [
        "adresse",
        "localisation",
        "venir",
        "comment venir",
        "draguignan",
        "parking",
        "vous êtes où",
        "vous etes où",
        "c'est où",
        "c est où",
    ]
    return any(k in t for k in keys)


def _is_gouter_intent(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in ["goûter", "gouter", "anniversaire", "gateau", "gâteau", "formule gouter", "formule goûter"])


def _is_fidelity_intent(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in ["fidélité", "fidelite", "points", "qr", "qr code", "récompense", "recompense"])


def _is_age_intent(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in ["âge", "age", "ans", "enfant", "enfants", "-12", "moins de 12", "12 ans", "7 ans"])


def _is_just_question_mark(text: str) -> bool:
    t = (text or "").strip()
    return t in ["?", "??", "???"]


# =========================================================
# Facts blocks (fiables)
# =========================================================


def common_orientation_block() -> str:
    return (
        "RÉPARTITION (pour orienter sans créer de confusion) :\n"
        "- Runningman gère : Game Zone / Action Game + Salle VR (jeux VR classiques).\n"
        "- Retroworld gère : le reste (Escape Game VR, quiz interactifs, salle enfant, anniversaires/événements, fidélité).\n"
        "- Même bâtiment : 815 avenue Pierre Brossolette, 83300 Draguignan.\n"
    )


def retroworld_facts_block() -> str:
    # Version "safe": aucune promesse de créneau
    return (
        "VÉRITÉS OFFICIELLES RETROWORLD (à respecter strictement) :\n"
        "- Horaires (indicatifs) : mardi à dimanche, 11h à 22h\n"
        "- Adresse : 815 avenue Pierre Brossolette, 83300 Draguignan\n"
        "- Contact : 04 94 47 94 64 / contact@retroworldfrance.com\n"
        "- Site : https://www.retroworldfrance.com\n"
        "- Activités : Escape Game VR, quiz interactifs, salle enfant, anniversaires, fidélité\n"
        "- RÈGLE CRÉNEAUX : le chat ne confirme ni ne bloque jamais un créneau.\n"
        "  Il peut préparer une demande/devis, mais la validation se fait par l'équipe.\n"
        "IMPORTANT : ne jamais dire 'allez voir sur le site' sauf si l'utilisateur demande explicitement un lien.\n"
    )


def runningman_facts_block(kb: Dict[str, Any]) -> str:
    ident = kb.get("identite", {}) or {}
    loc = (ident.get("localisation", {}) or {})
    contact = (ident.get("contact", {}) or {})

    age = kb.get("age_et_accompagnement", {}) or {}
    acc = (age.get("accompagnement", {}) or {})
    age_min = age.get("age_minimum", 7)

    tarif = ((kb.get("tarification", {}) or {}).get("action_game", {}) or {})
    enfant = (tarif.get("enfant", {}) or {})
    adulte = (tarif.get("adulte_accompagnateur", {}) or {})

    cap = ((kb.get("horaires_et_creneaux", {}) or {}).get("capacite", {}) or {})
    cap_msg = cap.get("message") or ""

    return (
        "VÉRITÉS OFFICIELLES RUNNINGMAN (à respecter strictement) :\n"
        f"- Âge minimum : {age_min} ans.\n"
        f"- Moins de 12 ans : {acc.get('moins_de_12_ans', 'Un adulte accompagnateur est requis.')}\n"
        f"- À partir de 12 ans : {acc.get('a_partir_de_12_ans', 'Il n’est plus nécessaire d’avoir un adulte accompagnateur.')}\n"
        "- RÈGLE CRÉNEAUX : le chat ne confirme ni ne bloque jamais un créneau.\n"
        "  Pour réserver/valider un horaire : site officiel ou téléphone.\n"
        f"- Tarifs action game (session 60 min) :\n"
        f"  - Enfant (-12 ans) : {enfant.get('prix', '?')} {enfant.get('unite', '')}\n"
        f"  - Adulte accompagnateur : {adulte.get('prix', '?')} {adulte.get('unite', '')}\n"
        + (f"- Capacité : {cap_msg}\n" if cap_msg else "")
        + f"- Adresse : {loc.get('adresse_complete', '815 avenue Pierre Brossolette, 83300 Draguignan, France')}\n"
        + f"- Contact : {contact.get('telephone', '04 98 09 30 59')} / {contact.get('site_web', 'https://runningmangames.fr')}\n"
        "IMPORTANT : ne jamais confondre “à partir de 12 ans (sans adulte)” avec “âge minimum”.\n"
    )


# =========================================================
# KB to prompt snippets
# =========================================================


def kb_snippets_retroworld(kb: Dict[str, Any], limit_lines: int = 80) -> List[str]:
    out: List[str] = []

    ident = kb.get("identite", {}) or {}
    if isinstance(ident, dict):
        name = ident.get("nom")
        role = ident.get("role")
        if name:
            out.append(f"Identité: {name}")
        if role:
            out.append(f"Mission: {role}")

    prompt = kb.get("prompt", {}) or {}
    if isinstance(prompt, dict):
        for k in [
            "reservation_non_confirmee",
            "gestion_liens_reservation",
            "etape_5_devis",
            "fidelite",
            "redirection_runningman",
            "redirection_enigmaniac",
        ]:
            v = prompt.get(k)
            if isinstance(v, str) and v.strip():
                out.append(f"{k}: {v.strip()}")

    inst = kb.get("instructions_generales")
    if isinstance(inst, list):
        for it in inst:
            s = str(it).strip()
            if s:
                out.append(f"Règle: {s}")

    # Keep it bounded
    if len(out) > limit_lines:
        out = out[:limit_lines]
    return out


def kb_snippets_runningman(kb: Dict[str, Any], limit_lines: int = 60) -> List[str]:
    out: List[str] = []

    # Some structured rules to keep model aligned
    root_rules = kb.get("regles_fondamentales_ia", {}) or {}
    if isinstance(root_rules, dict):
        interdits = root_rules.get("interdictions_absolues")
        if isinstance(interdits, list) and interdits:
            out.append("Interdictions: " + ", ".join([str(x) for x in interdits[:10]]))

    for k in ["instructions_generales", "anti_erreurs"]:
        arr = kb.get(k)
        if isinstance(arr, list):
            for it in arr:
                s = str(it).strip()
                if s:
                    out.append(s)

    # Include contact rule if present
    ident = kb.get("identite", {}) or {}
    if isinstance(ident, dict):
        loc = ident.get("localisation", {}) or {}
        if isinstance(loc, dict):
            rule = loc.get("regle_bot")
            if isinstance(rule, str) and rule.strip():
                out.append(f"Adresse: {rule.strip()}")

    if len(out) > limit_lines:
        out = out[:limit_lines]
    return out


# =========================================================
# OpenAI call (Responses API)
# =========================================================


def call_openai_chat(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    if not _openai_client:
        return (
            "Le service IA est indisponible pour le moment. Vous pouvez nous appeler au 04 94 47 94 64 (Retroworld) ou au 04 98 09 30 59 (Runningman).",
            {"error": "openai_not_ready"},
        )

    kwargs: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "input": messages,
        "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
        "store": False,
    }

    # Reasoning effort (GPT-5 family)
    if OPENAI_REASONING_EFFORT:
        kwargs["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}

    # temperature only allowed when reasoning effort is "none"
    if OPENAI_REASONING_EFFORT == "none":
        kwargs["temperature"] = OPENAI_TEMPERATURE

    resp = _openai_client.responses.create(**kwargs)

    text = (getattr(resp, "output_text", "") or "").strip()

    usage: Dict[str, Any] = {}
    try:
        usage_obj = getattr(resp, "usage", None)
        if usage_obj is not None:
            # openai-python may expose a dict-like or object
            if hasattr(usage_obj, "model_dump"):
                usage = usage_obj.model_dump()  # type: ignore
            elif isinstance(usage_obj, dict):
                usage = usage_obj
            else:
                usage = {
                    "input_tokens": getattr(usage_obj, "input_tokens", None),
                    "output_tokens": getattr(usage_obj, "output_tokens", None),
                    "total_tokens": getattr(usage_obj, "total_tokens", None),
                }
    except Exception:
        usage = {}

    return text, usage


# =========================================================
# Conversation storage (v2, 1 fil)
# =========================================================

CONV_DIR = os.path.join(DATA_DIR, "conversations")
os.makedirs(CONV_DIR, exist_ok=True)


def _conversation_path(conversation_id: str) -> str:
    conversation_id = re.sub(r"[^a-zA-Z0-9_\-]", "", conversation_id or "")
    if not conversation_id:
        conversation_id = _new_conversation_id()
    return os.path.join(CONV_DIR, f"{conversation_id}.json")


def load_conversation_obj(conversation_id: str) -> Dict[str, Any]:
    path = _conversation_path(conversation_id)
    data = _safe_read_json(path, {})

    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        data.setdefault("version", 2)
        data.setdefault("id", conversation_id)
        data.setdefault("created", data.get("created") or _utc_iso())
        data.setdefault("messages", [])
        return data

    return {"version": 2, "id": conversation_id, "created": _utc_iso(), "user_id": None, "brand_last": None, "messages": []}


def save_conversation_obj(conversation_id: str, obj: Dict[str, Any]) -> None:
    path = _conversation_path(conversation_id)
    _safe_write_json(path, obj)


def load_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    obj = load_conversation_obj(conversation_id)
    msgs = obj.get("messages") or []
    return msgs if isinstance(msgs, list) else []


def append_conversation_turn(
    conversation_id: str,
    brand_effective: str,
    user_text: str,
    assistant_reply: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    obj = load_conversation_obj(conversation_id)
    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        msgs = []

    ts = _utc_iso()

    user_text = (user_text or "").strip()
    if user_text:
        msgs.append({"role": "user", "content": user_text, "ts": ts})

    msgs.append({"role": "assistant", "content": str(assistant_reply or ""), "ts": ts})

    obj["version"] = 2
    obj["id"] = conversation_id
    obj.setdefault("created", obj.get("created") or ts)
    obj["brand_last"] = str(brand_effective or "")

    if extra and isinstance(extra, dict):
        obj["extra_last"] = extra
        uid = extra.get("user_id")
        if uid:
            obj["user_id"] = uid

    obj["messages"] = msgs
    save_conversation_obj(conversation_id, obj)


def list_conversations() -> List[str]:
    try:
        files = [f for f in os.listdir(CONV_DIR) if f.endswith(".json")]
        files.sort(reverse=True)
        return [os.path.splitext(f)[0] for f in files]
    except Exception:
        return []


def prune_messages_for_prompt(messages: List[Dict[str, Any]], max_pairs: int = 12) -> List[Dict[str, str]]:
    if not isinstance(messages, list):
        return []
    clipped = [
        m for m in messages
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content") is not None
    ]
    clipped = clipped[-(max_pairs * 2):]
    out: List[Dict[str, str]] = []
    for m in clipped:
        out.append({"role": str(m.get("role")), "content": str(m.get("content") or "")})
    return out


# =========================================================
# Tokens
# =========================================================


def _require_admin_token(req) -> bool:
    tok = (req.args.get("token") or "").strip()
    if not tok:
        tok = (req.headers.get("X-Admin-Token") or "").strip()
    return bool(tok) and bool(ADMIN_DASHBOARD_TOKEN) and tok == ADMIN_DASHBOARD_TOKEN


def _require_user_token(req) -> bool:
    tok = (req.args.get("token") or "").strip()
    if not tok:
        tok = (req.headers.get("X-User-Token") or "").strip()
    return bool(tok) and bool(USER_HISTORY_TOKEN) and tok == USER_HISTORY_TOKEN


# =========================================================
# Payload parsing (support widget: message + user_id + conversation_id)
# =========================================================


def _payload_text(payload: Dict[str, Any]) -> str:
    # priorité au champ "message"
    msg = payload.get("message")
    if msg is not None:
        return str(msg)

    # fallback: messages[{role:user,content}]
    arr = payload.get("messages")
    if isinstance(arr, list):
        for m in reversed(arr):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content") or "")
    return ""


def _payload_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = payload.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def _get_or_create_conversation_id(payload: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """BETON:
    - on accepte conversation_id envoyé par le widget (root)
    - sinon, map via user_id
    - sinon cookie (fallback)
    - sinon génération
    """
    conversation_id = str(payload.get("conversation_id") or "").strip()
    user_id = str(payload.get("user_id") or "").strip()

    if not conversation_id and user_id:
        conversation_id = get_user_conversation(user_id) or ""

    if not conversation_id:
        cookie_val = (request.cookies.get(CONV_COOKIE_NAME) or "").strip()
        if cookie_val:
            conversation_id = cookie_val

    if not conversation_id:
        conversation_id = _new_conversation_id()

    if user_id:
        set_user_conversation(user_id, conversation_id)

    return conversation_id, (user_id or None)


# =========================================================
# Safety: anti-creneaux / anti-confirmation
# =========================================================


_FORBIDDEN_BOOKING_RE = [
    r"\bje (vous )?(confirme|confirme la|confirme votre)\b",
    r"\b(réservation|reservation) (est )?(confirmée|confirmee|validée|validee)\b",
    r"\bje (bloque|bloque donc|bloque votre|réserve|reserve)\b",
    r"\b(nous|on) vous attend(ons)?\b",
    r"\b(c['’]?est|c est) (disponible|ok|bon)\b",
    r"\bdisponible pour\b",
    r"\bcréneau confirmé\b",
]


def sanitize_booking_reply(reply: str, user_text: str, owner: str) -> str:
    """Si on parle réservation/créneau et que le texte dérape (confirmations), on remplace par un message sûr."""
    if not reply:
        return reply

    if not _is_reservation_intent(user_text):
        return reply

    low = _norm(reply)
    if not any(re.search(p, low) for p in _FORBIDDEN_BOOKING_RE):
        return reply

    # Contact selon activité
    kb_rm = load_kb("runningman")
    ident_rm = kb_rm.get("identite", {}) or {}
    contact_rm = (ident_rm.get("contact", {}) or {})
    rm_tel = contact_rm.get("telephone", "04 98 09 30 59")
    rm_site = contact_rm.get("site_web", "https://runningmangames.fr")

    rw_tel = "04 94 47 94 64"
    rw_site = "https://www.retroworldfrance.com"

    if owner == "runningman":
        contact_line = f"Pour confirmer un horaire, merci de contacter Runningman : {rm_tel} ou {rm_site}."
    else:
        contact_line = f"Pour confirmer un horaire, merci de contacter Retroworld : {rw_tel} ou {rw_site}."

    return (
        "Je peux bien sûr vous aider à préparer votre demande, mais je ne peux pas confirmer ni bloquer un créneau via le chat.\n"
        + contact_line
        + "\n\n"
        "Pour que je vous guide au mieux, pouvez-vous me préciser : l’activité, le nombre de participants et le jour souhaité ?"
    )


def maybe_prefix_common_greeting(reply: str, is_first_turn: bool) -> str:
    if not is_first_turn:
        return reply

    prefix = (
        "Bienvenue chez Runningman et Retroworld, comment puis-je vous aider ?\n"
        "\n"
        "Pour vous orienter rapidement :\n"
        "- Game Zone + salle VR : Runningman\n"
        "- Le reste (Escape Game VR, quiz, salle enfant, anniversaires, fidélité) : Retroworld\n"
        "\n"
    )
    return prefix + (reply or "")


# =========================================================
# Core pipeline
# =========================================================


def process_chat(brand_entry: str, user_text: str, conversation_id: str) -> Tuple[str, Dict[str, Any], str]:
    brand_entry = (brand_entry or "auto").lower().strip()
    user_text = user_text or ""

    # owner/topic
    if brand_entry in ("retroworld", "runningman"):
        owner = brand_entry
    else:
        owner = detect_owner_from_text(user_text)

    kb_rw = load_kb("retroworld")
    kb_rm = load_kb("runningman")

    # Pare-chocs règle d'âge (Runningman) : réponse directe et sûre
    if owner == "runningman" and _is_age_intent(user_text):
        ident = kb_rm.get("identite", {}) or {}
        contact = (ident.get("contact", {}) or {})
        reply = (
            "Runningman est accessible dès 7 ans.\n"
            "Pour les moins de 12 ans, un adulte accompagnateur est requis.\n"
            "À partir de 12 ans, l’adulte n’est plus obligatoire.\n"
            f"Pour réserver/valider un horaire : {contact.get('site_web','https://runningmangames.fr')} (ou {contact.get('telephone','04 98 09 30 59')})."
        )
        return reply, {"mode": "rule_based_age"}, owner

    # System prompt commun
    system = (
        "Vous êtes l’assistant COMMUN de Runningman Game Zone et de Retroworld France (même bâtiment à Draguignan).\n"
        "Votre mission est d’aider la clientèle sans créer de problème : vous êtes prudent, fiable et orienté solution.\n\n"
        "RÈGLES DE SÉCURITÉ (ABSOLUES) :\n"
        "- Le chat ne confirme JAMAIS un créneau et ne dit jamais qu’un horaire est 'disponible'.\n"
        "- Le chat ne bloque jamais une réservation, ne dit jamais 'réservation confirmée', ne dit jamais 'nous vous attendons'.\n"
        "- Vous pouvez préparer une demande ou un devis, mais la validation finale se fait par l’équipe (téléphone/site).\n"
        "- Si vous n’êtes pas sûr (info absente), vous le dites et vous proposez le contact humain.\n\n"
        "RÈGLE DE RÉPARTITION :\n"
        "- Runningman : Game Zone / Action Game + Salle VR (jeux VR classiques).\n"
        "- Retroworld : le reste (Escape Game VR, quiz interactifs, salle enfant, anniversaires/événements, fidélité).\n"
        "- Si c’est ambigu, posez 1 seule question de clarification (Runningman ou Retroworld).\n\n"
        "LIENS :\n"
        "- Ne donnez un lien que si l’utilisateur le demande explicitement (ex: 'donnez-moi le lien').\n"
        "- Sinon, donnez plutôt le téléphone/site en canal de confirmation.\n\n"
        "STYLE : vouvoiement, clair, 3 à 10 lignes. Listes à puces si utile.\n\n"
        + common_orientation_block()
        + "\n"
        + retroworld_facts_block()
        + "\n"
        + runningman_facts_block(kb_rm)
    )

    # Ajout KB (résumée)
    kb_lines: List[str] = []
    kb_lines.extend(kb_snippets_retroworld(kb_rw))
    kb_lines.extend(kb_snippets_runningman(kb_rm))

    if kb_lines:
        system += "\nConnaissances (KB résumée) :\n"
        for line in kb_lines[:160]:
            system += f"- {line}\n"

    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

    # server-side memory
    if conversation_id:
        hist = load_conversation_messages(conversation_id)
        messages.extend(prune_messages_for_prompt(hist, max_pairs=12))

    # user message
    messages.append({"role": "user", "content": user_text})

    # soft hints
    hints: List[str] = []
    if _is_just_question_mark(user_text):
        hints.append(
            "L'utilisateur a envoyé seulement '?'. Demandez ce qu'il souhaite: Game Zone/VR (Runningman) ou Escape VR/quiz/salle enfant/fidélité (Retroworld)."
        )

    if _is_reservation_intent(user_text):
        hints.append(
            "Réservation/Créneau: ne jamais confirmer ni dire 'disponible'. Collecter activité + nombre + jour souhaité, puis orienter vers téléphone/site pour validation."
        )

    if hints:
        messages.append({"role": "system", "content": "GUIDE DE RÉPONSE:\n- " + "\n- ".join(hints)})

    reply, usage = call_openai_chat(messages)

    # pare-chocs anti-confirmation
    reply = sanitize_booking_reply(reply, user_text=user_text, owner=owner)

    return reply, usage, owner


# =========================================================
# Routes
# =========================================================


@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": SERVICE_NAME, "status": "ok", "time_utc": _utc_iso()}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# --- FAQ (fichiers static) ---
@app.route("/faq/retroworld", methods=["GET"])
def faq_retroworld():
    path = os.path.join(app.static_folder, "faq_retroworld.json")
    data = _safe_read_json(path, {"items": []})
    if not isinstance(data, dict):
        data = {"items": []}
    return jsonify(data), 200


@app.route("/faq/runningman", methods=["GET"])
def faq_runningman():
    path = os.path.join(app.static_folder, "faq_runningman.json")
    data = _safe_read_json(path, {"items": []})
    if not isinstance(data, dict):
        data = {"items": []}
    return jsonify(data), 200


# --- Compat route: /chat (owner auto) ---
@app.route("/chat", methods=["POST"])
def chat_auto():
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    user_text = _payload_text(payload).strip()
    if not user_text:
        return jsonify({"error": "missing_message"}), 400

    metadata = _payload_metadata(payload)
    conversation_id, user_id = _get_or_create_conversation_id(payload)

    # check first turn BEFORE generating (avoid greeting duplicates)
    is_first_turn = len(load_conversation_messages(conversation_id)) == 0

    reply, usage, owner = process_chat("auto", user_text, conversation_id)
    reply = maybe_prefix_common_greeting(reply, is_first_turn=is_first_turn)

    append_conversation_turn(
        conversation_id=conversation_id,
        brand_effective=owner,
        user_text=user_text,
        assistant_reply=reply,
        extra={
            "metadata": metadata,
            "brand_effective": owner,
            "openai_usage": usage,
            "user_id": user_id,
        },
    )

    resp = make_response(
        jsonify(
            {
                "reply": reply,
                "answer": reply,
                "brand_effective": owner,
                "brand_entry": "auto",
                "conversation_id": conversation_id,
            }
        ),
        200,
    )
    resp.set_cookie(CONV_COOKIE_NAME, conversation_id, max_age=60 * 60 * 24 * 30, samesite="Lax")
    return resp


# --- Owner route: /chat/<brand> (compat) ---
@app.route("/chat/<brand>", methods=["POST"])
def chat_brand(brand: str):
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    user_text = _payload_text(payload).strip()
    if not user_text:
        return jsonify({"error": "missing_message"}), 400

    metadata = _payload_metadata(payload)
    conversation_id, user_id = _get_or_create_conversation_id(payload)

    is_first_turn = len(load_conversation_messages(conversation_id)) == 0

    reply, usage, owner = process_chat(brand, user_text, conversation_id)
    reply = maybe_prefix_common_greeting(reply, is_first_turn=is_first_turn)

    append_conversation_turn(
        conversation_id=conversation_id,
        brand_effective=owner,
        user_text=user_text,
        assistant_reply=reply,
        extra={
            "metadata": metadata,
            "brand_entry": brand,
            "brand_effective": owner,
            "openai_usage": usage,
            "user_id": user_id,
        },
    )

    resp = make_response(
        jsonify(
            {
                "reply": reply,
                "answer": reply,
                "brand_effective": owner,
                "brand_entry": brand,
                "conversation_id": conversation_id,
            }
        ),
        200,
    )
    resp.set_cookie(CONV_COOKIE_NAME, conversation_id, max_age=60 * 60 * 24 * 30, samesite="Lax")
    return resp


# --- User history (option token) ---
@app.route("/user/api/history", methods=["GET"])
def user_api_history():
    if USER_HISTORY_TOKEN and not _require_user_token(request):
        return jsonify({"error": "forbidden"}), 403

    conversation_id = (request.args.get("conversation_id") or "").strip()
    user_id = (request.args.get("user_id") or "").strip()

    if not conversation_id and user_id:
        conversation_id = get_user_conversation(user_id) or ""

    if not conversation_id:
        conversation_id = (request.cookies.get(CONV_COOKIE_NAME) or "").strip()

    if not conversation_id:
        return jsonify({"error": "missing_conversation_id"}), 400

    obj = load_conversation_obj(conversation_id)
    return jsonify({"conversation_id": conversation_id, "conversation": obj}), 200


# --- KB endpoints ---
@app.route("/kb/<brand>", methods=["GET"])
def kb_get(brand: str):
    kb = load_kb(brand)
    return jsonify(kb), 200


@app.route("/kb/<brand>", methods=["POST"])
def kb_upsert(brand: str):
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    kb = load_kb(brand)

    # Compat: accept "items" field, but keep full KB dict otherwise
    new_items = payload.get("items")
    if isinstance(new_items, list):
        kb["items"] = new_items

    ok = save_kb(brand, kb)
    return jsonify({"ok": ok, "brand": brand}), 200


# --- Admin API ---
@app.route("/admin/api/conversations", methods=["GET"])
def admin_api_conversations():
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    conv_ids = list_conversations()
    items: List[Dict[str, Any]] = []
    for cid in conv_ids:
        obj = load_conversation_obj(cid)
        msgs = obj.get("messages") or []
        brand_eff = str(obj.get("brand_last") or "unknown")

        preview = ""
        timestamp = None
        if isinstance(msgs, list) and msgs:
            lastm = msgs[-1] if isinstance(msgs[-1], dict) else {}
            timestamp = lastm.get("ts")

            last_user = ""
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "user":
                    last_user = str(m.get("content") or "")
                    break
            preview = (last_user or str(lastm.get("content") or "")).strip()
            preview = re.sub(r"\s+", " ", preview)
            if len(preview) > 120:
                preview = preview[:117] + "..."

        items.append(
            {
                "id": cid,
                "brand": brand_eff,
                "preview": preview,
                "timestamp": timestamp,
                "user_id": obj.get("user_id"),
            }
        )

    return jsonify(items), 200


@app.route("/admin/api/conversation/<conversation_id>", methods=["GET"])
def admin_api_conversation_detail(conversation_id: str):
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    obj = load_conversation_obj(conversation_id)
    msgs: List[Dict[str, Any]] = obj.get("messages") or []
    simple = [{"role": m.get("role"), "content": m.get("content"), "ts": m.get("ts")} for m in msgs if isinstance(m, dict)]
    return jsonify({"conversation_id": conversation_id, "messages": simple, "conversation": obj, "brand_final": obj.get("brand_last")}), 200


@app.route("/admin/api/diag", methods=["GET"])
def admin_api_diag():
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    def kb_diag(brand: str) -> Dict[str, Any]:
        fname = f"kb_{brand}.json"
        path = os.path.join(KB_DIR, fname)
        exists = os.path.exists(path)
        data = _safe_read_json(path, None) if exists else None
        load_ok = isinstance(data, dict)
        return {"exists": exists, "file": fname, "path": path, "load_ok": load_ok}

    try:
        conv_files = len([f for f in os.listdir(CONV_DIR) if f.endswith(".json")])
    except Exception:
        conv_files = 0

    return jsonify(
        {
            "service": SERVICE_NAME,
            "port": PORT,
            "has_admin_token": bool(ADMIN_DASHBOARD_TOKEN),
            "has_user_history_token": bool(USER_HISTORY_TOKEN),
            "paths": {"kb_dir": KB_DIR, "data_dir": DATA_DIR, "conv_dir": CONV_DIR},
            "kb": {"retroworld": kb_diag("retroworld"), "runningman": kb_diag("runningman")},
            "logs": {"conversations_files_count": conv_files},
            "openai": {
                "client_ready": bool(_openai_client),
                "key_present": bool(OPENAI_API_KEY),
                "model": OPENAI_MODEL,
                "reasoning_effort": OPENAI_REASONING_EFFORT,
                "init_error": _openai_init_error,
            },
        }
    ), 200


# --- Pages / static ---
@app.route("/admin", methods=["GET"])
def admin_page():
    return send_from_directory(app.static_folder, "admin.html")


@app.route("/admin/conversations", methods=["GET"])
def admin_conversations_page():
    return send_from_directory(app.static_folder, "admin.html")


@app.route("/<path:path>", methods=["GET"])
def static_proxy(path: str):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    logger.info("Starting %s on port %s", SERVICE_NAME, PORT)
    app.run(host="0.0.0.0", port=PORT)
