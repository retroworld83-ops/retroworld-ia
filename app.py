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

# Modèle par défaut (override via env)
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-5.2").strip()

# GPT-5: temperature uniquement si reasoning.effort = "none"
OPENAI_REASONING_EFFORT = (os.getenv("OPENAI_REASONING_EFFORT") or "none").strip().lower()
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS") or os.getenv("OPENAI_MAX_TOKENS") or "900")

# Admin token (legacy alias supported)
ADMIN_DASHBOARD_TOKEN = (os.getenv("ADMIN_DASHBOARD_TOKEN") or os.getenv("ADMIN_API_TOKEN") or "").strip()
USER_HISTORY_TOKEN = (os.getenv("USER_HISTORY_TOKEN") or "").strip()

LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(SERVICE_NAME)

app = Flask(__name__, static_folder="static")

# -------------------------
# CORS: utilise ALLOWED_ORIGINS (Render)
# -------------------------
allowed = (os.getenv("ALLOWED_ORIGINS") or "").strip()
if not allowed:
    CORS(app, supports_credentials=True)
else:
    origins = [o.strip() for o in allowed.split(",") if o.strip()]
    if "*" in origins:
        CORS(app, supports_credentials=True)
    else:
        CORS(app, resources={r"/*": {"origins": origins}}, supports_credentials=True)

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
# Helpers
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


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


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
# Intents + routing
# =========================================================


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
    ]
    if any(k in t for k in keys):
        return True
    if re.search(r"\b\d{1,2}h(\d{2})?\b", t):
        return True
    return False


def _is_just_question_mark(text: str) -> bool:
    t = (text or "").strip()
    return t in ["?", "??", "???"]


def detect_owner_from_text(text: str) -> str:
    """
    RÉPARTITION OFFICIELLE (mise à jour) :
    - Retroworld : VR, Escape VR, Quiz, Goûter à volonté + gâteau, Anniversaire, Fidélité/points
    - Runningman : Game Zone, Escape game NON-VR, Salle enfant
    """
    t = _norm(text)

    # Marques explicites (prioritaires)
    if "retroworld" in t:
        return "retroworld"
    if "runningman" in t or "running man" in t:
        return "runningman"

    # Escape VR => Retroworld
    if "escape" in t and "vr" in t:
        return "retroworld"

    # VR (VR classique aussi) => Retroworld
    if "vr" in t or "casque" in t or "meta quest" in t or "quest" in t:
        return "retroworld"

    # Quiz => Retroworld
    if "quiz" in t:
        return "retroworld"

    # Goûter / gâteau / anniversaire => Retroworld
    if "goûter" in t or "gouter" in t or "gâteau" in t or "gateau" in t or "anniversaire" in t:
        return "retroworld"

    # Fidélité / points => Retroworld
    if "fidélité" in t or "fidelite" in t or "points" in t or "qr" in t:
        return "retroworld"

    # Runningman: salle enfant
    if "salle enfant" in t or "salle enfants" in t or ("salle" in t and "enfant" in t):
        return "runningman"

    # Runningman: game zone / action game
    if "game zone" in t or "gamezone" in t or "action game" in t:
        return "runningman"

    # Escape game NON-VR => Runningman
    if "escape" in t and "vr" not in t:
        return "runningman"

    # Ambigu => on demande, mais on stocke "auto"
    return "auto"


# =========================================================
# Facts blocks (fiables)
# =========================================================


def common_orientation_block() -> str:
    return (
        "RÉPARTITION (OFFICIELLE) :\n"
        "- Retroworld : VR, Escape VR, quiz, goûter à volonté + gâteau (anniversaires), fidélité/points.\n"
        "- Runningman : Game Zone, Escape game non-VR, Salle enfant.\n"
        "- Même bâtiment : 815 avenue Pierre Brossolette, 83300 Draguignan.\n"
    )


def retroworld_facts_block() -> str:
    return (
        "VÉRITÉS OFFICIELLES RETROWORLD (à respecter strictement) :\n"
        "- Horaires (indicatifs) : mardi à dimanche, 11h à 22h\n"
        "- Adresse : 815 avenue Pierre Brossolette, 83300 Draguignan\n"
        "- Contact : 04 94 47 94 64 / contact@retroworldfrance.com\n"
        "- Site : https://www.retroworldfrance.com\n"
        "- Activités : VR, Escape VR, quiz, goûter à volonté + gâteau (anniversaires), fidélité\n"
        "- RÈGLE CRÉNEAUX : le chat ne confirme ni ne bloque jamais un créneau.\n"
        "  Il peut préparer une demande/devis, mais la validation se fait par l'équipe.\n"
        "IMPORTANT : ne donner un lien que si l'utilisateur le demande explicitement.\n"
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
        "- Activités : Game Zone, Escape game non-VR, Salle enfant\n"
        f"- Âge minimum (selon activité) : {age_min} ans.\n"
        f"- Moins de 12 ans : {acc.get('moins_de_12_ans', 'Un adulte accompagnateur est requis.')}\n"
        f"- À partir de 12 ans : {acc.get('a_partir_de_12_ans', 'Il n’est plus nécessaire d’avoir un adulte accompagnateur.')}\n"
        "- RÈGLE CRÉNEAUX : le chat ne confirme ni ne bloque jamais un créneau.\n"
        "  Pour réserver/valider un horaire : site officiel ou téléphone.\n"
        f"- Tarifs Action Game (si demandé) :\n"
        f"  - Enfant (-12 ans) : {enfant.get('prix', '?')} {enfant.get('unite', '')}\n"
        f"  - Adulte accompagnateur : {adulte.get('prix', '?')} {adulte.get('unite', '')}\n"
        + (f"- Capacité : {cap_msg}\n" if cap_msg else "")
        + f"- Adresse : {loc.get('adresse_complete', '815 avenue Pierre Brossolette, 83300 Draguignan, France')}\n"
        + f"- Contact : {contact.get('telephone', '04 98 09 30 59')} / {contact.get('site_web', 'https://runningmangames.fr')}\n"
    )


def kb_snippets_retroworld(kb: Dict[str, Any], limit_lines: int = 80) -> List[str]:
    out: List[str] = []
    prompt = kb.get("prompt", {}) or {}
    if isinstance(prompt, dict):
        for k in ["reservation_non_confirmee", "gestion_liens_reservation", "etape_5_devis", "fidelite", "anti_erreurs"]:
            v = prompt.get(k)
            if isinstance(v, str) and v.strip():
                out.append(f"{k}: {v.strip()}")
    inst = kb.get("instructions_generales")
    if isinstance(inst, list):
        for it in inst:
            s = str(it).strip()
            if s:
                out.append(f"Règle: {s}")
    return out[:limit_lines]


def kb_snippets_runningman(kb: Dict[str, Any], limit_lines: int = 60) -> List[str]:
    out: List[str] = []
    for k in ["instructions_generales", "anti_erreurs"]:
        arr = kb.get(k)
        if isinstance(arr, list):
            for it in arr:
                s = str(it).strip()
                if s:
                    out.append(s)
    return out[:limit_lines]


# =========================================================
# OpenAI call (Responses API)
# =========================================================


def call_openai_chat(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    if not _openai_client:
        return (
            "Le service IA est indisponible pour le moment. Vous pouvez contacter Retroworld au 04 94 47 94 64 ou Runningman au 04 98 09 30 59.",
            {"error": "openai_not_ready"},
        )

    kwargs: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "input": messages,
        "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
        "store": False,
    }

    if OPENAI_REASONING_EFFORT:
        kwargs["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}

    if OPENAI_REASONING_EFFORT == "none":
        kwargs["temperature"] = OPENAI_TEMPERATURE

    resp = _openai_client.responses.create(**kwargs)
    text = (getattr(resp, "output_text", "") or "").strip()

    usage: Dict[str, Any] = {}
    try:
        usage_obj = getattr(resp, "usage", None)
        if usage_obj is not None:
            if hasattr(usage_obj, "model_dump"):
                usage = usage_obj.model_dump()  # type: ignore
            elif isinstance(usage_obj, dict):
                usage = usage_obj
    except Exception:
        usage = {}

    return text, usage


# =========================================================
# Conversation storage
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

    if (user_text or "").strip():
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
    clipped = [m for m in messages if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content") is not None]
    clipped = clipped[-(max_pairs * 2):]
    return [{"role": str(m.get("role")), "content": str(m.get("content") or "")} for m in clipped]


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
# Payload parsing
# =========================================================


def _payload_text(payload: Dict[str, Any]) -> str:
    msg = payload.get("message")
    if msg is not None:
        return str(msg)
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
# Greeting + anti-creneaux
# =========================================================

START_SIGNAL = "__start__"

_FORBIDDEN_BOOKING_RE = [
    r"\bje (vous )?(confirme|confirme la|confirme votre)\b",
    r"\b(réservation|reservation) (est )?(confirmée|confirmee|validée|validee)\b",
    r"\bje (bloque|bloque donc|bloque votre|réserve|reserve)\b",
    r"\b(nous|on) vous attend(ons)?\b",
    r"\b(c['’]?est|c est) (disponible|ok|bon)\b",
    r"\bdisponible pour\b",
    r"\bcréneau confirmé\b",
]


def common_greeting_text() -> str:
    return (
        "Bienvenue chez Runningman et Retroworld, comment puis-je vous aider ?\n\n"
        "Pour vous orienter rapidement :\n"
        "- Retroworld : VR, Escape VR, quiz, goûter à volonté + gâteau\n"
        "- Runningman : Game Zone, Escape game non-VR, Salle enfant"
    )


def sanitize_booking_reply(reply: str, user_text: str, owner: str) -> str:
    if not reply:
        return reply
    if not _is_reservation_intent(user_text):
        return reply

    low = _norm(reply)
    if not any(re.search(p, low) for p in _FORBIDDEN_BOOKING_RE):
        return reply

    kb_rm = load_kb("runningman")
    ident_rm = kb_rm.get("identite", {}) or {}
    contact_rm = (ident_rm.get("contact", {}) or {})
    rm_tel = contact_rm.get("telephone", "04 98 09 30 59")
    rm_site = contact_rm.get("site_web", "https://runningmangames.fr")

    rw_tel = "04 94 47 94 64"
    rw_site = "https://www.retroworldfrance.com"

    if owner == "runningman":
        contact_line = f"Pour confirmer un horaire, merci de contacter Runningman : {rm_tel} ou {rm_site}."
    elif owner == "retroworld":
        contact_line = f"Pour confirmer un horaire, merci de contacter Retroworld : {rw_tel} ou {rw_site}."
    else:
        contact_line = (
            f"Pour confirmer un horaire : Retroworld ({rw_tel}) ou Runningman ({rm_tel})."
        )

    return (
        "Je peux vous aider à préparer votre demande, mais je ne peux pas confirmer ni bloquer un créneau via le chat.\n"
        + contact_line
        + "\n\n"
        "Pouvez-vous préciser : l’activité, le nombre de participants et le jour souhaité ?"
    )


def maybe_prefix_common_greeting(reply: str, is_first_turn: bool) -> str:
    if not is_first_turn:
        return reply
    return common_greeting_text() + "\n\n" + (reply or "")


# =========================================================
# Core pipeline
# =========================================================


def process_chat(brand_entry: str, user_text: str, conversation_id: str) -> Tuple[str, Dict[str, Any], str]:
    brand_entry = (brand_entry or "auto").lower().strip()
    user_text = user_text or ""

    owner = brand_entry if brand_entry in ("retroworld", "runningman") else detect_owner_from_text(user_text)

    kb_rw = load_kb("retroworld")
    kb_rm = load_kb("runningman")

    system = (
        "Vous êtes l’assistant COMMUN de Runningman et de Retroworld (même bâtiment à Draguignan).\n"
        "Votre mission est d’aider la clientèle sans créer de problème : prudence, clarté, orientation.\n\n"
        "RÈGLES ABSOLUES :\n"
        "- Ne jamais confirmer ni bloquer un créneau.\n"
        "- Ne jamais dire qu’un horaire est 'disponible'.\n"
        "- Vous pouvez préparer une demande ou un devis, mais la validation se fait par l’équipe.\n"
        "- Si information incertaine, poser 1 question simple ou proposer le contact humain.\n\n"
        "RÉPARTITION OFFICIELLE :\n"
        "- Retroworld : VR, Escape VR, quiz, goûter à volonté + gâteau (anniversaires), fidélité/points.\n"
        "- Runningman : Game Zone, Escape game non-VR, Salle enfant.\n\n"
        "LIENS :\n"
        "- Ne donner un lien que si l’utilisateur le demande explicitement.\n\n"
        "STYLE : vouvoiement, 3 à 10 lignes, liste à puces si utile.\n\n"
        + common_orientation_block()
        + "\n"
        + retroworld_facts_block()
        + "\n"
        + runningman_facts_block(kb_rm)
    )

    kb_lines: List[str] = []
    kb_lines.extend(kb_snippets_retroworld(kb_rw))
    kb_lines.extend(kb_snippets_runningman(kb_rm))
    if kb_lines:
        system += "\nConnaissances (KB résumée) :\n"
        for line in kb_lines[:140]:
            system += f"- {line}\n"

    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

    if conversation_id:
        hist = load_conversation_messages(conversation_id)
        messages.extend(prune_messages_for_prompt(hist, max_pairs=12))

    messages.append({"role": "user", "content": user_text})

    hints: List[str] = []
    if owner == "auto":
        hints.append("C'est ambigu: demandez si c'est Retroworld (VR/Escape VR/quiz/goûter) ou Runningman (Game Zone/Escape non-VR/Salle enfant).")
    if _is_just_question_mark(user_text):
        hints.append("L'utilisateur a envoyé seulement '?'. Demandez ce qu'il souhaite (Retroworld ou Runningman).")
    if _is_reservation_intent(user_text):
        hints.append("Réservation: ne jamais confirmer, collecter activité + nombre + jour souhaité, puis orienter vers téléphone/site.")
    if hints:
        messages.append({"role": "system", "content": "GUIDE:\n- " + "\n- ".join(hints)})

    reply, usage = call_openai_chat(messages)
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


@app.route("/chat", methods=["POST"])
def chat_auto():
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    user_text = _payload_text(payload).strip()
    metadata = _payload_metadata(payload)
    conversation_id, user_id = _get_or_create_conversation_id(payload)

    # Start handshake (évite doublon de "Bienvenue")
    if user_text == START_SIGNAL:
        greeting = common_greeting_text()
        append_conversation_turn(
            conversation_id=conversation_id,
            brand_effective="auto",
            user_text="",
            assistant_reply=greeting,
            extra={"metadata": metadata, "brand_effective": "auto", "openai_usage": {"mode": "start"}, "user_id": user_id},
        )
        resp = make_response(
            jsonify({"reply": greeting, "answer": greeting, "brand_effective": "auto", "brand_entry": "auto", "conversation_id": conversation_id}),
            200,
        )
        resp.set_cookie(CONV_COOKIE_NAME, conversation_id, max_age=60 * 60 * 24 * 30, samesite="Lax")
        return resp

    if not user_text:
        return jsonify({"error": "missing_message"}), 400

    is_first_turn = len(load_conversation_messages(conversation_id)) == 0

    reply, usage, owner = process_chat("auto", user_text, conversation_id)
    reply = maybe_prefix_common_greeting(reply, is_first_turn=is_first_turn)

    append_conversation_turn(
        conversation_id=conversation_id,
        brand_effective=owner,
        user_text=user_text,
        assistant_reply=reply,
        extra={"metadata": metadata, "brand_effective": owner, "openai_usage": usage, "user_id": user_id},
    )

    resp = make_response(
        jsonify({"reply": reply, "answer": reply, "brand_effective": owner, "brand_entry": "auto", "conversation_id": conversation_id}),
        200,
    )
    resp.set_cookie(CONV_COOKIE_NAME, conversation_id, max_age=60 * 60 * 24 * 30, samesite="Lax")
    return resp


@app.route("/chat/<brand>", methods=["POST"])
def chat_brand(brand: str):
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    user_text = _payload_text(payload).strip()
    metadata = _payload_metadata(payload)
    conversation_id, user_id = _get_or_create_conversation_id(payload)

    if user_text == START_SIGNAL:
        greeting = common_greeting_text()
        append_conversation_turn(
            conversation_id=conversation_id,
            brand_effective=brand,
            user_text="",
            assistant_reply=greeting,
            extra={"metadata": metadata, "brand_entry": brand, "brand_effective": brand, "openai_usage": {"mode": "start"}, "user_id": user_id},
        )
        resp = make_response(
            jsonify({"reply": greeting, "answer": greeting, "brand_effective": brand, "brand_entry": brand, "conversation_id": conversation_id}),
            200,
        )
        resp.set_cookie(CONV_COOKIE_NAME, conversation_id, max_age=60 * 60 * 24 * 30, samesite="Lax")
        return resp

    if not user_text:
        return jsonify({"error": "missing_message"}), 400

    is_first_turn = len(load_conversation_messages(conversation_id)) == 0

    reply, usage, owner = process_chat(brand, user_text, conversation_id)
    reply = maybe_prefix_common_greeting(reply, is_first_turn=is_first_turn)

    append_conversation_turn(
        conversation_id=conversation_id,
        brand_effective=owner,
        user_text=user_text,
        assistant_reply=reply,
        extra={"metadata": metadata, "brand_entry": brand, "brand_effective": owner, "openai_usage": usage, "user_id": user_id},
    )

    resp = make_response(
        jsonify({"reply": reply, "answer": reply, "brand_effective": owner, "brand_entry": brand, "conversation_id": conversation_id}),
        200,
    )
    resp.set_cookie(CONV_COOKIE_NAME, conversation_id, max_age=60 * 60 * 24 * 30, samesite="Lax")
    return resp


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

    new_items = payload.get("items")
    if isinstance(new_items, list):
        kb["items"] = new_items

    ok = save_kb(brand, kb)
    return jsonify({"ok": ok, "brand": brand}), 200


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

        items.append({"id": cid, "brand": brand_eff, "preview": preview, "timestamp": timestamp, "user_id": obj.get("user_id")})

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
