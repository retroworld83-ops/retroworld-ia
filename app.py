import os
import json
import time
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from flask import Flask, request, jsonify, send_from_directory, redirect, make_response

# ------------------------------------------------------------
# App + Paths
# ------------------------------------------------------------
APP_NAME = "retroworld-ia"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CONV_DIR = DATA_DIR / "conversations"
STATIC_DIR = BASE_DIR / "static"
CONFIG_DIR = BASE_DIR / "config"

CONV_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Env
# ------------------------------------------------------------
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
BRAND_ID_DEFAULT = os.getenv("BRAND_ID", "retroworld").strip().lower()

DEBUG_LOGS = os.getenv("DEBUG_LOGS", "false").strip().lower() in ("1", "true", "yes", "on")

# Admin security
ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN", "").strip()
ADMIN_DASHBOARD_TOKEN = os.getenv("ADMIN_DASHBOARD_TOKEN", "").strip()
ADMIN_EMAILS = os.getenv("ADMIN_EMAILS", "").strip()

# CORS
ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# FAQ enabled brands (public faq availability)
FAQ_ENABLED_BRANDS = [b.strip().lower() for b in os.getenv("FAQ_ENABLED_BRANDS", "retroworld,runningman").split(",") if b.strip()]

# Public brands shown in widget /brands.json
PUBLIC_BRANDS = [b.strip().lower() for b in os.getenv("PUBLIC_BRANDS", "retroworld,runningman").split(",") if b.strip()]

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2").strip()
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "none").strip()
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", os.getenv("OPENAI_MAX_OUTPUT_TOKENS".lower(), "900")))

# ------------------------------------------------------------
# Brands config
# ------------------------------------------------------------
DEFAULT_BRANDS = {
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
        "contact_phone": "",
        "contact_email": "",
        "website": "",
        "domains": [],
    },
}


def log(*args):
    if DEBUG_LOGS:
        print("[DBG]", *args, flush=True)


def load_yaml_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        log("YAML load failed:", path, e)
        return None


brands_cfg = load_yaml_if_exists(CONFIG_DIR / "brands.yaml") or {}
BRANDS: Dict[str, Dict[str, Any]] = DEFAULT_BRANDS.copy()
if isinstance(brands_cfg, dict) and "brands" in brands_cfg and isinstance(brands_cfg["brands"], dict):
    for bid, cfg in brands_cfg["brands"].items():
        if not isinstance(cfg, dict):
            continue
        bid2 = str(bid).strip().lower()
        base = BRANDS.get(bid2, {"name": bid2, "short": bid2})
        base.update(cfg)
        BRANDS[bid2] = base

for bid in list(BRANDS.keys()):
    BRANDS[bid]["id"] = bid

# ------------------------------------------------------------
# Knowledge Base handling
# ------------------------------------------------------------
KB_FILES = {
    "retroworld": BASE_DIR / "kb_retroworld.json",
    "runningman": BASE_DIR / "kb_runningman.json",
    "enigmaniac": BASE_DIR / "kb_enigmaniac.json",
}

# For widget compatibility: static FAQ JSON files expected
STATIC_FAQ_FILES = {
    "retroworld": STATIC_DIR / "faq_retroworld.json",
    "runningman": STATIC_DIR / "faq_runningman.json",
    # legacy path for some widgets: /static/static/faq_runningman.json
    "runningman_legacy": STATIC_DIR / "static" / "faq_runningman.json",
}

(STATIC_DIR / "static").mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log("JSON read failed:", path, e)
        return None


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_kb_exists():
    # If missing, create minimal KBs.
    for bid, p in KB_FILES.items():
        if p.exists():
            continue
        write_json(
            p,
            {
                "brand": bid,
                "updated": now_iso(),
                "items": [
                    {
                        "question": "FAQ non publiée pour le moment",
                        "answer": "Cette FAQ n'est pas encore disponible. Veuillez contacter l'équipe pour une réponse fiable.",
                        "tags": ["faq"],
                    }
                ],
            },
        )

    # Also ensure static FAQ files exist for enabled brands, with widget-friendly schema.
    for bid in ["retroworld", "runningman"]:
        kb = read_json(KB_FILES[bid]) or {"brand": bid, "items": [], "updated": now_iso()}
        static_payload = {"brand": bid, "updated": kb.get("updated", now_iso()), "items": kb.get("items", [])}
        write_json(STATIC_FAQ_FILES[bid], static_payload)
        if bid == "runningman":
            write_json(STATIC_FAQ_FILES["runningman_legacy"], static_payload)


ensure_kb_exists()

# ------------------------------------------------------------
# Flask App
# ------------------------------------------------------------
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

# ------------------------------------------------------------
# Helpers: security / CORS / brand selection
# ------------------------------------------------------------
RESERVATION_FORBIDDEN_PATTERNS = [
    r"\b(c['’]?est réservé|réservé|confirmé|confirmée|je vous bloque|on vous bloque|bloqué|bloquée)\b",
]

RESERVATION_INTENT_PATTERNS = [
    r"\b(réserver|reservation|réservation|dispo|disponibilit|créneau|horaire|anniversaire|goûter|acompte)\b",
]


def normalize_brand(b: str) -> str:
    return (b or "").strip().lower()


def detect_brand_from_origin() -> Optional[str]:
    origin = (request.headers.get("Origin") or "").strip().lower()
    referer = (request.headers.get("Referer") or "").strip().lower()
    host = (request.host or "").strip().lower()

    candidates = [origin, referer, host]
    for cand in candidates:
        for bid, cfg in BRANDS.items():
            for d in cfg.get("domains", []) or []:
                if d and d.lower() in cand:
                    return bid
    return None


def detect_brand_from_text(text: str) -> Optional[str]:
    t = (text or "").lower()
    # Basic keyword heuristic
    if "runningman" in t:
        return "runningman"
    if "enigmaniac" in t or "enigma" in t:
        return "enigmaniac"
    if "retroworld" in t or "retro world" in t:
        return "retroworld"
    return None


def get_brand_id(payload: dict) -> str:
    # priority: payload -> header -> query -> text -> origin/ref -> env default
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


def corsify(resp):
    origin = (request.headers.get("Origin") or "").strip().rstrip("/")
    if ALLOWED_ORIGINS:
        if origin in ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
    else:
        # permissive if not set
        if origin:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
        else:
            resp.headers["Access-Control-Allow-Origin"] = "*"

    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Brand-Id"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


def require_admin_token() -> bool:
    token = ""
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    else:
        token = (request.args.get("token") or request.headers.get("X-Admin-Token") or "").strip()

    valid = set([t for t in [ADMIN_API_TOKEN, ADMIN_DASHBOARD_TOKEN] if t])
    if not valid:
        # If no token configured, deny by default (secure)
        return False
    return token in valid


# ------------------------------------------------------------
# KB Retrieval (simple, deterministic)
# ------------------------------------------------------------
def kb_items_for_brand(bid: str) -> List[dict]:
    kb = read_json(KB_FILES.get(bid, Path(""))) or {}
    items = kb.get("items") or []
    if not isinstance(items, list):
        return []
    cleaned = []
    for it in items:
        if not isinstance(it, dict):
            continue
        q = (it.get("question") or "").strip()
        a = (it.get("answer") or "").strip()
        tags = it.get("tags") or []
        if q and a:
            cleaned.append({"question": q, "answer": a, "tags": tags if isinstance(tags, list) else []})
    return cleaned


def search_kb(items: List[dict], query: str, limit: int = 6) -> List[dict]:
    q = (query or "").lower().strip()
    if not q:
        return []
    tokens = [t for t in re.split(r"\s+", q) if t]
    scored = []
    for it in items:
        text = (it["question"] + " " + it["answer"]).lower()
        score = 0
        for t in tokens:
            if t in text:
                score += 1
        if score > 0:
            scored.append((score, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:limit]]


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


# ------------------------------------------------------------
# OpenAI (Responses API)
# ------------------------------------------------------------
def openai_ready() -> bool:
    return bool(OPENAI_API_KEY)


def openai_answer(system: str, user: str, max_output_tokens: int) -> str:
    """
    Uses OpenAI Responses API via HTTPS.
    No external dependencies required.
    """
    import requests  # type: ignore

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
        "max_output_tokens": max_output_tokens,
    }

    # For GPT-5.2: temperature allowed only when reasoning_effort = none
    if OPENAI_REASONING_EFFORT:
        payload["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}
    if OPENAI_REASONING_EFFORT == "none":
        payload["temperature"] = OPENAI_TEMPERATURE

    r = requests.post("https://api.openai.com/v1/responses", headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    # extract output text
    out_texts = []
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out_texts.append(c.get("text", ""))
    return ("\n".join(out_texts)).strip()


# ------------------------------------------------------------
# Safety / policy layer
# ------------------------------------------------------------
GLOBAL_RULES = """Vous êtes un assistant multi-établissements (Retroworld / Runningman / Enigmaniac) dans le même bâtiment.
Règles strictes:
- Ne promettez jamais une réservation (pas d'accès au planning). Ne dites jamais: "réservé", "confirmé", "je vous bloque", "c'est réservé".
- Si l'utilisateur demande une disponibilité / créneau / réservation, répondez: "Je n’ai pas accès au planning en temps réel, c’est à confirmer par l’équipe." Puis indiquez le bon contact.
- Ne mélangez pas les informations entre établissements. Si la question concerne un autre établissement, précisez-le clairement et donnez uniquement des infos fiables.
- Si une information manque, dites-le et proposez de contacter l'équipe.
- Style: professionnel, clair, concis. Pas de tutoiement.
"""

def build_system_prompt(bid: str, cross_bids: List[str]) -> str:
    cfg = BRANDS.get(bid, {})
    who = cfg.get("name", bid)
    contact = format_contact(bid)
    cross_txt = ""
    if cross_bids:
        cross_txt = "Vous pouvez aussi utiliser ces infos (si la question les mentionne explicitement) :\n"
        for cb in cross_bids:
            cross_txt += f"- {BRANDS.get(cb, {}).get('name', cb)} : {format_contact(cb)}\n"
    return f"""{GLOBAL_RULES}

Établissement principal: {who}
Contact principal: {contact}

{cross_txt}
"""


def enforce_no_reservation_promises(text: str) -> Tuple[str, bool]:
    lowered = (text or "").lower()
    for pat in RESERVATION_FORBIDDEN_PATTERNS:
        if re.search(pat, lowered, flags=re.IGNORECASE):
            # Replace dangerous wording softly
            safe = re.sub(pat, "à confirmer par l’équipe (je n’ai pas accès au planning en direct)", text, flags=re.IGNORECASE)
            return safe, True
    return text, False


def needs_reservation_disclaimer(user_msg: str) -> bool:
    t = (user_msg or "").lower()
    for pat in RESERVATION_INTENT_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    return False


def add_disclaimer_if_needed(answer: str, bid: str, user_msg: str) -> str:
    if not needs_reservation_disclaimer(user_msg):
        return answer
    disclaimer = "Je n’ai pas accès au planning en temps réel, c’est à confirmer par l’équipe. "
    contact = format_contact(bid)
    if contact:
        disclaimer += f"Contact: {contact}"
    if disclaimer.lower() not in (answer or "").lower():
        return answer.rstrip() + "\n\n" + disclaimer
    return answer


# ------------------------------------------------------------
# Conversations storage
# ------------------------------------------------------------
def conv_path(conv_id: str) -> Path:
    return CONV_DIR / f"{conv_id}.json"


def new_conv_id(prefix: str = "rw") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def load_conv(conv_id: str) -> dict:
    p = conv_path(conv_id)
    if not p.exists():
        return {"id": conv_id, "created": now_iso(), "messages": [], "meta": {}}
    return read_json(p) or {"id": conv_id, "created": now_iso(), "messages": [], "meta": {}}


def save_conv(conv: dict):
    p = conv_path(conv.get("id", "unknown"))
    write_json(p, conv)


def append_message(conv: dict, role: str, content: str, extra: Optional[dict] = None):
    conv.setdefault("messages", [])
    conv["messages"].append(
        {
            "ts": now_iso(),
            "role": role,
            "content": content,
            "extra": extra or {},
        }
    )


# ------------------------------------------------------------
# Admin flags / diagnostics
# ------------------------------------------------------------
FLAG_PATTERNS = {
    "devis": r"\b(devis|privatis|entreprise|ce\b|comit[ée]\s*d['’]?entreprise|team\s*building|groupe)\b",
    "reservation": r"\b(réserver|réservation|dispo|créneau|anniversaire|goûter|acompte)\b",
    "reclamation": r"\b(rembourse|annul|plainte|probl[eè]me|litige|panne)\b",
    "croise": r"\b(retroworld|runningman|enigmaniac)\b.*\b(retroworld|runningman|enigmaniac)\b",
    "promesse_resa": r"\b(réservé|confirmé|je vous bloque|c['’]?est réservé|bloqué)\b",
    "a_relire": r"\b(peut[- ]?être|probablement|je pense|à priori|il me semble)\b",
}


def compute_flags(conv: dict) -> List[str]:
    msgs = conv.get("messages") or []
    blob = "\n".join([(m.get("content") or "") for m in msgs]).lower()
    flags = []
    for name, pat in FLAG_PATTERNS.items():
        if re.search(pat, blob, flags=re.IGNORECASE | re.DOTALL):
            flags.append(name)
    # a_valider: composite
    if any(f in flags for f in ("devis", "reclamation")):
        flags.append("a_valider")
    return sorted(set(flags))


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.after_request
def after(resp):
    return corsify(resp)


@app.route("/", methods=["GET"])
def index():
    return redirect("/static/chat-widget.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "ok": True,
            "app": APP_NAME,
            "time": now_iso(),
            "openai_configured": openai_ready(),
            "brands": list(BRANDS.keys()),
            "faq_enabled_brands": FAQ_ENABLED_BRANDS,
            "public_brands": PUBLIC_BRANDS,
        }
    )


@app.route("/brands.json", methods=["GET"])
def brands_json():
    # Public list for widgets
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
    # Simple HTML page showing FAQ (optional)
    # Most users use the widget's FAQ tab (client-side)
    brand_id = normalize_brand(request.args.get("brand_id") or request.args.get("brand") or BRAND_ID_DEFAULT)
    if brand_id not in FAQ_ENABLED_BRANDS:
        return make_response("FAQ indisponible pour le moment.", 404)
    return redirect(f"/static/chat-widget.html?tab=faq&brand={brand_id}")


@app.route("/faq.json", methods=["GET"])
def faq_json():
    brand_id = normalize_brand(request.args.get("brand_id") or request.args.get("brand") or BRAND_ID_DEFAULT)

    if brand_id == "all":
        # Return all enabled FAQs
        payload = []
        for bid in FAQ_ENABLED_BRANDS:
            payload.append({"brand": bid, "items": kb_items_for_brand(bid)})
        return jsonify({"items": payload, "updated": now_iso()})

    if brand_id not in FAQ_ENABLED_BRANDS:
        return jsonify({"brand": brand_id, "updated": now_iso(), "items": []}), 404

    kb = read_json(KB_FILES.get(brand_id, Path(""))) or {"brand": brand_id, "items": []}
    # widget-friendly schema: items with question/answer
    return jsonify({"brand": brand_id, "updated": kb.get("updated", now_iso()), "items": kb.get("items", [])})


# Legacy aliases (some widgets hardcode these)
@app.route("/faq_retroworld.json", methods=["GET"])
def faq_retroworld_alias():
    return send_from_directory(str(STATIC_DIR), "faq_retroworld.json")


@app.route("/faq_runningman.json", methods=["GET"])
def faq_runningman_alias():
    return send_from_directory(str(STATIC_DIR), "faq_runningman.json")


# ------------------ CHAT ------------------
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
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
    conv["meta"]["last_seen"] = now_iso()

    append_message(conv, "user", msg)

    # Cross brand detection: if user mentions another brand, we can include their KB as well.
    cross = []
    mentioned = detect_brand_from_text(msg)
    if mentioned and mentioned != brand_id and mentioned in BRANDS:
        cross.append(mentioned)

    # Determine KB snippets (deterministic)
    primary_items = kb_items_for_brand(brand_id)
    primary_hits = search_kb(primary_items, msg, limit=5)

    cross_hits = []
    for cb in cross:
        items = kb_items_for_brand(cb)
        cross_hits.extend(search_kb(items, msg, limit=3))

    kb_context = ""
    if primary_hits or cross_hits:
        kb_context += "Informations FAQ pertinentes:\n"
        for it in primary_hits:
            kb_context += f"- ({brand_id}) Q: {it['question']} | A: {it['answer']}\n"
        for it in cross_hits:
            # find which brand by looking up in their KB lists
            kb_context += f"- (autre établissement) Q: {it['question']} | A: {it['answer']}\n"

    # If OpenAI not configured, return message
    if not openai_ready():
        answer = "Le service IA n'est pas configuré (OPENAI_API_KEY manquante)."
        append_message(conv, "assistant", answer, extra={"brand_id": brand_id, "flags": ["openai_missing"]})
        save_conv(conv)
        return jsonify({"ok": True, "conversation_id": conv_id, "brand_id": brand_id, "answer": answer})

    sys_prompt = build_system_prompt(brand_id, cross)
    user_prompt = msg
    if kb_context:
        user_prompt = kb_context + "\n\nQuestion utilisateur:\n" + msg

    try:
        raw_answer = openai_answer(sys_prompt, user_prompt, max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS)
    except Exception as e:
        log("OpenAI error:", e)
        raw_answer = "Désolé, je rencontre un souci technique. Pouvez-vous réessayer ou contacter l’équipe ?"

    # Safety enforcement
    safe_answer, promised = enforce_no_reservation_promises(raw_answer)
    safe_answer = add_disclaimer_if_needed(safe_answer, brand_id, msg)

    flags = []
    if promised:
        flags.append("promesse_resa")

    append_message(conv, "assistant", safe_answer, extra={"brand_id": brand_id, "flags": flags})
    save_conv(conv)

    return jsonify({"ok": True, "conversation_id": conv_id, "brand_id": brand_id, "answer": safe_answer})


# ------------------ ADMIN UI ------------------
@app.route("/admin", methods=["GET"])
def admin_page():
    return send_from_directory(str(STATIC_DIR), "admin.html")


@app.route("/admin/faq", methods=["GET"])
def admin_faq_page():
    return send_from_directory(str(STATIC_DIR), "admin-faq.html")


# ------------------ ADMIN API ------------------
@app.route("/admin/api/diag", methods=["GET"])
def admin_diag():
    if not require_admin_token():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    return jsonify(
        {
            "ok": True,
            "time": now_iso(),
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

    # List all conversation JSON files
    items = []
    for p in sorted(CONV_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        conv = read_json(p) or {}
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
def admin_get_conversation(conv_id):
    if not require_admin_token():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    conv = load_conv(conv_id)
    conv["flags"] = compute_flags(conv)
    return jsonify({"ok": True, "conversation": conv})


@app.route("/admin/api/export.csv", methods=["GET"])
def admin_export_csv():
    if not require_admin_token():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    # Basic CSV export for quick diagnostics
    import csv
    from io import StringIO

    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(["conversation_id", "brand_id", "ts", "role", "content", "flags"])

    for p in sorted(CONV_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        conv = read_json(p) or {}
        cid = conv.get("id", p.stem)
        brand_id = (conv.get("meta") or {}).get("brand_id", "")
        flags = "|".join(compute_flags(conv))
        for m in conv.get("messages") or []:
            writer.writerow([cid, brand_id, m.get("ts", ""), m.get("role", ""), (m.get("content", "") or "").replace("\n", " "), flags])

    resp = make_response(out.getvalue())
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
    kb = read_json(KB_FILES[bid]) or {"brand": bid, "updated": now_iso(), "items": []}
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
    if bid not in FAQ_ENABLED_BRANDS:
        # allow saving even if not enabled, but it won't be public
        pass

    # validate schema
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

    kb_out = {"brand": bid, "updated": now_iso(), "items": cleaned}
    write_json(KB_FILES[bid], kb_out)

    # refresh static FAQ files (widget)
    if bid in ("retroworld", "runningman"):
        static_payload = {"brand": bid, "updated": kb_out["updated"], "items": kb_out["items"]}
        write_json(STATIC_FAQ_FILES[bid], static_payload)
        if bid == "runningman":
            write_json(STATIC_FAQ_FILES["runningman_legacy"], static_payload)

    return jsonify({"ok": True, "saved": True, "updated": kb_out["updated"], "count": len(cleaned)})


# ------------------------------------------------------------
# Static fallback routes
# ------------------------------------------------------------
@app.route("/static/<path:filename>", methods=["GET"])
def static_files(filename):
    return send_from_directory(str(STATIC_DIR), filename)


if __name__ == "__main__":
    # For local run
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=DEBUG_LOGS)
