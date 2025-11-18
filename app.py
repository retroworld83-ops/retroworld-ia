import os
import json
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib.request
import urllib.error

# ---------------------------------------------------------
# CONFIG GLOBALE
# ---------------------------------------------------------

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retroworld-ia")

# Dossiers de base
BASE_DATA_DIR = "/mnt/data"
BASE_APP_DIR = "/app"

BASE_LOG_DIR = os.getenv("LOG_DIR", os.path.join(BASE_DATA_DIR, "logs"))
CONVERSATIONS_LOG_DIR = os.path.join(BASE_LOG_DIR, "conversations")
QWEEKLE_LOG_DIR = os.path.join(BASE_LOG_DIR, "qweekle")

for d in [BASE_LOG_DIR, CONVERSATIONS_LOG_DIR, QWEEKLE_LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # adapte si besoin

# Admin
ADMIN_DASHBOARD_TOKEN = os.getenv("ADMIN_DASHBOARD_TOKEN", "changeme_admin_token")

# Qweekle (webhook uniquement pour l’instant)
QWEEKLE_WEBHOOK_SECRET = os.getenv("QWEEKLE_WEBHOOK_SECRET", "")
QWEEKLE_SOURCE_NAME = os.getenv("QWEEKLE_SOURCE_NAME", "retroworld-qweekle")

# Marques supportées
SUPPORTED_BRANDS = {"retroworld", "runningman"}


# ---------------------------------------------------------
# OUTILS GÉNÉRAUX
# ---------------------------------------------------------

def load_kb(brand: str) -> Dict[str, Any]:
    """
    Charge la KB d'une marque :
    1) /mnt/data/kb_<brand>.json (prioritaire, version modifiable)
    2) /app/kb_<brand>.json (version embarquée dans l'image)
    """
    brand = brand.lower()
    candidates = [
        os.path.join(BASE_DATA_DIR, f"kb_{brand}.json"),
        os.path.join(BASE_APP_DIR, f"kb_{brand}.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    kb = json.load(f)
                logger.info("KB chargée pour %s depuis %s", brand, path)
                return kb
            except Exception as e:
                logger.error("Erreur lecture KB %s: %s", path, e)
    logger.warning("KB introuvable pour brand %s, utilisation d'une KB vide.", brand)
    return {}


def save_kb(brand: str, kb_data: Dict[str, Any]) -> None:
    """
    Sauvegarde/écrase la KB d'une marque dans /mnt/data/kb_<brand>.json
    (sans toucher à /app).
    """
    brand = brand.lower()
    path = os.path.join(BASE_DATA_DIR, f"kb_{brand}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)
    logger.info("KB %s mise à jour dans %s", brand, path)


def call_openai_chat(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    """
    Appelle l'API OpenAI Chat. On passe une liste de messages.
    Retourne (réponse_texte, usage_dict).
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY manquant")

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
        # Log détaillé pour debug
        err_body = e.read().decode("utf-8", errors="ignore")
        logger.error("HTTPError OpenAI (%s): %s", e.code, err_body)
        raise
    except urllib.error.URLError as e:
        logger.error("URLError OpenAI: %s", e)
        raise

    obj = json.loads(body)
    content = obj["choices"][0]["message"]["content"]
    usage = obj.get("usage", {})
    return content, usage


def build_prompt(
    brand: str,
    kb: Dict[str, Any],
    messages: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Construit la liste de messages au format OpenAI Chat :
    - un gros message system basé sur la KB (+ fallback si KB vide)
    - un message de contexte (source, URL, conversation_id)
    - puis l'historique utilisateur/assistant reçu
    """
    brand = brand.lower()

    system_parts: List[str] = []

    # 1) Identité
    identite = kb.get("identite") if isinstance(kb, dict) else None
    if isinstance(identite, dict):
        nom = identite.get("nom") or brand.title()
        role = identite.get("role") or ""
        system_parts.append(f"Tu es l'assistant IA de {nom}. {role}".strip())
    elif isinstance(identite, str):
        system_parts.append(identite)

    # 2) Prompt général détaillé
    prompt_section = kb.get("prompt") if isinstance(kb, dict) else None
    if isinstance(prompt_section, str):
        system_parts.append(prompt_section)
    elif isinstance(prompt_section, dict):
        # On parcourt dans l'ordre stable des clés pour que le comportement soit reproductible
        for key in sorted(prompt_section.keys()):
            v = prompt_section[key]
            if isinstance(v, str) and v.strip():
                system_parts.append(v.strip())

    # 3) Instructions générales éventuelles
    instr = kb.get("instructions_generales") if isinstance(kb, dict) else None
    if isinstance(instr, list):
        for item in instr:
            if isinstance(item, str):
                system_parts.append(item)
    elif isinstance(instr, str):
        system_parts.append(instr)

    # 4) Rappel éventuel sur la marque d'entrée vs marque effective
    brand_entry = metadata.get("brand_entry")
    brand_effective = metadata.get("brand_effective")
    if brand_entry and brand_effective and brand_entry != brand_effective:
        system_parts.append(
            f"La conversation vient d'un canal associé à '{brand_entry}', "
            f"mais tu dois répondre en utilisant les règles et tarifs de '{brand_effective}'. "
            "Explique clairement au client s'il s'agit de Retroworld (VR, quiz, salle enfant) "
            "ou de Runningman (action game, mini-jeux physiques)."
        )

    # 5) Anti erreurs (pas d'invention de prix / promos / liens)
    anti_err = kb.get("anti_erreurs") if isinstance(kb, dict) else None
    if isinstance(anti_err, list):
        for item in anti_err:
            if isinstance(item, str):
                system_parts.append(item)
    elif isinstance(anti_err, str):
        system_parts.append(anti_err)

    # 6) Fallback de sécurité si KB vide ou très courte
    if not kb or not system_parts:
        if brand == "retroworld":
            system_parts.append(
                "Tu es l'assistant officiel de Retroworld France à Draguignan. "
                "Tu réponds en français, vouvoiement uniquement, sur les sujets suivants : "
                "jeux VR, escape games VR, quiz interactif, salle enfant, anniversaires et fidélité. "
                "Tu donnes toujours rapidement les prix, la durée et le nombre de joueurs possibles. "
                "Si tu n'es pas sûr d'une information, tu le dis et proposes au client d'appeler Retroworld au 04 94 47 94 64."
            )
        elif brand == "runningman":
            system_parts.append(
                "Tu es l'assistant pour Runningman Game Zone. "
                "Tu expliques que pour les informations précises (tarifs, réservations) concernant l'action game, "
                "il faut contacter directement Runningman au 04 98 09 30 59 ou via leur site www.runningmangames.fr. "
                "Tu peux néanmoins orienter le client vers Retroworld pour les activités VR, escape games VR, quiz et salle enfant."
            )

    system_text = "\n\n".join([p for p in system_parts if p.strip()])

    prompt_messages: List[Dict[str, str]] = []
    if system_text:
        prompt_messages.append({"role": "system", "content": system_text})

    # Contexte métadonnées (source, URL, conversation_id partagé)
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
            f"{conv_id} pour que la conversation continue sur l'autre site."
        )
    if meta_context:
        prompt_messages.append({"role": "system", "content": " ".join(meta_context)})

    # Historique utilisateur / assistant
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if not role or content is None:
            continue
        if role not in ("user", "assistant", "system"):
            continue
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
    """
    Append un enregistrement dans un fichier JSONL par conversation.
    """
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
    """
    Lit tous les enregistrements d'une conversation (JSONL) et les renvoie triés par timestamp.
    """
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


# ---------------------------------------------------------
# ROUTAGE MARQUE (Runningman / Retroworld)
# ---------------------------------------------------------

def detect_brand_from_text(text: str, default: str = "runningman") -> str:
    """
    Détecte si une demande parle plutôt de Retroworld (VR, quiz, salle enfant)
    ou de Runningman (action game, mini-jeux physiques).
    """
    if not text:
        return default

    t = text.lower()

    # Mots-clés typiques Retroworld
    retro_keywords = [
        "vr",
        "réalité virtuelle",
        "realite virtuelle",
        "escape vr",
        "escape game vr",
        "jeux vr",
        "jeu vr",
        "casque vr",
        "quiz",
        "quizz",
        "quiz interactif",
        "salle enfant",
        "mur interactif",
        "anniversaire vr",
        "retroworld",
        "rétroworld",
        "fidélité",
        "fidelite",
        "carte de fidélité",
        "points fidélité",
        "points de fidelite",
    ]

    # Mots-clés typiques Runningman
    running_keywords = [
        "action game",
        "game zone",
        "runningman",
        "running man",
        "mini-jeux",
        "mini jeux",
        "parcours",
        "parcour",
        "salle runningman",
        "gilet",
        "capteur",
        "mission physique",
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


def classify_conversation_brands(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyse les enregistrements d'une conversation pour déterminer :
      - brand_final : "runningman", "retroworld", "mixed" ou "unknown"
      - brands_seen : liste des marques rencontrées
    L'intention finale est la dernière marque effective vue.
    """
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


# ---------------------------------------------------------
# QWEEKLE – WEBHOOK (log brut pour l’instant)
# ---------------------------------------------------------

def append_qweekle_event(event_type: str, payload: Dict[str, Any]) -> None:
    """
    Log d'un événement Qweekle dans un fichier JSONL dédié par type d'event.
    """
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
# ENDPOINTS
# ---------------------------------------------------------

@app.route("/", methods=["GET", "HEAD"])
def root():
    """
    Petit endpoint racine pour les checks Render / humains.
    Évite un 404 sur HEAD / et donne un aperçu rapide de l'état.
    """
    return jsonify(
        {
            "service": "retroworld-ia",
            "status": "ok",
            "time": time.time(),
            "brands": list(SUPPORTED_BRANDS),
        }
    ), 200


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    """
    Empêche les 404 gênants sur /favicon.ico dans les logs.
    On ne renvoie pas d'icône pour l'instant.
    """
    return "", 204


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": time.time()}), 200


@app.route("/chat/<brand>", methods=["POST"])
def chat_route(brand: str):
    """
    Endpoint générique de chat :
    - /chat/retroworld
    - /chat/runningman

    Pour runningman, on route automatiquement vers Retroworld si la demande
    porte clairement sur VR / quiz / salle enfant.

    Le même endpoint servira pour :
    - le widget web,
    - le téléphone (Vonage) : il suffit d'envoyer un 'source': 'phone' dans metadata,
      et éventuellement un conversation_id partagé entre web et téléphone.
    """
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

    # Dernier message user
    last_user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_text = str(msg.get("content") or "")
            break

    # Détermination de la marque effective
    effective_brand = brand_entry
    if brand_entry == "runningman":
        effective_brand = detect_brand_from_text(last_user_text, default="runningman")

    # conversation_id : on fait confiance à ce que le front envoie
    conversation_id = metadata.get("conversation_id")
    if not conversation_id:
        conversation_id = f"{effective_brand}_{int(time.time() * 1000)}"
        metadata["conversation_id"] = conversation_id

    metadata["brand_entry"] = brand_entry
    metadata["brand_effective"] = effective_brand

    # Chargement KB
    kb = load_kb(effective_brand)

    # Construction du prompt
    try:
        prompt_messages = build_prompt(effective_brand, kb, messages, metadata)
    except Exception as e:
        logger.error("Erreur build_prompt: %s", e)
        return jsonify({"error": "prompt_build_failed"}), 500

    # Appel OpenAI
    try:
        reply_text, usage = call_openai_chat(prompt_messages)
    except Exception as e:
        logger.error("Erreur OpenAI: %s", e)
        return jsonify({"error": "openai_error", "details": str(e)}), 502

    # Logs
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
        logger.error("Erreur append_conversation_log: %s", e)

    return jsonify(
        {
            "reply": reply_text,
            "brand_used": effective_brand,
            "brand_entry": brand_entry,
        }
    ), 200


@app.route("/kb/upsert/<brand>", methods=["POST"])
def kb_upsert(brand: str):
    """
    Met à jour (ou crée) la KB d'une marque.
    Écrit dans /mnt/data/kb_<brand>.json
    """
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    if not isinstance(body, dict):
        return jsonify({"error": "invalid_kb"}), 400

    try:
        save_kb(brand, body)
    except Exception as e:
        logger.error("Erreur save_kb(%s): %s", brand, e)
        return jsonify({"error": "kb_save_failed"}), 500

    return jsonify({"status": "ok", "brand": brand}), 200


@app.route("/webhooks/qweekle", methods=["POST"])
def qweekle_webhook():
    """
    Réception des webhooks Qweekle.
    Pour l'instant : log brut, avec vérification du secret si défini.
    """
    if QWEEKLE_WEBHOOK_SECRET:
        incoming_secret = request.headers.get("X-Qweekle-Secret") or ""
        if incoming_secret != QWEEKLE_WEBHOOK_SECRET:
            logger.warning("Webhook Qweekle rejeté (secret invalide)")
            return jsonify({"error": "forbidden"}), 403

    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    event_type = payload.get("event_type") or payload.get("type") or "unknown"
    logger.info("Webhook Qweekle reçu: %s", event_type)
    append_qweekle_event(event_type, payload)

    # TODO plus tard : traitement booking.created, sale.created, etc.
    return jsonify({"status": "ok", "event_type": event_type}), 200


# ---------------------------------------------------------
# ADMIN – API CONVERSATIONS
# ---------------------------------------------------------

@app.route("/admin/api/conversations", methods=["GET"])
def admin_api_conversations():
    """
    Retourne la liste des conversations pour l'admin.
    GET ?token=ADMIN_DASHBOARD_TOKEN
    """
    token = request.args.get("token") or ""
    if token != ADMIN_DASHBOARD_TOKEN:
        return jsonify({"error": "forbidden"}), 403

    convs: List[Dict[str, Any]] = []

    if not os.path.isdir(CONVERSATIONS_LOG_DIR):
        return jsonify(convs), 200

    for fname in os.listdir(CONVERSATIONS_LOG_DIR):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(CONVERSATIONS_LOG_DIR, fname)
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
        brand_final = brand_info["brand_final"]

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
def admin_api_conversation_detail(conversation_id: str):
    """
    Retourne le détail complet d'une conversation (tous les tours).
    GET ?token=ADMIN_DASHBOARD_TOKEN
    """
    token = request.args.get("token") or ""
    if token != ADMIN_DASHBOARD_TOKEN:
        return jsonify({"error": "forbidden"}), 403

    records = load_conversation_records(conversation_id)
    if not records:
        return jsonify({"error": "not_found"}), 404

    return jsonify(
        {
            "conversation_id": conversation_id,
            "records": records,
        }
    ), 200


# ---------------------------------------------------------
# ADMIN – INTERFACE HTML
# ---------------------------------------------------------

@app.route("/admin/conversations", methods=["GET"])
def admin_conversations_page():
    token = request.args.get("token") or ""
    if token != ADMIN_DASHBOARD_TOKEN:
        return "Forbidden", 403

    return """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<title>Admin IA – Conversations</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root {
    --bg: #020617;
    --bg-card: #0b1120;
    --border: #1f2937;
    --text: #e5e7eb;
    --muted: #9ca3af;
    --accent: #38bdf8;
    --brand-retro: #6366f1;
    --brand-run: #22c55e;
    --brand-mix: #f97316;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: radial-gradient(circle at top, #0f172a, #020617 55%);
    color: var(--text);
  }
  .admin-shell {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px 16px 40px;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    margin-bottom: 18px;
  }
  .title-block h1 {
    margin: 0;
    font-size: 24px;
  }
  .title-block p {
    margin: 4px 0 0;
    font-size: 13px;
    color: var(--muted);
  }
  .filters {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .chip {
    border-radius: 999px;
    border: 1px solid var(--border);
    padding: 6px 10px;
    font-size: 12px;
    cursor: pointer;
    background: rgba(15,23,42,0.9);
    color: var(--muted);
  }
  .chip.active {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(56, 189, 248, 0.1);
  }
  .chip[data-brand="runningman"].active { border-color: var(--brand-run); color: var(--brand-run); }
  .chip[data-brand="retroworld"].active { border-color: var(--brand-retro); color: var(--brand-retro); }
  .chip[data-brand="mixed"].active { border-color: var(--brand-mix); color: var(--brand-mix); }

  .toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 14px;
    align-items: center;
  }
  .search-input {
    flex: 1;
    min-width: 200px;
  }
  .search-input input {
    width: 100%;
    padding: 7px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: #020617;
    color: var(--text);
    font-size: 13px;
  }
  .btn-small {
    border-radius: 999px;
    border: 1px solid var(--border);
    padding: 6px 10px;
    font-size: 12px;
    background: #020617;
    color: var(--muted);
    cursor: pointer;
  }
  .btn-small:hover {
    border-color: var(--accent);
    color: var(--accent);
  }

  .table-wrapper {
    border-radius: 14px;
    border: 1px solid var(--border);
    background: rgba(15,23,42,0.95);
    overflow: hidden;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  thead {
    background: #020617;
  }
  th, td {
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    text-align: left;
    vertical-align: top;
  }
  th {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
  }
  tr:hover td {
    background: rgba(15,23,42,0.7);
  }
  .badge {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .badge-run { background: rgba(34,197,94,0.18); color: #4ade80; }
  .badge-retro { background: rgba(99,102,241,0.18); color: #a5b4fc; }
  .badge-mix { background: rgba(249,115,22,0.18); color: #fbbf24; }
  .badge-unknown { background: rgba(148,163,184,0.18); color: #cbd5f5; }

  .pill {
    display: inline-flex;
    border-radius: 999px;
    padding: 2px 7px;
    font-size: 11px;
    color: var(--muted);
    border: 1px solid rgba(148,163,184,0.3);
  }
  .pill-channel {
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .preview {
    color: var(--text);
  }
  .muted {
    color: var(--muted);
    font-size: 11px;
  }
  @media (max-width: 768px) {
    th:nth-child(2), td:nth-child(2) { display: none; }
    th:nth-child(4), td:nth-child(4) { display: none; }
  }

  .conv-detail {
    margin-top: 18px;
    border-radius: 14px;
    border: 1px solid var(--border);
    background: radial-gradient(circle at top left, rgba(56,189,248,0.18), transparent 45%), #020617;
    padding: 14px;
    max-height: 420px;
    overflow: auto;
  }
  .conv-detail-title {
    font-size: 13px;
    margin-bottom: 8px;
    color: var(--muted);
  }
  .bubble {
    max-width: 75%;
    margin-bottom: 8px;
    padding: 7px 10px;
    border-radius: 14px;
    font-size: 13px;
    line-height: 1.4;
    white-space: pre-wrap;
  }
  .bubble-user {
    margin-left: auto;
    background: #1f2937;
    border-bottom-right-radius: 4px;
  }
  .bubble-bot {
    margin-right: auto;
    background: #020617;
    border: 1px solid #1f2937;
    border-bottom-left-radius: 4px;
  }
  .meta-line {
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 10px;
  }
</style>
</head>
<body>
<div class="admin-shell">
  <header>
    <div class="title-block">
      <h1>Conversations IA</h1>
      <p>Vue d'ensemble des échanges Retroworld / Runningman (chat & téléphone).</p>
    </div>
    <div class="filters">
      <button class="chip active" data-filter="all">Tout</button>
      <button class="chip" data-filter="runningman" data-brand="runningman">Runningman</button>
      <button class="chip" data-filter="retroworld" data-brand="retroworld">Retroworld</button>
      <button class="chip" data-filter="mixed" data-brand="mixed">Mix des deux</button>
    </div>
  </header>

  <div class="toolbar">
    <div class="search-input">
      <input type="text" id="search" placeholder="Rechercher dans les questions, sources, IDs…" />
    </div>
    <button class="btn-small" id="btn-refresh">Rafraîchir</button>
  </div>

  <div class="table-wrapper">
    <table>
      <thead>
        <tr>
          <th style="width: 155px;">Date</th>
          <th style="width: 85px;">Canal</th>
          <th style="width: 120px;">Marque</th>
          <th style="width: 150px;">Source</th>
          <th>Dernière question / aperçu</th>
        </tr>
      </thead>
      <tbody id="rows">
        <tr><td colspan="5" class="muted">Chargement…</td></tr>
      </tbody>
    </table>
  </div>

  <div class="conv-detail" id="convDetail">
    <div class="conv-detail-title">Détail de la conversation</div>
    <div class="muted">Cliquez sur une conversation dans le tableau pour voir le fil complet ici.</div>
  </div>
</div>

<script>
(function() {
  const params = new URLSearchParams(window.location.search);
  const token = params.get("token") || "";
  const rowsEl = document.getElementById("rows");
  const searchInput = document.getElementById("search");
  const btnRefresh = document.getElementById("btn-refresh");
  const chips = Array.from(document.querySelectorAll(".chip"));
  const convDetail = document.getElementById("convDetail");

  let allData = [];
  let currentFilter = "all";
  let searchTerm = "";
  let currentConvId = null;

  function formatDate(ts) {
    if (!ts) return "";
    try {
      const d = new Date(ts * 1000);
      return d.toLocaleString("fr-FR", {
        day: "2-digit",
        month: "2-digit",
        year: "2-digit",
        hour: "2-digit",
        minute: "2-digit"
      });
    } catch(e) { return ""; }
  }

  function brandBadge(conv) {
    const b = conv.brand_final;
    if (b === "runningman") return '<span class="badge badge-run">Runningman</span>';
    if (b === "retroworld") return '<span class="badge badge-retro">Retroworld</span>';
    if (b === "mixed") return '<span class="badge badge-mix">Mix</span>';
    return '<span class="badge badge-unknown">Inconnu</span>';
  }

  function channelPill(conv) {
    const ch = (conv.channel || "web").toUpperCase();
    return '<span class="pill pill-channel">' + ch + '</span>';
  }

  function sourcePill(conv) {
    const s = conv.source || "n/a";
    return '<span class="pill">' + s + '</span>';
  }

  function render() {
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

    const html = filtered.map(c => `
      <tr onclick="viewConversation('${c.conversation_id}')">
        <td>
          <div>${formatDate(c.timestamp)}</div>
          <div class="muted">${c.conversation_id}</div>
        </td>
        <td>${channelPill(c)}</td>
        <td>${brandBadge(c)}</td>
        <td>${sourcePill(c)}</td>
        <td>
          <div class="preview">${c.preview || "<span class='muted'>(pas de message utilisateur)</span>"}</div>
        </td>
      </tr>
    `).join("");
    rowsEl.innerHTML = html;
  }

  async function loadData() {
    rowsEl.innerHTML = '<tr><td colspan="5" class="muted">Chargement…</td></tr>';
    try {
      const res = await fetch(`/admin/api/conversations?token=${encodeURIComponent(token)}`);
      if (!res.ok) {
        rowsEl.innerHTML = '<tr><td colspan="5" class="muted">Erreur de chargement ('+res.status+')</td></tr>';
        return;
      }
      allData = await res.json();
      render();
    } catch (e) {
      console.error(e);
      rowsEl.innerHTML = '<tr><td colspan="5" class="muted">Erreur réseau.</td></tr>';
    }
  }

  window.viewConversation = async function(id) {
    currentConvId = id;
    convDetail.innerHTML = '<div class="conv-detail-title">Conversation ' + id + '</div><div class="muted">Chargement…</div>';
    try {
      const res = await fetch(`/admin/api/conversation/${encodeURIComponent(id)}?token=${encodeURIComponent(token)}`);
      if (!res.ok) {
        convDetail.innerHTML = '<div class="conv-detail-title">Conversation ' + id + '</div><div class="muted">Erreur de chargement ('+res.status+')</div>';
        return;
      }
      const data = await res.json();
      const records = data.records || [];
      if (!records.length) {
        convDetail.innerHTML = '<div class="conv-detail-title">Conversation ' + id + '</div><div class="muted">Aucun enregistrement pour cette conversation.</div>';
        return;
      }
      let html = '<div class="conv-detail-title">Conversation ' + id + '</div>';
      records.forEach(rec => {
        const userMsgs = rec.user_messages || [];
        const reply = rec.assistant_reply || "";
        userMsgs
          .filter(m => m.role === "user")
          .forEach(m => {
            html += '<div class="bubble bubble-user">' + (m.content || "") + '</div>';
          });
        if (reply) {
          html += '<div class="bubble bubble-bot">' + reply + '</div>';
        }
        if (rec.timestamp) {
          const d = new Date(rec.timestamp * 1000).toLocaleString("fr-FR");
          html += '<div class="meta-line">' + d + '</div>';
        }
      });
      convDetail.innerHTML = html;
      convDetail.scrollTop = convDetail.scrollHeight;
    } catch (e) {
      console.error(e);
      convDetail.innerHTML = '<div class="conv-detail-title">Conversation ' + id + '</div><div class="muted">Erreur réseau.</div>';
    }
  }

  searchInput.addEventListener("input", function() {
    searchTerm = this.value;
    render();
  });

  btnRefresh.addEventListener("click", loadData);

  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      chips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      currentFilter = chip.getAttribute("data-filter") || "all";
      render();
    });
  });

  loadData();
})();
</script>
</body>
</html>
    """


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    # Lancement du service Flask (web + futur assistant téléphone via même /chat/<brand>)
    app.run(host="0.0.0.0", port=port)
