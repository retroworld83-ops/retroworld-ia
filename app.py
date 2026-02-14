import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# --- 1) CONFIG ---
# Charge .env en local (sur Render, vos variables sont déjà présentes)
load_dotenv()

# Logs : pilotés par DEBUG_LOGS=true/false
debug_logs = os.environ.get("DEBUG_LOGS", "false").strip().lower() == "true"
log_level = logging.DEBUG if debug_logs else logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Flask
app = Flask(__name__)

# CORS : restreint si ALLOWED_ORIGINS est fourni (recommandé)
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "").strip()
if allowed_origins:
    origins_list = [o.strip() for o in allowed_origins.split(",") if o.strip()]
    if "*" in origins_list:
        CORS(app, supports_credentials=True)
        logger.info("CORS: wildcard (*) autorisé")
    else:
        CORS(app, resources={r"/*": {"origins": origins_list}}, supports_credentials=True)
        logger.info("CORS: origines autorisées = %s", origins_list)
else:
    CORS(app, supports_credentials=True)
    logger.warning("CORS: ALLOWED_ORIGINS vide -> accès ouvert (à éviter en prod)")

# --- 2) IMPORT DU PROMPT SYSTEME ---
# IMPORTANT : le fichier est dans src/data/system_data.py
# Pour éviter les erreurs d'import selon l'arborescence, on ajoute /app/src au path si nécessaire.
# (Sur Render, votre app est généralement sous /app)
try:
    # Essai direct (si vous lancez depuis la racine et que src est dans PYTHONPATH)
    from src.data.system_data import SYSTEM_PROMPT
except Exception:
    try:
        # Fallback: on ajoute explicitement le dossier "src"
        import sys
        base_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(base_dir, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from data.system_data import SYSTEM_PROMPT  # type: ignore
    except Exception as e:
        logger.critical("ERREUR : Impossible d'importer SYSTEM_PROMPT (%s)", str(e))
        SYSTEM_PROMPT = "Erreur configuration : Base de connaissance introuvable."

# --- 3) OPENAI CLIENT ---
api_key = os.environ.get("OPENAI_API_KEY", "").strip()
if not api_key:
    logger.warning("ATTENTION : OPENAI_API_KEY manquante !")

client = OpenAI(api_key=api_key)

# Paramètres IA (depuis vos variables Render)
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o").strip()  # vous avez gpt-5.2
REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "").strip().lower()  # optionnel
try:
    TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.7"))
except ValueError:
    TEMPERATURE = 0.7

try:
    MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "600"))
except ValueError:
    MAX_OUTPUT_TOKENS = 600

# --- 4) ROUTES ---

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "service": "Pôle Loisirs AI (Retroworld / Runningman / Enigmaniac)",
        "public_base_url": os.environ.get("PUBLIC_BASE_URL", "").strip(),
        "model": MODEL_NAME,
        "version": "3.1"
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    """
    Reçoit:
      - message: str
      - history: [{role: 'user'|'assistant', content: '...'}, ...] (optionnel)
    Renvoie:
      - reply: str
    """
    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        history = data.get("history") or []

        if not user_message:
            return jsonify({"error": "Message vide"}), 400
        if not isinstance(history, list):
            history = []

        # Construire le payload messages
        messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Garder les 10 derniers échanges valides (anti-spam / anti-boucle)
        for msg in history[-10:]:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                messages_payload.append({"role": role, "content": content.strip()})

        messages_payload.append({"role": "user", "content": user_message})

        # Appel OpenAI
        # Note: selon SDK/modèle, reasoning_effort peut ne pas être supporté.
        # On l'applique uniquement s'il est renseigné.
        kwargs = dict(
            model=MODEL_NAME,
            messages=messages_payload,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

        # Tentative d'option reasoning (si supporté par votre modèle/SDK)
        if REASONING_EFFORT in ("low", "medium", "high", "none"):
            # On évite de casser si le SDK ne supporte pas : on catch plus bas.
            kwargs["reasoning_effort"] = REASONING_EFFORT  # type: ignore

        try:
            response = client.chat.completions.create(**kwargs)
        except TypeError:
            # Le SDK n'accepte pas reasoning_effort => on retente sans
            kwargs.pop("reasoning_effort", None)
            response = client.chat.completions.create(**kwargs)

        bot_reply = response.choices[0].message.content or ""

        if debug_logs:
            logger.debug("Payload roles: %s", [m["role"] for m in messages_payload])
            logger.debug("User: %s", user_message)
            logger.debug("Reply: %s", bot_reply[:400])

        return jsonify({"reply": bot_reply})

    except Exception as e:
        logger.error("ERREUR CRITIQUE /chat : %s", str(e))
        return jsonify({
            "reply": (
                "Désolé, je rencontre un souci technique momentané.\n"
                "Retroworld : 04 94 47 94 64\n"
                "Runningman : 04 98 09 30 59"
            )
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
