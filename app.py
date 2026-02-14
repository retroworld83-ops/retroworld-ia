import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
# On charge les variables d'environnement (API Key, Port...)
load_dotenv()

# Configuration des logs (pour voir ce qui se passe sur Render)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialisation de Flask
app = Flask(__name__)
CORS(app)  # Autorise votre site web à parler au bot

# --- 2. IMPORT DU CERVEAU ---
# On charge le prompt intelligent qu'on a créé (Retroworld + Runningman + Enigmaniac)
try:
    from data.system_data import SYSTEM_PROMPT
except ImportError:
    logger.critical("ERREUR : Le fichier data/system_data.py est introuvable !")
    SYSTEM_PROMPT = "Erreur configuration : Base de connaissance introuvable."

# --- 3. CLIENT OPENAI ---
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.warning("ATTENTION : Pas de clé API OpenAI détectée !")

client = OpenAI(api_key=api_key)

# --- 4. LES ROUTES (INTERFACES) ---

@app.route('/', methods=['GET'])
def index():
    """Page d'accueil simple pour vérifier que le serveur tourne."""
    return jsonify({
        "status": "online",
        "service": "Pôle Loisirs AI (Retroworld / Runningman / Enigmaniac)",
        "version": "3.0"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Route pour le monitoring (Render/Uptime)."""
    return jsonify({"status": "ok"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    """
    LE CŒUR DU SYSTÈME
    Reçoit le message du site web -> Interroge l'IA -> Renvoie la réponse.
    """
    try:
        # Récupération des données envoyées par le site
        data = request.json
        user_message = data.get('message', '').strip()
        history = data.get('history', [])  # Historique de la conversation

        if not user_message:
            return jsonify({"error": "Message vide"}), 400

        # --- CONSTRUCTION DE L'INTELLIGENCE ---
        
        # 1. On injecte le SYSTEM PROMPT (Le Cerveau avec tous les tarifs et jeux)
        messages_payload = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # 2. On ajoute la mémoire (les 10 derniers messages)
        # Ça permet à l'IA de se souvenir : "Ah, il a dit qu'il avait 8 ans juste avant"
        # On nettoie l'historique pour garder un format propre
        for msg in history[-10:]:
            role = msg.get('role')
            content = msg.get('content')
            if role in ['user', 'assistant'] and content:
                messages_payload.append({"role": role, "content": content})

        # 3. On ajoute la question actuelle du client
        messages_payload.append({"role": "user", "content": user_message})

        # 4. Appel à OpenAI (GPT-4o)
        response = client.chat.completions.create(
            model="gpt-4o",  # Le modèle le plus intelligent pour la vente
            messages=messages_payload,
            temperature=0.7, # Créativité contrôlée (ni robotique, ni délirant)
            max_tokens=600   # Longueur max de la réponse
        )

        # Récupération de la réponse
        bot_reply = response.choices[0].message.content

        # Log pour debug (visible dans la console Render)
        logger.info(f"Question: {user_message[:50]}... -> Réponse générée.")

        return jsonify({
            "reply": bot_reply
        })

    except Exception as e:
        logger.error(f"ERREUR CRITIQUE : {str(e)}")
        # En cas de panne de l'IA, on renvoie un message de secours propre
        return jsonify({
            "reply": "Désolé, je rencontre un petit problème technique momentané. Vous pouvez nous joindre directement au 04 94 47 94 64 (Retroworld) ou 04 98 09 30 59 (Runningman)."
        }), 500

# --- 5. LANCEMENT DU SERVEUR ---
if __name__ == '__main__':
    # Récupération du port dynamique (obligatoire pour Render)
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
