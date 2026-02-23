import os
import re
import json
from datetime import datetime
from flask import Flask, request, jsonify

from src.data.system_data import SYSTEM_PROMPT

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")

# --------------------------------------------------
# OpenAI (Responses API)
# --------------------------------------------------

def openai_answer(system_prompt, user_message):
    import requests

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_message}]}
        ],
        "max_output_tokens": 800
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=payload,
        timeout=30
    )

    r.raise_for_status()
    data = r.json()

    texts = []
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                texts.append(c.get("text"))

    return "\n".join(texts).strip()


# --------------------------------------------------
# Sécurité réservation
# --------------------------------------------------

FORBIDDEN_WORDS = [
    "c'est réservé",
    "réservé",
    "confirmé",
    "je vous bloque",
    "bloqué"
]

def secure_reply(reply):
    lower = reply.lower()
    for word in FORBIDDEN_WORDS:
        if word in lower:
            reply += "\n\n⚠️ Je n’ai pas accès au planning en temps réel. La disponibilité est à confirmer par l’équipe."
            break
    return reply


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "openai_configured": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL
    })


@app.post("/chat")
def chat():
    if not OPENAI_API_KEY:
        return jsonify({
            "ok": True,
            "answer": "Le service IA n'est pas configuré (OPENAI_API_KEY manquante)."
        })

    data = request.get_json() or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"ok": False, "error": "message manquant"})

    reply = openai_answer(SYSTEM_PROMPT, message)
    reply = secure_reply(reply)

    return jsonify({
        "ok": True,
        "answer": reply
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
