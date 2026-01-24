# Retroworld / Runningman IA (Chat)

Service Flask prêt pour Render. Objectif: un chat **cohérent** (la conversation complète est envoyée à OpenAI autant que possible) avec un **modèle récent** (par défaut **gpt-4.1-mini**) et un mécanisme de fallback si un modèle n'est pas accessible.

## Déployer sur Render

1) Créez un repo Git et poussez ce dossier.
2) Render → New → **Web Service**
3) Environment: **Docker**
4) Renseignez les variables d'environnement (onglet "Environment"):

Variables minimum:
- `OPENAI_API_KEY` (obligatoire)
- `OPENAI_MODEL=gpt-4.1-mini` (minimum requis)
OPENAI_FALLBACK_MODELS=gpt-4.1-mini,gpt-4.1,gpt-4o

Recommandé:
- `OPENAI_HISTORY_MODE=full`
- `OPENAI_MAX_HISTORY_PAIRS=120`
- `OPENAI_PROMPT_CHAR_BUDGET=32000`
- `ADMIN_DASHBOARD_TOKEN` (protège `/admin`)
- `USER_HISTORY_TOKEN` (protège `/user/<user_id>/history`)
- `TZ=Europe/Paris`

Endpoints utiles:
- `GET /health`
- `POST /chat` (brand auto)
- `POST /chat/retroworld`
- `POST /chat/runningman`

## Intégrer le chat au site (iframe)

Dans WordPress (bloc HTML personnalisé):

```html
<iframe src="https://<votre-app>.onrender.com/static/chat-widget.html"
        style="width:100%;max-width:760px;height:560px;border:0;border-radius:12px;overflow:hidden"></iframe>
```

## Notes sur la cohérence (important)

- Par défaut, `OPENAI_HISTORY_MODE=full`: on envoie l'historique complet.
- Si la conversation devient trop longue, l'appli compresse automatiquement la partie la plus ancienne dans un résumé "system", tout en gardant la partie récente intégrale.
- Si vous voulez forcer un comportement "seulement les derniers messages": `OPENAI_HISTORY_MODE=recent`.
