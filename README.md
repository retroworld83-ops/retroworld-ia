# Retroworld / Runningman / Enigmaniac IA (Chat)

Service Flask prêt pour Render (Docker).

 - Widget chat: `/static/chat-widget.html` (supporte `?brand=retroworld|runningman|enigmaniac`)
 - Dashboard admin: `/admin` (liste des conversations + lecture)
 - Admin FAQ: `/admin/faq` (édition JSON par établissement)
 - FAQ (JSON + page): `/faq.json` et `/faq?brand_id=...`

## Déployer sur Render

1) Créez un repo Git et poussez ce dossier.
2) Render → New → **Web Service**
3) Environment: **Docker**
4) Ajoutez vos variables d'environnement (onglet "Environment"):

Variables minimum:
- `OPENAI_API_KEY` (obligatoire)
- `OPENAI_MODEL` (ex: `gpt-5.2` ou `gpt-4.1-mini`)

Multi-établissements:
- `BRAND_ID=retroworld` (marque par défaut)
- `ALLOWED_ORIGINS=https://www.retroworldfrance.com,https://retroworldfrance.com,https://www.runningmangames.fr,https://runningmangames.fr`
- (facultatif) `BRANDS_CONFIG_PATH=/app/config/brands.yaml`

Fallback (optionnel):
- `OPENAI_FALLBACK_MODELS=gpt-4.1-mini,gpt-4.1,gpt-4o`

Historique (optionnel):
- `OPENAI_HISTORY_MODE=full` (ou `recent`)
- `OPENAI_MAX_HISTORY_PAIRS=120`
- `OPENAI_PROMPT_CHAR_BUDGET=32000`

Sécurité (optionnel):
- `ADMIN_DASHBOARD_TOKEN` : protège les endpoints `/admin/api/*` (le dashboard lui-même charge, puis demande le token)
- `USER_HISTORY_TOKEN` : protège `/user/<user_id>/history`

Fuseau horaire (optionnel):
- `TZ=Europe/Paris`

Endpoints utiles:
- `GET /health`
- `POST /chat` (brand auto)
- `POST /chat/retroworld`
- `POST /chat/runningman`
- `POST /chat/enigmaniac`

## Intégrer le chat au site (iframe)

Dans WordPress (bloc HTML personnalisé):

```html
<iframe src="https://<votre-app>.onrender.com/static/chat-widget.html"
        style="width:100%;max-width:760px;height:560px;border:0;border-radius:12px;overflow:hidden"></iframe>
```

## FAQ

Le contenu FAQ est dans:
- `kb_retroworld.json`
- `kb_runningman.json`
 - `kb_enigmaniac.json`

Vous pouvez les enrichir (questions/réponses) sans toucher au code.
