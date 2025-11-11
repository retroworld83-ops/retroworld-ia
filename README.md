# Retroworld IA — Voix + Chat

## 1) Créer le repo
Créez un nouveau dépôt GitHub public, copiez les fichiers:
- app.py
- Dockerfile
- .env.example
- static/chat-widget.html
- (optionnel) kb.json si vous voulez précharger des FAQs

## 2) Déployer sur Render
- New → **Web Service**
- Source: **Public Git repo** → collez l’URL de votre repo
- Environment: **Docker**
- Region: **EU (Frankfurt)**
- Port: **8080**
- Env vars (onglet "Environment"):
  - `OPENAI_API_KEY = sk-...`
  - `ADMIN_EMAILS = retroworld83@hotmail.com,contact@retroworldfrance.com`
  - `ADMIN_API_TOKEN = mettez-une-chaine-secrete`
  - `TZ = Europe/Paris`
- Créez le service. Attendez le build & le “Live”.

Test: `https://<votre-app>.onrender.com/health`

## 3) Intégrer le chat au site
**Option A — Iframe simple**  
Dans WordPress (IONOS) → Page → Bloc “HTML personnalisé” :
```html
<iframe src="https://<votre-app>.onrender.com/static/chat-widget.html"
        style="width:100%;max-width:760px;height:560px;border:0;border-radius:12px;overflow:hidden"></iframe>
