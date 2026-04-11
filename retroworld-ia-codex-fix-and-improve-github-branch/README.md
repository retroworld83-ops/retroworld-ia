# Retroworld IA V2

Backend Flask pour un chatbot multi-etablissements autour de Retroworld, Runningman et Enigmaniac.

## Ce qui a ete pousse plus loin

- backend modularise dans `src/retroworld_ia/`
- stockage SQLite pour conversations, messages, utilisateurs admin et leads
- historique conversationnel reinjecte au modele OpenAI
- admin securise par utilisateur, mot de passe, session et CSRF
- dashboard analytics avec scoring de leads
- base metier structuree editable depuis l'admin
- widget public enrichi avec actions rapides, cartes d'offres et FAQ

## Structure

```text
app.py
src/retroworld_ia/
  app_factory.py
  config.py
  routes/
    public.py
    admin.py
  services/
    ai.py
    auth.py
    conversations.py
    knowledge.py
    logging_store.py
src/data/
  system_data.py
  knowledge_base.json
static/
  admin-login.html
  admin.html
  admin-faq.html
  admin-knowledge.html
  chat-widget.html
tests/test_smoke.py
```

## Variables d'environnement

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `ADMIN_USERNAME`
- `ADMIN_PASSWORD`
- `SECRET_KEY`
- `ALLOWED_ORIGINS`
- `PUBLIC_BASE_URL`
- `FAQ_ENABLED_BRANDS`
- `PUBLIC_BRANDS`
- `CHAT_HISTORY_MESSAGES`
- `APP_DB_PATH`
- `APP_DATA_DIR`
- `LEAD_WEBHOOK_URL` pour une integration future

## Admin

- `/admin/login` : connexion
- `/admin` : dashboard, conversations, analytics, leads
- `/admin/faq` : edition FAQ
- `/admin/knowledge` : edition de la base metier par marque

## Widget

Le widget public charge maintenant :

- les FAQ via `/faq.json`
- les cartes de contenu metier via `/knowledge.json`
- le chat via `/chat`

## Tests

```bash
python -m unittest tests.test_smoke -v
```

## Deploiement

Le projet reste compatible avec Docker et Render via `Dockerfile`, `render.yaml` et `Procfile`.
