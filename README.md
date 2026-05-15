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
- `CORRECTION_MEMORY_MAX` nombre maximum de corrections reinjectees dans un prompt
- `OPENAI_CORRECTIONS_VECTOR_STORE_ID` optionnel, pour synchroniser les corrections approuvees vers un Vector Store OpenAI
- `APP_DB_PATH`
- `APP_DATA_DIR`
- `LEAD_WEBHOOK_URL` pour une integration future

## Admin

- `/admin/login` : connexion
- `/admin` : dashboard, conversations, analytics, leads
- `/admin/faq` : edition FAQ
- `/admin/knowledge` : edition de la base metier par marque
- `/admin/api/corrections` : memoire de corrections de reponses, separee de la FAQ

## Widget

Le widget public charge maintenant :

- les FAQ via `/faq.json`
- les cartes de contenu metier via `/knowledge.json`
- le chat via `/chat`

## Memoire de corrections

Depuis l'admin, une correction peut etre enregistree avec un declencheur et une reponse corrigee. Les prochaines questions proches recuperent ces corrections et les ajoutent au prompt OpenAI, sans modifier les fichiers FAQ publics.

OpenAI ne "reapprend" pas automatiquement sur une correction ponctuelle. Si `OPENAI_CORRECTIONS_VECTOR_STORE_ID` est configure, la correction est aussi envoyee dans le Vector Store OpenAI indique, et le chat peut rechercher dans ce Vector Store si la base locale ne suffit pas. Si la synchronisation ou la recherche OpenAI echoue, la correction locale reste active et le chat continue de fonctionner.

## Tests

```bash
python -m unittest tests.test_smoke -v
```

## Deploiement

Le projet reste compatible avec Docker et Render via `Dockerfile`, `render.yaml` et `Procfile`.
