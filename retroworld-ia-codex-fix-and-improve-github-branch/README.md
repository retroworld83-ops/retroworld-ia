# Pôle Loisirs Draguignan – Chatbot Multi‑Établissements

Ce dépôt contient une application Flask clef en main qui fournit :

- Un **chatbot multi‑établissements** (Retroworld, Runningman, Enigmaniac) piloté par un **prompt unique** situé dans `src/data/system_data.py`. Le chatbot ne s’appuie sur aucune base de connaissances externe ; toutes les informations proviennent de ce prompt.
- Une **FAQ publique** accessible par `/faq.json?brand_id=…`, afin d’alimenter l’onglet « FAQ » du widget.
- Une **interface administrateur** (`/admin`) pour consulter l’historique des conversations, les exporter au format CSV et identifier rapidement les demandes sensibles (devis, réclamation, réservation, etc.).

## Structure du dépôt

```
retroworld-ia-final/
├── app.py                 # Application Flask principale
├── requirements.txt       # Dépendances Python
├── README.md              # Ce fichier
├── src/
│   ├── __init__.py        # Nécessaire pour l’import des packages
│   └── data/
│       ├── __init__.py    # Vide
│       └── system_data.py # Prompt système complet (à personnaliser)
├── static/
│   ├── chat-widget.html   # Widget d’interface client (iframe)
│   ├── admin.html         # Tableau de bord des conversations
│   ├── admin-faq.html     # Éditeur de FAQ
│   ├── faq_retroworld.json# FAQ publique pour Retroworld
│   ├── faq_runningman.json# FAQ publique pour Runningman
│   └── static/
│       └── faq_runningman.json # Alias (legacy) pour certains widgets
└── data/
    └── conversations/      # Stockage des conversations (JSON)
```

## Configuration

L’application lit ses variables d’environnement pour se configurer. Voici les plus importantes :

- `OPENAI_API_KEY` (obligatoire pour activer l’IA) : clé API OpenAI.
- `OPENAI_MODEL` (défaut `gpt-5.2`) : modèle à utiliser.
- `BRAND_ID` (défaut `retroworld`) : marque par défaut si aucune n’est détectée.
- `ADMIN_API_TOKEN` / `ADMIN_DASHBOARD_TOKEN` : jetons pour sécuriser l’accès aux routes administrateur.
- `ALLOWED_ORIGINS` : liste séparée par des virgules des origines autorisées pour CORS.
- `FAQ_ENABLED_BRANDS` (`retroworld,runningman,enigmaniac` par défaut) : marques pour lesquelles la FAQ est publiée.
- `PUBLIC_BASE_URL` : URL publique du service (affichée dans `/brands.json`).
- `DEBUG_LOGS` (`true`/`false`) : active l’affichage de logs de debug côté serveur.

## Exécution locale

Pour lancer l’application en local :

```bash
cd retroworld-ia-final
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Variables d’environnement minimales
export OPENAI_API_KEY=sk-…
export ADMIN_API_TOKEN=retroAdminSuperToken2025
export ADMIN_DASHBOARD_TOKEN=retroAdminSuperToken2025

python app.py
```

Le serveur écoute sur `http://localhost:5000`. Les principaux endpoints sont :

- `/chat` (POST) : envoie une question au bot ; paramètres JSON : `message`, `conversation_id` (optionnel), `brand_id` (optionnel). Le serveur renvoie `answer` et un identifiant de conversation.
- `/faq.json?brand_id=retroworld` : renvoie la FAQ publique pour la marque.
- `/admin` : interface d’administration (nécessite un jeton dans l’en‑tête `Authorization: Bearer <token>`).
- `/health` : état du serveur (clés configurées, marques actives, etc.).

## Personnalisation du prompt

Le fichier `src/data/system_data.py` contient la totalité des descriptions et règles du Pôle Loisirs. Modifiez‑le pour enrichir ou corriger les informations. Le chatbot utilisera toujours ce prompt comme unique source de vérité.

## Contributions

Ce projet a été construit à partir des échanges précédents avec un utilisateur. Il vise à fournir un chatbot fiable sans injection de bases de connaissances externes, tout en offrant une gestion simple des conversations et de la FAQ. N’hésitez pas à adapter le code à vos besoins (ajout de scénarios, amélioration de l’interface, etc.).


## Déploiement Render (Docker)

Ce dépôt inclut un `Dockerfile` pour Render en mode Docker.

- Build Command: *(laisser vide en mode Docker)*
- Start Command: *(laisser vide en mode Docker)*
- Render détecte automatiquement le `Dockerfile` à la racine.

Le conteneur démarre avec:

```bash
gunicorn -w 2 -k gthread -b 0.0.0.0:$PORT app:app
```

Assurez-vous de définir vos variables d'environnement dans Render (notamment `OPENAI_API_KEY`, `ADMIN_API_TOKEN`, `ADMIN_DASHBOARD_TOKEN`, `ALLOWED_ORIGINS`, `PUBLIC_BASE_URL`).


### Dépannage Render: "failed to read dockerfile"
Si Render affiche `failed to read dockerfile: open Dockerfile: no such file or directory`:
- Vérifiez que le service pointe bien sur le bon dépôt **et la bonne branche** (`main`).
- Vérifiez que le commit déployé contient bien `Dockerfile` (fichier à la racine).
- Si vous utilisez Blueprint, utilisez le `render.yaml` de ce dépôt (il pointe explicitement vers `./Dockerfile`).


### Note sur Render: clone du dépôt à chaque build
C'est **normal**: Render clone le dépôt à chaque nouveau déploiement/build.
Cela ne signifie pas une erreur en soi.

Pour éviter les ambiguïtés de runtime:
- mode Docker: utiliser `Dockerfile` (et `render.yaml` si Blueprint),
- mode Python natif: `Procfile` est fourni (`web: gunicorn ...`).

`app.py` supporte aussi `SERVER_MODE=auto|flask|gunicorn` (par défaut `auto`) et passe automatiquement en gunicorn sur Render quand l'application est lancée via `python app.py`.


## Git: publier sur la branche `main`

Si vos changements existent seulement sur une branche locale (ex: `work`), ils ne seront pas déployés par un service qui suit `main`.

Exemple de commandes:

```bash
git checkout -B main
git push -u origin main
```

Ensuite, configurez Render pour suivre `main` et déclenchez un redeploy.


### Widget embarqué sur un autre domaine
Si le widget est servi depuis un domaine différent de l'API, passez la base API en query string:

```
https://retroworld-ia.onrender.com/static/chat-widget.html?api=https://retroworld-ia.onrender.com
```

Le widget accepte `api`, `api_base` ou `base_url` et conserve la valeur en localStorage (`rw_widget_api_base`).
