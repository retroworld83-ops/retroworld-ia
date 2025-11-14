FROM python:3.11-slim

# Installer curl (et nettoyer le cache APT)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Code
COPY app.py /app/

# Bases de connaissance JSON (Retroworld + Runningman)
COPY kb_retroworld.json /app/
COPY kb_runningman.json /app/

# Fichiers statiques (widget chat, etc.)
# ⚠️ Si ton dossier s'appelle "statique" (comme sur ta capture GitHub),
# garde cette ligne. Si tu le renommes en "static", change aussi ici.
COPY statique /app/static

# Dépendances Python
RUN pip install --no-cache-dir flask flask-cors

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python", "app.py"]
