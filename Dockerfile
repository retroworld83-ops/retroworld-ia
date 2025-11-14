FROM python:3.11-slim

# Installer curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Code
COPY app.py /app/

# KB Retroworld + Runningman
COPY kb_retroworld.json /app/
COPY kb_runningman.json /app/

# Dossier statique (ton dossier actuel est : static/)
COPY static /app/static

# Dépendances Python
RUN pip install --no-cache-dir flask flask-cors

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python", "app.py"]
