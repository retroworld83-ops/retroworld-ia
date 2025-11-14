FROM python:3.11-slim

WORKDIR /app

# Code
COPY app.py /app/

# Bases de connaissance JSON (Retroworld + Runningman)
COPY kb_retroworld.json /app/
COPY kb_runningman.json /app/

# Fichiers statiques (widget chat, etc.)
COPY static /app/static

# Dépendances
RUN pip install --no-cache-dir flask flask-cors

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python", "app.py"]
