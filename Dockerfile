FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY app.py /app/
COPY kb_retroworld.json /app/
COPY kb_runningman.json /app/
COPY static /app/static

RUN pip install --no-cache-dir flask flask-cors openai gunicorn

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-10000} app:app"]
