FROM python:3.11-slim

WORKDIR /app

# (Optionnel mais pratique) curl pour diagnostiquer en prod
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-10000} app:app"]
