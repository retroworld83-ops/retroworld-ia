FROM python:3.11-slim
WORKDIR /app
COPY app.py /app/
COPY static /app/static
RUN pip install --no-cache-dir flask flask-cors
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python","app.py"]
