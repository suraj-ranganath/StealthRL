FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY demo/requirements.txt /app/demo/requirements.txt
RUN pip install --no-cache-dir -r /app/demo/requirements.txt

COPY demo /app/demo

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn demo.stealthrl_demo.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
