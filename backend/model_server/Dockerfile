FROM python:3.11-slim AS builder
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS final
WORKDIR /app

RUN addgroup --system app && \
    adduser --system --group app && \
    mkdir -p /app/app && \
    chown -R app:app /app \

USER app

COPY --from=builder /opt/venv /opt/venv

COPY ../app/logger_config.py ./app/
COPY ../app/classifier.py ./app/

COPY __init__.py ./
COPY main.py ./

ENV PATH="/opt/venv/bin:$PATH" \
    LOG_TO_FILE=false \
    LOG_LEVEL=INFO

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]