FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ .

CMD ["uvicorn", "app.main_api:app", "--host", "0.0.0.0", "--port", "800"]
