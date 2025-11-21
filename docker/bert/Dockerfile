FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/saved/bert ./models/saved/bert

EXPOSE 7002

CMD ["uvicorn", "src.transformer.distilBert_service:app", "--host", "0.0.0.0", "--port", "7002"]