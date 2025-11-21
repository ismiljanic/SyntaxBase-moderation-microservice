FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/saved/toxic_bert ./models/saved/toxic_bert

EXPOSE 7003

CMD ["uvicorn", "src.transformer.toxic_service:app", "--host", "0.0.0.0", "--port", "7003", "--reload"]