FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/saved/classical ./models/saved/classical

EXPOSE 7001

CMD ["uvicorn", "src.classical.service:app", "--host", "0.0.0.0", "--port", "7001", "--reload"]
