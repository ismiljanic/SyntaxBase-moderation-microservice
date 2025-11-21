FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/saved/Qwen3-4B-Thinking-2507-MLX-4bit ./models/saved/Qwen3-4B-Thinking-2507-MLX-4bit

EXPOSE 7004

CMD ["uvicorn", "src.llm.llm_service:app", "--host", "0.0.0.0", "--port", "7004", "--reload"]