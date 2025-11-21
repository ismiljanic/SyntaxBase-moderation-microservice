from fastapi import FastAPI
from pydantic import BaseModel
from src.llm.llm_moderation_client import LLMModerationClient
from pathlib import Path
import os

app = FastAPI(title="LLM Moderation Service")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "src/llm/prompts/api_call.prompt.txt"
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3-4b-thinking-plus")

llm_client = LLMModerationClient(
    base_url="http://host.docker.internal:1234/v1/chat/completions",
    # base_url="http://localhost:1234/v1/chat/completions",
    model_name=LLM_MODEL_NAME,
    prompt_path=str(PROMPT_PATH)
)

class Item(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Item):
    result = llm_client.classify_comment(item.text)
    return {
        "label": result["label"],
        "reasoning": result["reasoning"]
    }