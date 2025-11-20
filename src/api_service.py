# src/api_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import classify_text_api
import uvicorn
from typing import Optional

app = FastAPI(title="SyntaxBase Moderation Microservice")

class Comment(BaseModel):
    text: str
    user_id: Optional[str] = None
    post_id: Optional[int] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify")
def classify(comment: Comment):
    try:
        result = classify_text_api(comment.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api_service:app", host="0.0.0.0", port=8000, reload=False)
