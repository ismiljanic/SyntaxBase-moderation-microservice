import json
import re
import requests
from typing import Dict, Any
from pathlib import Path

class LLMModerationClient:
    """
    Thin client for querying a local LLM running via LM Studio (OpenAI-compatible API).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1/chat/completions",
        model_name: str = "phi-4-reasoning-plus",
        prompt_path: str = "prompts/phi-4-reasoning-plus.prompt.txt"
    ):
        self.base_url = base_url
        self.model_name = model_name

        prompt_file = Path(__file__).parent / prompt_path
        self.system_prompt = prompt_file.read_text(encoding="utf-8").strip()

    def _extract_json(self, text: str):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in model output: {text}")
        return json.loads(match.group(0))

    def classify_comment(self, comment: str) -> Dict[str, Any]:

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": comment}
            ],
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "moderation_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "enum": ["safe", "mild", "toxic", "severe"]
                            },
                            "reasoning": {"type": "string"}
                        },
                        "required": ["label", "reasoning"]
                    }
                }
            }
        }

        response = requests.post(self.base_url, json=payload, timeout=60)

        if response.status_code != 200:
            raise RuntimeError(f"LLM API error {response.status_code}: {response.text}")

        content = response.json()["choices"][0]["message"]["content"].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return self._extract_json(content)


if __name__ == "__main__":
    client = LLMModerationClient()
    test_comment = "I will beat you when I see you!"
    output = client.classify_comment(test_comment)
    print(json.dumps(output, indent=2))