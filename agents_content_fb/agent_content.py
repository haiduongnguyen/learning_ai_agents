import requests
from datetime import datetime
from pathlib import Path


class OllamaAgent:
    def __init__(self, model="phi", host="http://localhost:11434"):
        self.model = model
        self.url = f"{host}/api/generate"

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=60,
        )

        if not response.ok:
            raise RuntimeError(response.text)

        return response.json()["response"].strip()

    def generate_and_save(self, prompt: str, folder="outputs") -> str:
        text = self.generate(prompt)

        Path(folder).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = Path(folder) / f"{timestamp}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        return text