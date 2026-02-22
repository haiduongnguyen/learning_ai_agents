import json
from datetime import datetime
from pathlib import Path


class QCAgent:
    def __init__(self, llm_agent):
        self.llm = llm_agent

    def evaluate(self, content: str, folder="qc_logs"):
        prompt = f"""
You are a strict content quality controller.

Evaluate the Facebook post below.

Criteria:
1. Clear and readable
2. No harmful or sensitive content
3. Suitable for professional audience
4. Not too long (max 150 words)

Return result ONLY in valid JSON format:

{{
  "score": number (0-10),
  "decision": "PASS" or "FAIL",
  "feedback": "short explanation"
}}

Post:
\"\"\"
{content}
\"\"\"
"""

        raw = self.llm.generate(prompt, temperature=0)

        try:
            result = json.loads(raw)
        except Exception:
            result = {
                "score": 0,
                "decision": "FAIL",
                "feedback": "Invalid JSON response from model"
            }

        # Save log
        Path(folder).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(f"{folder}/{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write("CONTENT:\n")
            f.write(content)
            f.write("\n\nQC RESULT:\n")
            f.write(json.dumps(result, indent=2, ensure_ascii=False))

        return result