"""Simple demo runner for the OllamaAgent.

Run: python agent_demo.py --prompt "Write a short product description"
"""
import argparse
from agents_content_fb.agent_content import OllamaAgent


def main():
    p = argparse.ArgumentParser(description="Run OllamaAgent demo")
    p.add_argument("--model", default="llama2")
    p.add_argument("--prompt", required=True)
    args = p.parse_args()

    agent = OllamaAgent(model=args.model)
    try:
        out = agent.generate(args.prompt)
        print(out)
    except Exception as e:
        print("Error calling Ollama:", e)


if __name__ == "__main__":
    main()
from langchain_community.llms import Ollama
import re

llm = Ollama(
    model="phi",
    temperature=0
)

def extract_amount(text: str) -> int:
    numbers = re.findall(r"\d+", text)
    return int(numbers[0]) if numbers else 0

def fraud_rule(amount: int) -> str:
    if amount > 500_000_000:
        return "HIGH RISK"
    return "LOW RISK"

def explain_decision(amount: int, decision: str) -> str:
    prompt = f"""
Bạn là trợ lý AI nội bộ cho ngân hàng.

Thông tin giao dịch:
- Số tiền: {amount}
- Kết quả đánh giá: {decision}

Hãy giải thích ngắn gọn bằng tiếng Việt cho nhân viên ngân hàng.
"""
    return llm.invoke(prompt)

# ---- RUN DEMO ----
query = "Check risk for transaction amount 700000000"

amount = extract_amount(query)
decision = fraud_rule(amount)
explanation = explain_decision(amount, decision)

print("Amount:", amount)
print("Decision:", decision)
print("Explanation:", explanation)
