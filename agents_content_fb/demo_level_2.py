from agent_content import OllamaAgent
from agent_qc import QCAgent


class ContentOrchestrator:
    def __init__(self, model="phi", max_retries=2):
        self.llm = OllamaAgent(model=model)
        self.qc = QCAgent(self.llm)
        self.max_retries = max_retries

    def generate_post(self, topic: str):
        prompt = f"""
Write a professional Facebook post (max 120 words)
about: {topic}

Keep tone professional and clear.
"""
        return self.llm.generate(prompt, temperature=0.4)

    def rewrite_post(self, original: str, feedback: str):
        prompt = f"""
Rewrite the following Facebook post based on QC feedback.

Feedback:
{feedback}

Post:
\"\"\"
{original}
\"\"\"

Keep under 120 words.
"""
        return self.llm.generate(prompt, temperature=0.3)

    def run(self, topic: str):
        post = self.generate_post(topic)

        for attempt in range(self.max_retries + 1):
            print(f"\n--- QC Attempt {attempt + 1} ---")

            result = self.qc.evaluate(post)

            print("QC Result:", result)

            if result["decision"] == "PASS":
                print("\n✅ Approved.")
                return post

            if attempt < self.max_retries:
                print("🔁 Rewriting...")
                post = self.rewrite_post(post, result["feedback"])
            else:
                print("❌ Max retries reached.")
                return post
            

orchestrator = ContentOrchestrator(model="phi", max_retries=2)

final_post = orchestrator.run("Smart personal finance habits")

print("\nFINAL POST:\n")
print(final_post)