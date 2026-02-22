from agent_content import OllamaAgent


class ConversationalOrchestrator:
    def __init__(self, model="phi", max_turns=4):
        self.llm = OllamaAgent(model=model)
        self.max_turns = max_turns

    def idea_agent(self, history):
        prompt = f"""
You are IdeaAgent.
Your job: write or revise a professional Facebook post.

Conversation history:
{history}

Respond ONLY with the new full revised post.
"""
        return self.llm.generate(prompt, temperature=0.4)

    def qc_agent(self, history):
        prompt = f"""
You are QCAgent.
Your job: strictly review the Facebook post.

Rules:
- Max 120 words
- Professional tone
- No harmful content

If acceptable → respond exactly:
APPROVED

If not → respond with:
FEEDBACK: <clear instruction for revision>

Conversation history:
{history}
"""
        return self.llm.generate(prompt, temperature=0)

    def run(self, topic: str):
        history = f"Topic: {topic}\n"

        print("\n=== Conversation Start ===\n")

        # Initial draft
        idea_response = self.llm.generate(
            f"Write a professional Facebook post (max 120 words) about: {topic}",
            temperature=0.4,
        )

        history += f"\nIdeaAgent:\n{idea_response}\n"

        print("IdeaAgent:\n", idea_response)

        for turn in range(self.max_turns):

            qc_response = self.qc_agent(history)

            history += f"\nQCAgent:\n{qc_response}\n"

            print("\nQCAgent:\n", qc_response)

            if "APPROVED" in qc_response:
                print("\n✅ Final Approved.")
                return idea_response

            # Extract feedback
            feedback = qc_response.replace("FEEDBACK:", "").strip()

            # Idea revises
            idea_response = self.idea_agent(history)

            history += f"\nIdeaAgent:\n{idea_response}\n"

            print("\nIdeaAgent (Revised):\n", idea_response)

        print("\n❌ Max turns reached.")
        return idea_response
    

conv = ConversationalOrchestrator(model="phi", max_turns=3)

final_post = conv.run("Smart personal finance habits")

print("\nFINAL RESULT:\n")
print(final_post)

