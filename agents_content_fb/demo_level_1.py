# from agent_content import OllamaAgent

# agent = OllamaAgent(model="phi")

# prompt = "Write a short Facebook post about saving money in Vietnamese."

# result = agent.generate_and_save(prompt)

# print(result)


from agent_content import OllamaAgent
from agent_qc import QCAgent

llm = OllamaAgent(model="phi")
qc = QCAgent(llm)

# Step 1: Generate content
post = llm.generate_and_save(
    "Write a short professional Facebook post about saving money in Vietnamese."
)

print("Generated Post:\n", post)

# Step 2: QC evaluation
qc_result = qc.evaluate(post)

print("\nQC Result:\n", qc_result)