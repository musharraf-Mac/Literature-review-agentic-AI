from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter  # or however you imported it earlier
import os
from dotenv import load_dotenv

load_dotenv()

# Stronger model for final writing quality — via OpenRouter as planned
llm_synth = ChatOpenRouter(
    api_key=os.getenv("OPENROUTER_APIKEY"),
    model="openai/gpt-4o-mini",  # confirm exact current model string on OpenRouter
    temperature=0.3,  # slightly higher than 0 — some writing variety, but still controlled
)

def synthesize_review(topic, qa_pairs):
    qa_text = ""
    for qa in qa_pairs:
        qa_text += f"\nQ: {qa['question']}\nA: {qa['answer']}\nSources: {', '.join(qa['sources'])}\n"

    prompt = f"""You are writing a literature review section on the following topic:

Topic: {topic}

Below are research questions with grounded answers drawn from real papers, including their sources. Synthesize these into a coherent, well-organized literature review. Structure it with clear paragraphs (not just restating each Q&A separately), draw connections between findings where relevant, and cite paper titles inline when referencing specific findings. Do not fabricate any information beyond what's given below.

{qa_text}

Write the literature review:"""

    response = llm_synth.invoke(prompt)
    return response.content