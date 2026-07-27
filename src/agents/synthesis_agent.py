from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter 
import os
from core.config import OPEN_ROUTER_API


llm_synth = ChatOpenRouter(
    api_key=OPEN_ROUTER_API,
    model="openai/gpt-4o-mini",  
    temperature=0.3,  
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

# critique review
def critique_review(draft, topic):
    prompt = f"""Review this literature review draft for the topic "{topic}". Check for:
1. Any claims not clearly supported by the cited sources
2. Awkward transitions or repetition
3. Missing synthesis (just listing findings vs. connecting them)

If it's good as-is, return it unchanged. Otherwise, return an improved version.

Draft:
{draft}

Improved version:"""
    response = llm_synth.invoke(prompt)
    return response.content


def synthesis_agent(topic, qa_pairs):
    draft = synthesize_review(topic, qa_pairs)
    print("Draft generated, running self-critique...")
    final = critique_review(draft, topic)
    return final