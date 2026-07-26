from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
llm_planner = ChatGroq(
    api_key=os.getenv("GROQ_API"),
    model="llama-3.3-70b-versatile",    
    temperature=0.3,
)
print(f"Planner LLM loaded successfully : {llm_planner.model_name}")

def formulate_questions(topic, n=5):
    prompt = f"""You are a research assistant preparing a literature review. Given this topic, break it down into {n} specific, answerable sub-questions that together would form a comprehensive literature review. Each question should target a distinct aspect (e.g., methods used, datasets, key findings, limitations, gaps).

Return ONLY a numbered list, no explanation.

Topic: {topic}"""
    response = llm_planner.invoke(prompt)
    lines = [line.strip() for line in response.content.split("\n") if line.strip()]
    questions = []
    for line in lines:
        if line[0].isdigit():
            q = line.split(".", 1)[-1].strip()
            questions.append(q)
    return questions[:n] if questions else [topic]

from core.vector_store import collection 

def retrieve_chunks(question, n_results=4):
    results = collection.query(query_texts=[question], n_results=n_results)
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    return list(zip(docs, metas))