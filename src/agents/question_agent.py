from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',"..")))
from core.vector_store import collection 
from core.config import GROQ_API

load_dotenv()
llm_planner = ChatGroq(
    api_key=GROQ_API,
    model="openai/gpt-oss-120b",    
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

def retrieve_chunks(question, n_results=4):
    results = collection.query(query_texts=[question], n_results=n_results)
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    return list(zip(docs, metas))

# ReAct agent for answering questions
llm_answerer = ChatGroq(api_key=GROQ_API, model="openai/gpt-oss-120b", temperature=0)

def answer_question(question, max_attempts=2):
    n_results = 4
    for attempt in range(max_attempts):
        retrieved = retrieve_chunks(question, n_results=n_results)
        if not retrieved:
            return {"question": question, "answer": "No relevant information found in the corpus.", "sources": []}

        context = "\n\n".join([f"[Source: {meta['title']}]\n{doc}" for doc, meta in retrieved])

        prompt = f"""Answer the following question using ONLY the context provided below. If the context is insufficient to answer, say "INSUFFICIENT_CONTEXT" instead of guessing.

Question: {question}

Context:
{context}

Answer (cite source titles where relevant):"""

        response = llm_answerer.invoke(prompt)
        answer = response.content.strip()

        if "INSUFFICIENT_CONTEXT" not in answer:
            sources = list(set(meta["title"] for _, meta in retrieved))
            return {"question": question, "answer": answer, "sources": sources}

        # ReAct step: reasoning that context was insufficient -> retrieve more broadly
        print(f"↻ Insufficient context for '{question}', retrying with more chunks...")
        n_results += 3

    return {"question": question, "answer": "Insufficient information found after retries.", "sources": []}

# Orchestrate overall questions
def retrieval_agent(topic):
    questions = formulate_questions(topic)
    print(f"📋 Formulated {len(questions)} questions")

    qa_pairs = []
    for q in questions:
        print(f"\n🔎 Answering: {q}")
        result = answer_question(q)
        qa_pairs.append(result)
        print(f"✓ {len(result['sources'])} sources used")

    return qa_pairs

