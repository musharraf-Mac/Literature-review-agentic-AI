import os
import requests
from pypdf import PdfReader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from .search_agent import search_agent
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',"..")))
from core.vector_store import collection
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import GROQ_API
import tempfile
import streamlit as st

def get_papers_dir():
    try:        
        # Running on Streamlit Cloud (or any environment) — use a safe temp dir
        return tempfile.mkdtemp(prefix="papers_")
    except Exception:
        # Local fallback — keep your normal data/papers folder for dev convenience
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(BASE_DIR, "..", "..", "data", "papers")
        path = os.path.abspath(path)
        os.makedirs(path, exist_ok=True)
        return path
    
SAVE_DIR = get_papers_dir()

# Download PDF and extract text
def download_pdf(pdf_url, save_dir=SAVE_DIR,filename=None):
    if not pdf_url or not str(pdf_url).startswith("http"):
        print(f"Invalid URL: {pdf_url}")
        return None
    os.makedirs(save_dir, exist_ok=True)
    filename = filename or pdf_url.split("/")[-1].replace(".pdf","") + ".pdf"
    filepath = os.path.join(save_dir, filename)
    
    if os.path.exists(filepath):
        print(f"Download skipped: {filename} already exists.")
        return filepath
    try:
        response = requests.get(pdf_url, timeout=30)
        if response.status_code == 200 and response.headers.get("content-type", "").startswith("application/pdf"):
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
            return filepath
        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}, Content-Type: {response.headers.get('content-type')}")
            return None
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return None

def extract_text(filepath):
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {filepath}: {e}")
        return ""
    
load_dotenv()
GROQ_API = os.getenv("GROQ_API")
llm_relevance = ChatGroq(model="openai/gpt-oss-20b", api_key=GROQ_API, temperature=0, max_tokens=1000)

def is_relevant(abstract, topic):
    if not abstract:
        print("Empty abstract, skipping")
        return False
    prompt = f"""Topic: {topic}
    Abstract: {abstract}
    Is this paper relavant to the topic? Answer with 'Yes' or 'No' only.
    """
    response = llm_relevance.invoke(prompt)
    print("Raw response:", repr(response.content))
    return "yes" in response.content.lower().strip()

# Chunking and storing
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def store_chunks(chunks, paper_meta, chunk_id_prefix):
    ids = [f"{chunk_id_prefix}_{i}" for i in range(len(chunks))]
    metadatas = [{"title": paper_meta["title"], "source": paper_meta["source"]} for _ in chunks]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
# Analysis agent
def analysis_agent(papers, topic):
    stored_count = 0
    for idx, paper in enumerate(papers):
        filepath = download_pdf(paper.get("pdf_url"), filename=f"paper_{idx}.pdf")
        if not filepath:
            print(f"Skipped (no PDF): {paper['title']}")
            continue

        text = extract_text(filepath)
        if not text:
            print(f"Skipped (no extractable text): {paper['title']}")
            continue

        chunks = chunk_text(text)
        store_chunks(chunks, paper, chunk_id_prefix=f"paper_{idx}")
        stored_count += 1
        print(f"Stored: {paper['title']} ({len(chunks)} chunks)")

    print(f"\n{stored_count} papers embedded into vector DB")
