import torch
import os
import requests
from pypdf import PdfReader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions 
from search_agent import search_agent

# Download PDF and extract text
def download_pdf(pdf_url, save_dir="data/papers",filename=None):
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

# Chunk test
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


# DataBase creation
client = chromadb.PersistentClient(path="../../data/vector_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")

collection = client.get_or_create_collection(name="literature_review", embedding_function=embedding_fn)

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
            print(f"✗ Skipped (no PDF): {paper['title']}")
            continue

        text = extract_text(filepath)
        if not text:
            print(f"✗ Skipped (no extractable text): {paper['title']}")
            continue

        chunks = chunk_text(text)
        store_chunks(chunks, paper, chunk_id_prefix=f"paper_{idx}")
        stored_count += 1
        print(f"✓ Stored: {paper['title']} ({len(chunks)} chunks)")

    print(f"\n{stored_count} papers embedded into vector DB")

# Test the analysis agent
topic = "cross-generator generalization AI-Generated text detection"
papers = search_agent(topic, max_results_per_query=5, display_results=True)
analysis_agent(papers, topic)
for p in papers[:5]:
    print(p["title"], "-> abstract length:", len(p.get("abstract", "")))