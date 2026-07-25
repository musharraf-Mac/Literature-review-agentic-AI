import torch
import os
import requests
from pypdf import PdfReader
from langchain_groq import ChatGroq
from dotenv import load_dotenv

def download_pdf(pdf_url, save_dir="data/papers",filename=None):
    if not pdf_url:
        print("No PDF URL provided.")
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

