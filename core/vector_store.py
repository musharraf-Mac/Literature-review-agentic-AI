import torch
import chromadb
from chromadb.utils import embedding_functions 
from langchain_text_splitters import RecursiveCharacterTextSplitter

client = chromadb.PersistentClient(path="../../data/vector_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")

collection = client.get_or_create_collection(name="literature_review", embedding_function=embedding_fn)

    