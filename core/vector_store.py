import chromadb
from chromadb.utils import embedding_functions 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "..", "data","vector_db")
VECTOR_DB_PATH = os.path.abspath(VECTOR_DB_PATH)


# client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
client = chromadb.Client()
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="literature_review", embedding_function=embedding_fn)

