# Calling LLM and setup API
import os
import langchain_openrouter
import langchain_groq
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter
import requests
import feedparser



# Run API key from .env file
load_dotenv()
GROQ_API = os.getenv("GROQ_API")
OPEN_ROUTER_API = os.getenv("OPEN_ROUTER_API")

if not GROQ_API:
    raise ValueError("API keys for GROQ must be set in the .env file.")
else:
    print("GROQ API key is set.")
if not OPEN_ROUTER_API:
    raise ValueError("API keys for OPEN_ROUTER must be set in the .env file.")
else:
    print("OPEN_ROUTER API key is set.")
    
llm_search = ChatGroq(model="openai/gpt-oss-20b", 
                      api_key=GROQ_API, 
                      temperature=0, 
                      max_tokens=1000)


def generate_search_queries(topic, n=3):
    prompt = f"""You are a research assistant. Generate {n} search queries for the topic: "{topic}".
    Each query should be concise and relevant to the topic. Return the queries as a list of strings.
    Return only numbered list of queries without any additional text or explanation.
    
    Topic:{topic}
    """
    response = llm_search.invoke(prompt) 
    lines = [line.strip() for line in response.content.split("\n") if line.strip()]
    queries =[]
    for line in lines:
        if line[0].isdigit():
            q = line.split('.',1)[1].strip()
            queries.append(q)
        return queries[:n] if queries else [topic]
    
def get_arxiv_pdf_url(entry): # For getting a proper PDF URL from the arXiv entry
    # entry.id looks like: http://arxiv.org/abs/2506.01923v1
    arxiv_id = entry.id.split("/abs/")[-1]
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
def search_arxiv(query, max_results=5):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    try:
        response = requests.get(base_url, params=params, timeout=20)
        feed = feedparser.parse(response.text)
    except Exception as e:
        print(f"arxiv search failed for query '{query}': {e}")
        return []
    
    results = []
    for entry in feed.entries:
        result = {
            "title": entry.title.replace('\n', ' ').strip(), # Remove newlines and extra spaces
            "authors":[a.name for a in entry.authors],
            "abstract": entry.summary.replace('\n', ' ').strip(), 
            "pdf_url": entry.published,
            "source": "arxiv"
        }
        results.append(result)
    return results
def search_semantic_scholar(query, max_results=5):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,authors,url,openAccessPdf,year"
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
    except Exception as e:
        print(f"Semantic Scholar search failed for '{query}': {e}")
        return []

    results = []
    for paper in data.get("data", []):
        results.append({
            "title": paper.get("title"),
            "authors": [a["name"] for a in paper.get("authors", [])],
            "abstract": paper.get("abstract") or "",
            "pdf_url": paper.get("openAccessPdf", {}).get("url") if paper.get("openAccessPdf") else None,
            "published": paper.get("year"),
            "source": "semantic_scholar"
        })
    return results

def search_agent(topic, max_results_per_query=5,display_results=True):
    queries = generate_search_queries(topic)
    print(f"Generated queries: {queries}")
    
    all_results = []
    for q in queries:
        all_results.extend(search_arxiv(q, max_results_per_query))
        # all_results.extend(search_semantic_scholar(q, max_results_per_query))
    
    seen = set()
    unique_results = []
    for r in all_results:
        if not r.get("title"):
            continue
        key = r["title"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    print(f"Found {len(unique_results)} unique papers")
    
    if display_results and unique_results:
        print("\n Search results:")
        print("=" * 60)
        for i, paper in enumerate(unique_results, 1):
            title = paper.get("title","No Title")
            authors = paper.get("authors",["Unknows"])[:2] # First 2 authors only
            year = paper.get("Year","N/A")
            
            print(f"{i:2}. Title: {title}")
            print(f"    Authors: {', '.join(authors) if authors else 'Unknown'}")
            print(f"    Year: {year}")
            print("-" * 60)
    return unique_results

if __name__ == "__main__":
    user_topic = "Tuberculosis detection by AI using chest X-ray images"
    max_papers = 5
    # user_topic = input("Enter a research topic: ")
    # max_papers = int(input("Enter the maximum number of papers to retrieve per query: "))
    papers = search_agent(user_topic, max_results_per_query=max_papers, display_results=True)