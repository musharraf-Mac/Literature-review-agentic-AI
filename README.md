# Multi-Agent Literature Review System

An agentic AI application that automates literature review generation for a given research topic — searching academic sources, filtering and embedding relevant papers, formulating research sub-questions, retrieving grounded answers via RAG, and synthesizing a coherent literature review.

Built for IT41043 — Intelligent Systems (Agentic AI), Horizon Campus.

**Live demo:** https://literature-review-agentic-ai.streamlit.app

---

## 1. Project Description

Reviewing literature for an active research topic is slow: finding relevant papers, reading them, and connecting findings into a coherent narrative takes hours of manual work. This system automates that pipeline end-to-end using four cooperating agents, grounded in real, freshly-retrieved papers rather than an LLM's frozen training memory.

**Type:** Option B — Research support tool
**Domain:** Supports ongoing research on TB detection XAI

## 3. Agent-to-Agent Communication

Agents exchange structured Python dictionaries/lists (JSON-serializable) at each handoff, orchestrated sequentially from `app.py`:

| From → To | Payload |
|---|---|
| Search Agent → Analysis Agent | `List[dict]` — `{title, authors, abstract, pdf_url, published, source}` per paper |
| Analysis Agent → (Vector DB) | Chunked + embedded documents with metadata `{title, source}` |
| Question Agent → Retrieval Agent | `List[str]` — formulated sub-questions |
| Retrieval Agent → Synthesis Agent | `List[dict]` — `{question, answer, sources}` per sub-question |

### Sequence Diagram

```
User → SearchAgent: topic
SearchAgent → SearchAgent: generate_search_queries(topic)
SearchAgent → arXiv API: query
SearchAgent → SemanticScholar API: query
SearchAgent → AnalysisAgent: paper list

AnalysisAgent → AnalysisAgent: download_pdf(), extract_text(), chunk_text()
AnalysisAgent → VectorDB: store_chunks()

QuestionAgent → QuestionAgent: formulate_questions(topic)
QuestionAgent → RetrievalAgent: sub-questions

loop for each sub-question
    RetrievalAgent → VectorDB: retrieve_chunks(question)
    VectorDB → RetrievalAgent: relevant chunks
    RetrievalAgent → RetrievalAgent: check sufficiency (ReAct)
    alt insufficient context
        RetrievalAgent → VectorDB: retrieve_chunks(question, broader)
    end
    RetrievalAgent → SynthesisAgent: {question, answer, sources}
end

SynthesisAgent → SynthesisAgent: synthesize_review()
SynthesisAgent → SynthesisAgent: critique_review() [self-reflection]
SynthesisAgent → User: final literature review
```

---

## 4. Agentic Design Patterns Used

| # | Pattern | Where implemented |
|---|---|---|
| 1 | **Tool-use** | `agents/search_agent.py` — LLM-driven query generation, calling external arXiv & Semantic Scholar APIs as tools |
| 2 | **Reflection / self-critique** | `agents/analys_agent.py` (relevance filtering — optional, see limitations) and `agents/synthesis_agent.py` — `critique_review()` reviews and improves its own draft before finalizing |
| 3 | **Planning / task-decomposition** | `agents/question_agent.py` — `formulate_questions()` breaks the broad topic into targeted, independently-answerable sub-questions |
| 4 | **ReAct (Reason + Act loop)** | `agents/question_agent.py` — `answer_question()` retrieves, reasons about whether context is sufficient, and re-retrieves with a broader query if not, before answering |
| 5 | **Orchestrator–worker** | `app.py` — coordinates all four agents in sequence, passing structured state between them |

---

## 5. Model Selection Strategy

| Sub-task | Model (provider) | Why chosen |
|---|---|---|
| Query generation (Search Agent) | `openai/gpt-oss-20b` (Groq) | Low latency, cheap; simple generative task doesn't need heavy reasoning |
| Relevance classification (Analysis Agent) | `openai/gpt-oss-20b` (Groq) | Fast binary classification over many papers; cost matters more than depth here |
| Question decomposition (Question Agent) | `openai/gpt-oss-120b` (Groq) | Needs stronger reasoning to produce distinct, well-scoped sub-questions rather than overlapping ones |
| Retrieval answering (Retrieval Agent) | `openai/gpt-oss-120b` (Groq) | Needs reliable grounding and reasonable tool-calling stability during the ReAct retrieval loop |
| Final synthesis + critique (Synthesis Agent) | `openai/gpt-4o-mini` (OpenRouter) | Highest-stakes output — coherent long-form writing and self-critique justify the higher cost/latency of a stronger model |

---

## 6. RAG Pipeline

- **Corpus:** papers dynamically retrieved per query from arXiv and Semantic Scholar APIs (not a static pre-collected set) — the corpus is built fresh for whatever topic the user enters.
- **Chunking:** `RecursiveCharacterTextSplitter`, chunk size 1000 characters, overlap 200.
- **Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers).
- **Vector store:** ChromaDB (in-memory client for deployment stability on Streamlit Cloud).
- **Retrieval:** top-k similarity search per sub-question, with a ReAct-style retry using a broader `n_results` if the LLM judges retrieved context insufficient.

---

## 7. Effort Level / Search Depth Control

The UI includes a slider (2–30) controlling how many papers are searched per generated query. Higher levels increase thoroughness but proportionally increase runtime (search + download + chunk + embed all scale with paper count). If fewer papers than requested are found for a given query, the app surfaces this to the user directly rather than failing silently.

---

## 8. Setup Instructions

### Local development
```bash
git clone https://github.com/musharraf-Mac/Literature-review-agentic-AI
cd literature-review-agentic-ai
pip install -r requirements.txt --break-system-packages
```

Create a `.env` file in the project root:
```
GROQ_API=your_groq_key_here
OPEN_ROUTER_API=your_openrouter_key_here
```

Run:
```bash
streamlit run app.py
```

### Deployment (Streamlit Community Cloud)
1. Push to `main` on GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing to this repo, branch `main`, file `app.py`.
3. In the app's **Settings → Secrets**, add (TOML format):
```toml
GROQ_API = "your_groq_key_here"
OPEN_ROUTER_API = "your_openrouter_key_here"
```

---

## 9. Known Limitations

- **Relevance filtering in the Analysis Agent was simplified/deprioritized** in favor of relying on the Search Agent's topic-targeted queries, due to time constraints — could be reintroduced as a stricter reflection step given more development time.
- **Vector DB is in-memory and non-persistent** between sessions/restarts (deliberate choice for Streamlit Cloud filesystem stability) — each user query rebuilds the corpus fresh, so results may vary slightly between identical repeated queries depending on API result freshness.
- **Semantic Scholar's unauthenticated API has rate limits** — heavy use (especially at high "effort levels") may hit throttling; a registered API key mitigates but does not eliminate this.
- **Deduplication is exact-title-match only** — near-duplicate papers with minor title wording differences are not merged.
- **Dependent on third-party model availability** — Groq periodically deprecates model versions (encountered during development); model strings may need updating if Groq/OpenRouter retire the models listed above.
- **PDF text extraction quality varies** by source formatting; some papers may yield partial or no extractable text and are skipped gracefully.

---

## 10. Tech Stack

- **Orchestration:** Python, LangChain, LangGraph
- **LLM providers:** Groq (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`), OpenRouter
- **Vector DB:** ChromaDB + sentence-transformers (`all-MiniLM-L6-v2`)
- **Search APIs:** arXiv API, Semantic Scholar API
- **PDF processing:** pypdf
- **UI/Deployment:** Streamlit, Streamlit Community Cloud

---

## 11. Disclosures

Built with assistance from Claude (Anthropic) for debugging, architecture discussion, and code review during development. All design decisions, topic selection, and final implementation choices are my own.