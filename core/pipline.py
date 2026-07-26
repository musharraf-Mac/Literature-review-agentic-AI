from agents.search_agent import search_agent
from agents.analys_agent import analysis_agent
from agents.question_agent import retrieval_agent
from agents.synthesis_agent import synthesis_agent

topic = "cross-generator generalization AI-generated text detection"

papers = search_agent(topic)
analysis_agent(papers, topic)
qa_pairs = retrieval_agent(topic)
final_review = synthesis_agent(topic, qa_pairs)

print(final_review)