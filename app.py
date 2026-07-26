import streamlit as st
from src.agents.search_agent import search_agent
from src.agents.analys_agent import analysis_agent
from src.agents.question_agent import retrieval_agent
from src.agents.synthesis_agent import synthesis_agent

st.set_page_config(page_title="Literature Review Agent", layout="wide")
st.title("Multi-agent Literature Review Agent")

topic = st.text_input("Enter a research topic:")
st.subheader("Adjust parameters:")
effort_level = st.slider("Higher levels search for more papers per query, improving thoroughness but taking longer.", min_value=2, max_value=30, value=5)
st.caption(f"Level {effort_level} will search up to {effort_level} papers per query. Higher levels may take longer to process.")

if st.button("Generate Literature Review") and topic:
    with st.spinner("Searching for papers..."):
        papers = search_agent(topic, effort_level, display_results=False)
        st.success(f"Found {len(papers)} papers")

    with st.spinner("Downloading, chunking, embedding..."):
        analysis_agent(papers, topic)
    st.success("Papers embedded into vector database")

    with st.spinner("Formulating questions and retrieving answers..."):
        qa_pairs = retrieval_agent(topic)
    st.success(f"Answered {len(qa_pairs)} sub-questions")

    with st.spinner("Synthesizing final review..."):
        final_review = synthesis_agent(topic, qa_pairs)

    st.markdown("## Generated Literature Review")
    st.markdown(final_review)

    with st.expander("View sources used"):
        for qa in qa_pairs:
            st.markdown(f"**Q:** {qa['question']}")
            st.markdown(f"**Sources:** {', '.join(qa['sources'])}")