"""
Ask Your Emails - Streamlit Web Application

Semantic email search with faceted navigation.
INFO 202 Final Project
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from search import EmailSearchEngine
from rag import EmailRAG
import config

# Page config
st.set_page_config(
    page_title="Ask Your Emails",
    page_icon="📧",
    layout="wide"
)


@st.cache_resource
def load_search_engine():
    engine = EmailSearchEngine()
    engine.initialize()
    return engine


@st.cache_resource
def load_rag():
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", config.ANTHROPIC_API_KEY)
        return EmailRAG(api_key=api_key)
    except (ValueError, KeyError):
        return None


def main():
    st.title("📧 Ask Your Emails")
    st.caption("Semantic email search over 14,929 Enron emails")

    with st.spinner("Loading..."):
        search_engine = load_search_engine()
        rag = load_rag()

    render_unified_search(search_engine, rag)


def render_unified_search(search_engine, rag):
    facets = search_engine.get_facet_values()

    # Example queries
    st.markdown("**Try these examples:**")
    example_col1, example_col2, example_col3 = st.columns(3)

    with example_col1:
        if st.button("California energy crisis"):
            st.session_state['search_query'] = "California energy crisis"

    with example_col2:
        if st.button("What were the main trading strategies?"):
            st.session_state['search_query'] = "What were the main trading strategies?"

    with example_col3:
        if st.button("Meeting schedules and appointments"):
            st.session_state['search_query'] = "Meeting schedules and appointments"

    st.markdown("---")

    # Main search input
    st.text_area(
        "Search or ask a question",
        key="search_query",
        placeholder="Enter keywords to search, or ask a question like 'What were the concerns about electricity prices?'",
        height=100
    )

    # Filters and options
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        selected_users = st.multiselect("Filter by Users", facets['users'], key="search_users")

    with col2:
        selected_folders = st.multiselect("Filter by Folders", facets['folders'], key="search_folders")

    with col3:
        top_k = st.number_input("Results", min_value=5, max_value=50, value=10, step=5)

        if rag:
            use_qa = st.checkbox("Generate AI answer", value=False, help="Use RAG to generate a comprehensive answer")
            if use_qa:
                num_context = st.number_input(
                    "Context emails",
                    min_value=1,
                    max_value=10,
                    value=config.MAX_CONTEXT_EMAILS,
                    key="num_context"
                )

    query = st.session_state.get('search_query', '')

    if query:
        with st.spinner("Searching..."):
            results = search_engine.search(
                query=query,
                top_k=top_k,
                users=selected_users if selected_users else None,
                folders=selected_folders if selected_folders else None
            )

        if results:
            # Generate AI answer if requested
            if rag and 'use_qa' in locals() and use_qa:
                with st.spinner("Generating answer..."):
                    response = rag.answer_question(query, results, max_emails=num_context)

                    st.markdown("### AI Answer")
                    st.markdown(response['answer'])

                    st.markdown("---")
                    st.markdown(f"**Based on {response['num_sources']} source emails**")
                    st.markdown("---")

            # Show search results
            st.success(f"Found {len(results)} results")

            for result in results:
                with st.expander(f"{result['rank']}. {result['subject']} (score: {result['score']:.2f})"):
                    st.markdown(f"**From:** {result['from']}")
                    st.markdown(f"**To:** {result['to']}")
                    st.markdown(f"**Folder:** {result['folder']}")
                    st.markdown("---")
                    st.text(result['body'])
        else:
            st.warning("No results found.")


if __name__ == "__main__":
    main()
