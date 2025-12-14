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


def get_filtered_results(results, selected_from, selected_to, selected_years):
    """Apply all filters to results"""
    filtered = results
    
    if selected_from:
        filtered = [r for r in filtered if r['from'] in selected_from]
    if selected_to:
        filtered = [r for r in filtered if r['to'] in selected_to]
    if selected_years:
        filtered = [r for r in filtered if r.get('date') and int(r['date'].split('-')[0]) in selected_years]
    
    return filtered


def clear_all():
    """Callback to clear all session state"""
    st.session_state['last_results'] = None
    st.session_state['search_executed'] = False
    st.session_state['search_query'] = ''
    st.session_state['from_filter'] = []
    st.session_state['to_filter'] = []
    st.session_state['year_filter'] = []


def main():
    st.title("Ask Your Emails")
    st.caption("Semantic email search over 14,929 Enron emails")

    with st.spinner("Loading "):
        search_engine = load_search_engine()
        rag = load_rag()

    render_unified_search(search_engine, rag)


def render_unified_search(search_engine, rag):
    # Initialize session state
    if 'last_results' not in st.session_state:
        st.session_state['last_results'] = None
    if 'search_executed' not in st.session_state:
        st.session_state['search_executed'] = False
    
    facets = search_engine.get_facet_values()

    # Example queries from evaluation dataset
    st.markdown("**Try these real questions from the dataset:**")
    example_col1, example_col2, example_col3 = st.columns(3)

    with example_col1:
        if st.button("Taliban vs. Texans"):
            st.session_state['search_query'] = "In the 'Taliban vs. Texans' joke email, how many Texans are claimed to be better than one thousand Taliban?"

    with example_col2:
        if st.button("Pigging amendment"):
            st.session_state['search_query'] = "In Lindy Donoho's October 25, 2001 email about 'Installation of Pigging Facilities,' who tendered the amendment first and who tendered it twice after?"

    with example_col3:
        if st.button("NWPL ownership"):
            st.session_state['search_query'] = "In that same Lindy Donoho pigging email, what will NWPL's ownership interest be if they execute the amendment?"

    st.markdown("---")

    # Main search input
    st.text_area(
        "Search or ask a question",
        key="search_query",
        placeholder="Enter keywords to search, or ask a question like 'What were the concerns about electricity prices?'",
        height=100
    )

    st.markdown("---")

    # Filters and options
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

    with col1:
        top_k = st.number_input("Results", min_value=5, max_value=50, value=10, step=5)

    # Get all available options from facets (static, not updated)
    all_from = sorted(facets.get('from', []))
    all_to = sorted(facets.get('to', []))
    all_years = sorted(facets.get('years', []))

    with col2:
        selected_from = st.multiselect(
            "From",
            all_from,
            key="from_filter",
            help="Filter by sender"
        )

    with col3:
        selected_to = st.multiselect(
            "To",
            all_to,
            key="to_filter",
            help="Filter by recipient"
        )

    with col4:
        selected_years = st.multiselect(
            "Year",
            all_years,
            key="year_filter",
            help="Filter by year"
        )

    st.markdown("---")

    # RAG options
    use_qa = False
    num_context = config.MAX_CONTEXT_EMAILS
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

    st.markdown("---")

    # Button row
    col1, col2 = st.columns([1, 2])

    with col1:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    with col2:
        st.button("Clear", use_container_width=True, on_click=clear_all)

    st.markdown("---")

    # Handle search button
    if search_clicked:
        query = st.session_state.get('search_query', '').strip()
        
        if not query:
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching "):
                results = search_engine.search(
                    query=query,
                    top_k=top_k,
                    users=None,
                    date_year_min=None,
                    date_year_max=None
                )
            
            # Apply filters to search results
            filtered_results = get_filtered_results(results, selected_from, selected_to, selected_years)
            
            st.session_state['last_results'] = filtered_results
            st.session_state['search_executed'] = True

    # Display results if search was executed
    if st.session_state['search_executed'] and st.session_state['last_results'] is not None:
        results = st.session_state['last_results']

        if results:
            # Generate AI answer if requested
            if rag and use_qa:
                with st.spinner("Generating answer "):
                    response = rag.answer_question(
                        st.session_state.get('search_query', ''),
                        results,
                        max_emails=num_context
                    )

                    st.markdown("### AI Answer")
                    st.markdown(response['answer'])

                    st.markdown("---")
                    st.markdown(f"**Based on {response['num_sources']} source emails**")
                    st.markdown("---")

            # Show search results
            st.success(f"Found {len(results)} results")

            for i, result in enumerate(results, 1):
                with st.expander(f"{i}. {result['subject']} (score: {result['score']:.2f})"):
                    st.markdown(f"**From:** {result['from']}")
                    st.markdown(f"**To:** {result['to']}")
                    st.markdown(f"**Date:** {result.get('date', 'N/A')}")
                    st.markdown(f"**Path:** `{result['path']}`")
                    st.markdown("---")
                    st.text(result['body'])
        else:
            st.warning("No results match your search and filters.")


if __name__ == "__main__":
    main()
