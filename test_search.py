#!/usr/bin/env python3
"""
Interactive Search Testing Script

Test the semantic search functionality with custom queries.

Usage:
    python test_search.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import config
from embedder import EmailEmbedder
from indexer import EmailVectorIndex


def print_result(result, show_body=False):
    """Pretty print a search result."""
    print(f"\n{'='*70}")
    print(f"[Rank {result['rank']}] Relevance Score: {result['score']:.3f}")
    print(f"{'='*70}")
    print(f"📧 Subject: {result['subject']}")
    print(f"👤 From: {result['from']}")
    print(f"👥 To: {result['to'][:60]}{'...' if len(result['to']) > 60 else ''}")
    print(f"📁 Folder: {result['folder']}")
    print(f"👨‍💼 User: {result['user']}")
    print(f"📅 Year: {result['date_year']}")

    if show_body:
        print(f"\n📄 Body Preview:")
        body_preview = result['body'][:300]
        print(f"{body_preview}{'...' if len(result['body']) > 300 else ''}")


def test_basic_search():
    """Test basic semantic search."""
    print("\n" + "🔍 " + "="*68)
    print("TESTING BASIC SEMANTIC SEARCH")
    print("="*70 + "\n")

    # Load the index and embedder
    print("Loading vector index and embedding model...")
    index = EmailVectorIndex()
    index.load_index()

    embedder = EmailEmbedder()

    print("\n✅ Index and model loaded successfully!\n")

    # Test queries
    test_queries = [
        "California energy crisis and power outages",
        "meeting schedule for next quarter",
        "natural gas prices and trading",
        "legal issues and contracts",
        "employee benefits and compensation"
    ]

    print("Running 5 test queries...\n")

    for query in test_queries:
        print("\n" + "🔎 " + "-"*68)
        print(f"QUERY: \"{query}\"")
        print("-"*70)

        # Generate query embedding
        query_embedding = embedder.embed_text(query)

        # Search
        results = index.search(query_embedding, top_k=3)

        print(f"\nFound {len(results)} results:")

        for result in results:
            print_result(result, show_body=False)

        input("\n⏎ Press Enter to continue to next query...")


def test_faceted_search():
    """Test search with faceted filters."""
    print("\n" + "🔍 " + "="*68)
    print("TESTING FACETED SEARCH (with filters)")
    print("="*70 + "\n")

    # Load the index and embedder
    index = EmailVectorIndex()
    index.load_index()
    embedder = EmailEmbedder()

    # Get available facet values
    facets = index.get_facet_values()

    print("Available facets:")
    print(f"  Users: {len(facets['users'])} unique users")
    print(f"  Folders: {len(facets['folders'])} unique folders")
    print(f"  Years: {facets['years']}")
    print()

    # Example 1: Filter by folder
    print("Example 1: Search in 'sent_items' folder only")
    print("-"*70)
    query = "power trading deal"
    print(f"Query: \"{query}\"")

    query_embedding = embedder.embed_text(query)
    results = index.search(
        query_embedding,
        top_k=3,
        filters={'folders': ['sent_items']}
    )

    print(f"\nResults (filtered to sent_items folder):")
    for result in results:
        print_result(result, show_body=False)

    input("\n⏎ Press Enter to continue...")

    # Example 2: Filter by user
    print("\n\nExample 2: Search emails from specific users")
    print("-"*70)

    # Show some example users
    print("Sample users:", facets['users'][:5])

    query = "meeting"
    print(f"\nQuery: \"{query}\"")
    print(f"Filter: First 3 users from the dataset")

    query_embedding = embedder.embed_text(query)
    results = index.search(
        query_embedding,
        top_k=3,
        filters={'users': facets['users'][:3]}
    )

    print(f"\nResults:")
    for result in results:
        print_result(result, show_body=False)

    input("\n⏎ Press Enter to continue...")


def test_custom_query():
    """Interactive custom query testing."""
    print("\n" + "🔍 " + "="*68)
    print("INTERACTIVE CUSTOM SEARCH")
    print("="*70 + "\n")

    # Load the index and embedder
    index = EmailVectorIndex()
    index.load_index()
    embedder = EmailEmbedder()

    facets = index.get_facet_values()

    print("You can now search the email archive!")
    print("Type 'quit' to exit.\n")

    while True:
        # Get query from user
        query = input("\n🔎 Enter your search query: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break

        if not query:
            continue

        # Ask for number of results
        try:
            top_k_input = input("How many results? (default 5): ").strip()
            top_k = int(top_k_input) if top_k_input else 5
        except ValueError:
            top_k = 5

        # Ask if they want to filter
        use_filter = input("Apply filters? (y/n, default n): ").strip().lower()

        filters = None
        if use_filter == 'y':
            print(f"\nAvailable folders (showing first 10): {facets['folders'][:10]}")
            folder_input = input("Filter by folder (comma-separated, or press Enter to skip): ").strip()

            if folder_input:
                folders = [f.strip() for f in folder_input.split(',')]
                filters = {'folders': folders}

        # Perform search
        print(f"\n🔍 Searching for: \"{query}\"")
        if filters:
            print(f"📁 Filters: {filters}")
        print("-"*70)

        query_embedding = embedder.embed_text(query)
        results = index.search(query_embedding, top_k=top_k, filters=filters)

        if not results:
            print("\n❌ No results found. Try different filters or query.")
            continue

        print(f"\n✅ Found {len(results)} results:\n")

        for result in results:
            print_result(result, show_body=True)

        # Ask if they want to see more details
        detail = input("\n📧 Enter rank number to see full email (or press Enter to continue): ").strip()

        if detail.isdigit():
            rank = int(detail)
            if 1 <= rank <= len(results):
                result = results[rank - 1]
                print("\n" + "="*70)
                print(f"FULL EMAIL - Rank {rank}")
                print("="*70)
                print(f"\n📧 Subject: {result['subject']}")
                print(f"👤 From: {result['from']}")
                print(f"👥 To: {result['to']}")
                print(f"📁 Folder: {result['folder']}")
                print(f"📅 Year: {result['date_year']}")
                print(f"\n{'='*70}")
                print("FULL BODY:")
                print("="*70)
                print(result['body'])
                print("="*70)

                input("\n⏎ Press Enter to continue...")


def main():
    """Main menu."""
    print("\n" + "="*70)
    print(" " * 20 + "THE INBOX CONDUCTOR")
    print(" " * 18 + "Search Engine Testing")
    print("="*70)

    while True:
        print("\n📋 Choose a test mode:")
        print("  1. Basic Semantic Search (5 example queries)")
        print("  2. Faceted Search (with filters)")
        print("  3. Interactive Custom Search")
        print("  4. Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == '1':
            test_basic_search()
        elif choice == '2':
            test_faceted_search()
        elif choice == '3':
            test_custom_query()
        elif choice == '4':
            print("\nGoodbye! 👋\n")
            break
        else:
            print("\n❌ Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
