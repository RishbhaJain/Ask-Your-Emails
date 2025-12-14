#!/usr/bin/env python3
"""Build vector index from processed emails."""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import config
from embedder import EmailEmbedder
from indexer import EmailVectorIndex


def main():

    emails_path = config.PROCESSED_DIR / "emails_subset.parquet"
    print(f"Loading emails from {emails_path}")
    emails_df = pd.read_parquet(emails_path)
    print(f"Loaded {len(emails_df)} emails")

    print("\nInitializing embedding model")
    embedder = EmailEmbedder()

    print("Generating embeddings")
    embeddings = embedder.embed_emails(emails_df)

    embeddings_path = config.PROCESSED_DIR / "email_embeddings.npy"
    print(f"Saving embeddings to {embeddings_path}")
    embedder.save_embeddings(embeddings, str(embeddings_path))

    print("\nBuilding FAISS index")
    index = EmailVectorIndex()
    index.build_index(emails_df, embeddings)
    index.save_index()

    # Quick test
    print("\nTesting search:")
    test_queries = [
        "California energy crisis",
        "meeting schedule next week",
        "power trading deals"
    ]

    for query in test_queries:
        index.test_search(query, embedder, top_k=3)
        print()

    facets = index.get_facet_values()
    print(f"\nIndexed {len(emails_df)} emails")
    print(f"  {len(facets['users'])} users")
    print(f"  {len(facets['folders'])} folders")
    print(f"  {min(facets['years'])}-{max(facets['years'])}")
    print(f"\nIndex saved to {index.index_path}")


if __name__ == "__main__":
    main()
