#!/usr/bin/env python3
"""
Test script to verify RAG/Q&A functionality
"""
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from search import EmailSearchEngine
from rag import EmailRAG
import config

def test_rag_questions():
    """Test RAG system with actual Q&A pairs"""

    print("=" * 80)
    print("RAG/Q&A SYSTEM TEST")
    print("=" * 80)
    print()

    # Load Q&A pairs
    print("Loading Q&A evaluation dataset...")
    qa_df = pd.read_csv('data/processed/eval_qa_pairs.csv')
    print(f"✓ Loaded {len(qa_df)} Q&A pairs\n")

    # Initialize search engine
    print("Initializing search engine...")
    search_engine = EmailSearchEngine()
    print("✓ Search engine ready\n")

    # Initialize RAG
    print("Initializing RAG system...")
    try:
        rag = EmailRAG()
        print("✓ RAG system ready\n")
    except Exception as e:
        print(f"✗ RAG initialization failed: {e}")
        print("\nWill test search functionality only...\n")
        rag = None

    # Test questions (spread across the dataset)
    test_indices = [0, 50, 100, 200, 300]

    for idx in test_indices:
        row = qa_df.iloc[idx]
        question = row['question']
        ground_truth = row['answer']
        user = row['user']

        print("=" * 80)
        print(f"TEST CASE #{idx}")
        print("=" * 80)
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"User: {user}")
        print()

        # Step 1: Test retrieval
        print("STEP 1: Semantic Search for Relevant Emails")
        print("-" * 80)
        results = search_engine.search(question, top_k=20)
        print(f"✓ Retrieved {len(results)} emails")

        if len(results) > 0:
            print(f"\nTop 3 Results:")
            for i, email in enumerate(results[:3], 1):
                print(f"  {i}. Score: {email['score']:.4f} | User: {email['user']}")
                print(f"     Subject: {email['subject'][:80]}")
            print()
        else:
            print("✗ No results found!\n")
            continue

        # Step 2: Test RAG answer generation (if available)
        if rag is not None:
            print("STEP 2: RAG Answer Generation")
            print("-" * 80)
            try:
                response = rag.answer_question(question, results, max_emails=5)
                rag_answer = response['answer']
                num_sources = response['num_sources']

                print(f"✓ Generated answer using {num_sources} source emails")
                print()
                print("RAG Answer:")
                print(rag_answer)
                print()

                # Simple comparison
                print("COMPARISON:")
                print("-" * 80)
                print(f"Ground Truth: {ground_truth}")
                print(f"RAG Answer:   {rag_answer}")
                print()

                # Check if key terms match
                gt_lower = ground_truth.lower()
                ra_lower = rag_answer.lower()

                # Extract potential key facts from ground truth
                key_terms = []
                if 'june 24, 2002' in gt_lower:
                    key_terms.append('june 24, 2002')
                if 'account holder' in gt_lower:
                    key_terms.append('account holder')
                if '382621' in gt_lower:
                    key_terms.append('382621')
                if 'asian options on gas' in gt_lower:
                    key_terms.append('asian options on gas')
                if 'too low' in gt_lower:
                    key_terms.append('too low')

                if key_terms:
                    matches = [term for term in key_terms if term in ra_lower]
                    print(f"Key Term Match: {len(matches)}/{len(key_terms)} terms found")
                    if matches:
                        print(f"Matched: {matches}")
                    if len(matches) < len(key_terms):
                        print(f"Missing: {[t for t in key_terms if t not in matches]}")

            except Exception as e:
                print(f"✗ RAG failed: {e}")
                import traceback
                traceback.print_exc()

        print()
        print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_rag_questions()
