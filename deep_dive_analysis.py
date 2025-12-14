#!/usr/bin/env python3
"""
Deep dive analysis of RAG retrieval failures
"""
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from search import EmailSearchEngine
from rag import EmailRAG
import config

def analyze_failed_case(qa_df, test_idx, search_engine):
    """Deep dive into a specific failed test case"""

    row = qa_df.iloc[test_idx]
    question = row['question']
    ground_truth = row['answer']
    user = row['user']
    correct_email = row['email']
    correct_path = row['path']

    print("=" * 80)
    print(f"DEEP DIVE: TEST CASE #{test_idx}")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Expected User: {user}")
    print(f"Expected Path: {correct_path}")
    print()

    # Show the correct email
    print("GROUND TRUTH EMAIL:")
    print("-" * 80)
    print(correct_email[:500])
    print("..." if len(correct_email) > 500 else "")
    print()

    # Test retrieval at different top_k values
    print("RETRIEVAL ANALYSIS:")
    print("-" * 80)

    for top_k in [5, 10, 20, 50]:
        results = search_engine.search(question, top_k=top_k)

        # Check if correct email is in results
        paths = [r['path'] for r in results]
        found_at = -1
        if correct_path in paths:
            found_at = paths.index(correct_path) + 1

        print(f"Top-{top_k}: {'✓' if found_at > 0 else '✗'} Correct email " +
              (f"found at position {found_at}" if found_at > 0 else "NOT found"))

    print()

    # Detailed look at top 10
    print("TOP 10 RETRIEVED EMAILS:")
    print("-" * 80)
    results = search_engine.search(question, top_k=10)

    for i, email in enumerate(results, 1):
        is_correct = "✓✓✓" if email['path'] == correct_path else ""
        print(f"{i}. Score: {email['score']:.4f} | User: {email['user']} {is_correct}")
        print(f"   Path: {email['path']}")
        print(f"   Subject: {email['subject'][:80]}")
        print(f"   Body preview: {email['body'][:150]}...")
        print()

    # Analyze why retrieval might have failed
    print("FAILURE ANALYSIS:")
    print("-" * 80)

    if len(results) > 0 and results[0]['path'] != correct_path:
        print("Issue: Wrong email ranked highest")
        print(f"Top result score: {results[0]['score']:.4f}")
        print(f"Top result user: {results[0]['user']}")
        print()

        # Check if correct user is in top results
        top_users = [r['user'] for r in results[:5]]
        if user in top_users:
            print(f"✓ Correct user '{user}' IS in top 5 results")
        else:
            print(f"✗ Correct user '{user}' NOT in top 5 results")
            print(f"  Top 5 users: {set(top_users)}")
        print()

        # Check semantic similarity between question and correct email
        print("Analyzing question-email overlap:")
        question_words = set(question.lower().split())
        email_words = set(correct_email.lower().split())
        overlap = question_words & email_words
        print(f"  Question words: {len(question_words)}")
        print(f"  Email words: {len(email_words)}")
        print(f"  Overlapping words: {len(overlap)}")
        if len(overlap) > 0:
            print(f"  Common words: {list(overlap)[:10]}")

    print()
    print()


def main():
    """Run deep dive analysis on failed cases"""

    print("=" * 80)
    print("RAG SYSTEM - DEEP DIVE ANALYSIS")
    print("=" * 80)
    print()

    # Load Q&A pairs
    qa_df = pd.read_csv('data/processed/eval_qa_pairs.csv')

    # Initialize search engine
    search_engine = EmailSearchEngine()

    # Cases that failed (based on previous test)
    failed_cases = [
        50,   # CDNOW - didn't retrieve exact answer
        100,  # Deal number - didn't retrieve right info
        200,  # Edison/Lloyd Spencer - wrong email retrieved
        300   # Kaiser Plant - wrong email retrieved
    ]

    print(f"Analyzing {len(failed_cases)} failed test cases...\n")

    for idx in failed_cases:
        analyze_failed_case(qa_df, idx, search_engine)

    print("=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print()
    print("Based on the analysis above, here are potential improvements:")
    print()
    print("1. INCREASE TOP_K FOR RAG:")
    print("   - Current: max_emails=5")
    print("   - Suggested: max_emails=10 or more")
    print("   - Reason: Some correct emails are ranked lower (position 6-10)")
    print()
    print("2. IMPROVE EMBEDDING STRATEGY:")
    print("   - Current: subject + body combined")
    print("   - Consider: Weighted combination or separate fields")
    print("   - Consider: Using a better embedding model (e.g., all-mpnet-base-v2)")
    print()
    print("3. ADD METADATA FILTERING:")
    print("   - Use user/sender as a filter when possible")
    print("   - Use date ranges when questions mention specific dates")
    print()
    print("4. HYBRID SEARCH:")
    print("   - Combine semantic search with keyword matching")
    print("   - Use BM25 or similar for exact term matching (deal numbers, dates)")
    print()
    print("5. RERANKING:")
    print("   - Add a reranking step using a cross-encoder")
    print("   - This can improve precision for the final top-K results")
    print()


if __name__ == "__main__":
    main()
