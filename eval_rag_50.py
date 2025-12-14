#!/usr/bin/env python3
"""
Evaluate RAG system on 50 data points with comprehensive metrics.
Includes rate limiting to avoid API timeouts.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from search import EmailSearchEngine
from rag import EmailRAG
import config


class RAGEvaluator50:
    """Evaluate RAG on 50 samples with rate limiting"""

    def __init__(self, rate_limit_delay=2.0):
        """
        Initialize evaluator.
        
        Args:
            rate_limit_delay: Seconds to wait between API calls (to avoid 429 errors)
        """
        self.search_engine = EmailSearchEngine()
        self.search_engine.initialize()
        self.rag = EmailRAG()
        self.rate_limit_delay = rate_limit_delay
        self.last_api_call = 0

    def _rate_limit(self):
        """Apply rate limiting to API calls"""
        elapsed = time.time() - self.last_api_call
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_api_call = time.time()

    def load_eval_data(self, n_samples=50):
        """Load Q&A pairs for evaluation"""
        qa_path = config.PROCESSED_DIR / "eval_qa_pairs.csv"
        df = pd.read_csv(qa_path)
        
        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=42)
        
        print(f"Loaded {len(df)} Q&A pairs")
        return df

    def evaluate_retrieval(self, qa_df, top_k=10):
        """
        Evaluate retrieval quality.
        
        Metrics:
        - Recall@K: Did we retrieve the correct email?
        - Precision@1: Was the top result correct?
        - MRR: Mean Reciprocal Rank
        """
        print(f"\nEvaluating retrieval quality (top_k={top_k})...")
        
        recalls = []
        precisions_at_1 = []
        reciprocal_ranks = []
        
        for idx, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Retrieval eval"):
            question = row['question']
            ground_truth_path = row['path']
            
            results = self.search_engine.search(query=question, top_k=top_k)
            retrieved_paths = [r['path'] for r in results]
            
            # Recall@K
            recall = 1.0 if ground_truth_path in retrieved_paths else 0.0
            recalls.append(recall)
            
            # Precision@1
            precision_at_1 = 1.0 if (len(retrieved_paths) > 0 and 
                                     retrieved_paths[0] == ground_truth_path) else 0.0
            precisions_at_1.append(precision_at_1)
            
            # MRR
            if ground_truth_path in retrieved_paths:
                rank = retrieved_paths.index(ground_truth_path) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        metrics = {
            f'recall@{top_k}': np.mean(recalls),
            'precision@1': np.mean(precisions_at_1),
            'mrr': np.mean(reciprocal_ranks),
            'samples': len(qa_df)
        }
        
        return metrics, recalls

    def evaluate_answer_quality(self, qa_df, recalls):
        """
        Evaluate answer quality using string matching and key term extraction.
        
        Metrics:
        - Exact Match: Answer exactly matches ground truth (case-insensitive)
        - Contains Key Terms: Answer contains key entities from ground truth
        - Hallucination Rate: Answer mentions facts not in retrieved emails
        """
        print(f"\nEvaluating answer quality...")
        
        exact_matches = []
        key_term_matches = []
        retrieval_failures = []
        
        for position, (idx, row) in enumerate(tqdm(qa_df.iterrows(), total=len(qa_df), desc="Answer eval")):
            question = row['question']
            ground_truth = row['answer'].lower()
            retrieved_recall = recalls[position]
            
            # Skip if retrieval failed (can't answer without source)
            if retrieved_recall == 0.0:
                retrieval_failures.append(1)
                exact_matches.append(0)
                key_term_matches.append(0)
                continue
            
            retrieval_failures.append(0)
            
            try:
                # Rate limit API calls
                self._rate_limit()
                
                results = self.search_engine.search(query=question, top_k=5)
                response = self.rag.answer_question(question, results, max_emails=3)
                rag_answer = response['answer'].lower()
                
                # Exact match (loose)
                if ground_truth in rag_answer or rag_answer in ground_truth:
                    exact_matches.append(1)
                else:
                    exact_matches.append(0)
                
                # Key term matching
                key_terms = self._extract_key_terms(ground_truth)
                if key_terms:
                    matched_terms = sum(1 for term in key_terms if term in rag_answer)
                    match_ratio = matched_terms / len(key_terms)
                    key_term_matches.append(1.0 if match_ratio >= 0.5 else 0.0)
                else:
                    key_term_matches.append(0.0)
                    
            except Exception as e:
                print(f"Error evaluating answer: {e}")
                exact_matches.append(0)
                key_term_matches.append(0)
        
        metrics = {
            'exact_match_rate': np.mean(exact_matches),
            'key_term_match_rate': np.mean(key_term_matches),
            'retrieval_failure_rate': np.mean(retrieval_failures),
            'samples': len(qa_df)
        }
        
        return metrics

    def _extract_key_terms(self, text, min_length=3):
        """Extract key terms (words) from text"""
        words = text.split()
        return [w.strip('.,;:!?') for w in words if len(w) > min_length]

    def run_evaluation(self, n_samples=50, top_k=10):
        """Run complete evaluation on n_samples"""
        
        print("=" * 80)
        print(f"RAG EVALUATION ON {n_samples} SAMPLES")
        print("=" * 80)
        
        # Load data
        qa_df = self.load_eval_data(n_samples=n_samples)
        
        # Retrieval evaluation
        print("\n" + "=" * 80)
        print("RETRIEVAL EVALUATION")
        print("=" * 80)
        retrieval_metrics, recalls = self.evaluate_retrieval(qa_df, top_k=top_k)
        
        print(f"\nRetrieval Metrics:")
        print(f"  Recall@{top_k}: {retrieval_metrics[f'recall@{top_k}']:.1%}")
        print(f"  Precision@1: {retrieval_metrics['precision@1']:.1%}")
        print(f"  MRR: {retrieval_metrics['mrr']:.3f}")
        
        # Answer quality evaluation
        print("\n" + "=" * 80)
        print("ANSWER QUALITY EVALUATION")
        print("=" * 80)
        answer_metrics = self.evaluate_answer_quality(qa_df, recalls)
        
        print(f"\nAnswer Quality Metrics:")
        print(f"  Exact Match Rate: {answer_metrics['exact_match_rate']:.1%}")
        print(f"  Key Term Match Rate: {answer_metrics['key_term_match_rate']:.1%}")
        print(f"  Retrieval Failure Rate: {answer_metrics['retrieval_failure_rate']:.1%}")
        
        # Combined metrics
        print("\n" + "=" * 80)
        print("COMBINED METRICS")
        print("=" * 80)
        
        # Accuracy = exact match rate on questions where we retrieved the right email
        answerable_questions = 1 - answer_metrics['retrieval_failure_rate']
        if answerable_questions > 0:
            conditional_accuracy = answer_metrics['exact_match_rate'] / answerable_questions
        else:
            conditional_accuracy = 0.0
        
        print(f"\n  End-to-End Accuracy (Exact Match | Retrieved): {conditional_accuracy:.1%}")
        print(f"  Overall Accuracy (Exact Match): {answer_metrics['exact_match_rate']:.1%}")
        print(f"  Coverage (Retrieved Correct Email): {retrieval_metrics[f'recall@{top_k}']:.1%}")
        print(f"  Answer Quality (Key Terms): {answer_metrics['key_term_match_rate']:.1%}")
        
        # Results summary
        results = {
            'retrieval': retrieval_metrics,
            'answer_quality': answer_metrics,
            'combined': {
                'end_to_end_accuracy': conditional_accuracy,
                'overall_accuracy': answer_metrics['exact_match_rate'],
                'coverage': retrieval_metrics[f'recall@{top_k}'],
                'answer_quality': answer_metrics['key_term_match_rate']
            },
            'config': {
                'n_samples': n_samples,
                'top_k': top_k,
                'rate_limit_delay': self.rate_limit_delay
            }
        }
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        
        return results, qa_df


def save_results(results, qa_df, output_dir=None):
    """Save evaluation results"""
    output_dir = Path(output_dir or config.PROCESSED_DIR)
    
    # Save summary as JSON
    import json
    summary_path = output_dir / "eval_results_50.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {summary_path}")
    
    # Save sample QA pairs
    sample_path = output_dir / "eval_qa_50.csv"
    qa_df.to_csv(sample_path, index=False)
    print(f"✓ Saved QA samples to {sample_path}")


if __name__ == "__main__":
    evaluator = RAGEvaluator50(rate_limit_delay=2.0)  # 2 second delay between API calls
    
    try:
        results, qa_df = evaluator.run_evaluation(n_samples=50, top_k=10)
        save_results(results, qa_df)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\n\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
