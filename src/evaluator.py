import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import config
from search import EmailSearchEngine
from rag import EmailRAG
from anthropic import Anthropic


class RAGEvaluator:
    """Evaluate RAG system using ground truth Q&A pairs"""

    def __init__(self, search_engine, rag):
        self.search_engine = search_engine
        self.rag = rag
        self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def load_eval_data(self, max_samples=None):
        """Load Q&A pairs for evaluation"""
        qa_path = config.PROCESSED_DIR / "eval_qa_pairs.parquet"
        df = pd.read_parquet(qa_path)

        if max_samples:
            df = df.sample(n=min(max_samples, len(df)), random_state=42)

        return df

    def evaluate_retrieval(self, qa_df, top_k=10):
        """Evaluate retrieval quality (Recall@K, MRR, Precision@1)"""
        print(f"Evaluating retrieval quality on {len(qa_df)} questions...")

        recalls = []
        reciprocal_ranks = []
        precision_at_1 = []

        for idx, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Retrieval eval"):
            question = row['question']
            ground_truth_path = row['path']

            results = self.search_engine.search(query=question, top_k=top_k)
            retrieved_paths = [r['path'] for r in results]

            if ground_truth_path in retrieved_paths:
                rank = retrieved_paths.index(ground_truth_path) + 1
                recalls.append(1.0)
                reciprocal_ranks.append(1.0 / rank)
                precision_at_1.append(1.0 if rank == 1 else 0.0)
            else:
                recalls.append(0.0)
                reciprocal_ranks.append(0.0)
                precision_at_1.append(0.0)

        metrics = {
            f'recall@{top_k}': np.mean(recalls),
            'mrr': np.mean(reciprocal_ranks),
            'precision@1': np.mean(precision_at_1),
            'total_questions': len(qa_df)
        }

        return metrics

    def evaluate_answer_quality(self, qa_df, sample_size=50):
        """Evaluate answer quality using Claude as judge"""
        print(f"Evaluating answer quality on {sample_size} samples...")

        sample_df = qa_df.sample(n=min(sample_size, len(qa_df)), random_state=42)

        scores = []
        evaluations = []

        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Answer eval"):
            question = row['question']
            ground_truth_answer = row['answer']

            # Generate RAG answer
            try:
                results = self.search_engine.search(query=question, top_k=10)

                if not results:
                    scores.append(0.0)
                    evaluations.append({
                        'question': question,
                        'ground_truth': ground_truth_answer,
                        'rag_answer': 'No results found',
                        'score': 0.0,
                        'feedback': 'No retrieval results'
                    })
                    continue

                rag_response = self.rag.answer_question(question, results)
                rag_answer = rag_response['answer']

                score, feedback = self._judge_answer(question, ground_truth_answer, rag_answer)

                scores.append(score)
                evaluations.append({
                    'question': question,
                    'ground_truth': ground_truth_answer,
                    'rag_answer': rag_answer,
                    'score': score,
                    'feedback': feedback
                })

            except Exception as e:
                print(f"Error evaluating question: {e}")
                scores.append(0.0)
                evaluations.append({
                    'question': question,
                    'ground_truth': ground_truth_answer,
                    'rag_answer': f'ERROR: {str(e)}',
                    'score': 0.0,
                    'feedback': f'Error: {str(e)}'
                })

        metrics = {
            'mean_score': np.mean(scores),
            'median_score': np.median(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'samples_evaluated': len(scores),
            'evaluations': evaluations
        }

        return metrics

    def _judge_answer(self, question, ground_truth, rag_answer):
        """Use Claude to judge answer quality (0-10 scale)"""
        prompt = f"""You are evaluating a RAG system's answer quality.

Question: {question}

Ground Truth Answer: {ground_truth}

RAG System Answer: {rag_answer}

Rate the RAG answer on a scale of 0-10 where:
- 10: Perfect answer, matches ground truth semantically and factually
- 7-9: Good answer, captures main points with minor differences
- 4-6: Partial answer, some correct information but missing key points
- 1-3: Poor answer, mostly incorrect or irrelevant
- 0: Completely wrong or no answer

Respond with ONLY:
SCORE: <number>
FEEDBACK: <brief explanation>"""

        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.content[0].text.strip()

            # Parse score and feedback
            score_line = [l for l in result.split('\n') if l.startswith('SCORE:')]
            feedback_line = [l for l in result.split('\n') if l.startswith('FEEDBACK:')]

            if score_line:
                score = float(score_line[0].replace('SCORE:', '').strip())
            else:
                score = 0.0

            if feedback_line:
                feedback = feedback_line[0].replace('FEEDBACK:', '').strip()
            else:
                feedback = result

            return score, feedback

        except Exception as e:
            print(f"Error judging answer: {e}")
            return 0.0, f"Error: {str(e)}"

    def run_full_evaluation(
        self,
        retrieval_samples: int = 100,
        answer_samples: int = 30,
        top_k: int = 10
    ) -> Dict:
        """
        Run comprehensive evaluation.

        Args:
            retrieval_samples: Number of samples for retrieval eval
            answer_samples: Number of samples for answer quality eval
            top_k: Top K for retrieval

        Returns:
            Dict with all evaluation metrics
        """
        print("=" * 60)
        print("RAG SYSTEM EVALUATION")
        print("=" * 60)

        # Load data
        qa_df = self.load_eval_data()
        print(f"\nLoaded {len(qa_df)} Q&A pairs for evaluation")

        # Retrieval evaluation
        print(f"\n--- Retrieval Evaluation (n={retrieval_samples}) ---")
        retrieval_df = qa_df.sample(n=min(retrieval_samples, len(qa_df)), random_state=42)
        retrieval_metrics = self.evaluate_retrieval(retrieval_df, top_k=top_k)

        print(f"\nRetrieval Metrics:")
        print(f"  Recall@{top_k}: {retrieval_metrics[f'recall@{top_k}']:.2%}")
        print(f"  MRR: {retrieval_metrics['mrr']:.3f}")
        print(f"  Precision@1: {retrieval_metrics['precision@1']:.2%}")

        # Answer quality evaluation
        print(f"\n--- Answer Quality Evaluation (n={answer_samples}) ---")
        answer_metrics = self.evaluate_answer_quality(qa_df, sample_size=answer_samples)

        print(f"\nAnswer Quality Metrics:")
        print(f"  Mean Score: {answer_metrics['mean_score']:.2f}/10")
        print(f"  Median Score: {answer_metrics['median_score']:.2f}/10")
        print(f"  Std Dev: {answer_metrics['std_score']:.2f}")
        print(f"  Range: [{answer_metrics['min_score']:.1f}, {answer_metrics['max_score']:.1f}]")

        # Combine results
        results = {
            'retrieval': retrieval_metrics,
            'answer_quality': answer_metrics,
            'config': {
                'retrieval_samples': retrieval_samples,
                'answer_samples': answer_samples,
                'top_k': top_k
            }
        }

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)

        return results


def save_evaluation_results(results: Dict, output_path: Path):
    """Save evaluation results to file."""
    import json

    # Remove evaluations list (too large for JSON)
    results_copy = results.copy()
    if 'answer_quality' in results_copy and 'evaluations' in results_copy['answer_quality']:
        evaluations = results_copy['answer_quality'].pop('evaluations')

        # Save evaluations separately
        eval_df = pd.DataFrame(evaluations)
        eval_path = output_path.parent / "evaluation_details.csv"
        eval_df.to_csv(eval_path, index=False)
        print(f"Saved detailed evaluations to {eval_path}")

    # Save summary
    with open(output_path, 'w') as f:
        json.dump(results_copy, f, indent=2)

    print(f"Saved evaluation summary to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Initializing RAG system for evaluation...")

    search_engine = EmailSearchEngine()
    search_engine.initialize()

    rag = EmailRAG()

    evaluator = RAGEvaluator(search_engine, rag)

    # Run evaluation
    results = evaluator.run_full_evaluation(
        retrieval_samples=100,
        answer_samples=30,
        top_k=10
    )

    # Save results
    output_path = config.PROCESSED_DIR / "evaluation_results.json"
    save_evaluation_results(results, output_path)
