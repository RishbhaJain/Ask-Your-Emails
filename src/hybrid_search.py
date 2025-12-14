import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import re


class HybridSearchEngine:
    """Combines semantic embeddings with BM25 keyword search"""

    def __init__(self, emails_df, embeddings, metadata):
        self.emails_df = emails_df
        self.embeddings = embeddings
        self.metadata = metadata
        
        # Pre-compute lowercase body text for exact matching
        self.body_lower = emails_df['body'].fillna('').astype(str).str.lower()
        self.subject_lower = emails_df['subject'].fillna('').astype(str).str.lower()

        print("Building BM25 keyword index...")
        self._build_bm25_index()
        print("BM25 index ready")

    def _build_bm25_index(self):
        """Build BM25 index from corpus"""
        corpus = []
        for _, row in self.emails_df.iterrows():
            subject = str(row.get('subject', ''))
            body = str(row.get('body', ''))
            text = f"{subject} {body}"
            tokens = self._tokenize(text)
            corpus.append(tokens)

        self.bm25 = BM25Okapi(corpus)
        self.corpus_tokens = corpus

    def _tokenize(self, text):
        """Tokenize text for BM25, preserving numbers and important terms"""
        text = text.lower()
        # Keep alphanumeric sequences (including numbers like 382621)
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens
    
    def _extract_key_terms(self, query):
        """Extract important terms from query for exact matching"""
        key_terms = []
        
        # Extract numbers (deal numbers, dates, amounts)
        numbers = re.findall(r'\b\d{3,}\b', query)
        key_terms.extend(numbers)
        
        # Extract specific dates
        dates = re.findall(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b', query.lower())
        key_terms.extend(dates)
        
        # Extract quoted terms
        quoted = re.findall(r'["\']([^"\'\n]+)["\']', query)
        key_terms.extend(quoted)
        
        # Extract proper nouns and specific terms (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        key_terms.extend([p.lower() for p in proper_nouns])
        
        return [term.lower() for term in key_terms if len(term) > 2]

    def _normalize_scores(self, scores):
        """Normalize scores to [0, 1]"""
        if len(scores) == 0:
            return scores

        min_score = scores.min()
        max_score = scores.max()

        if max_score - min_score < 1e-10:
            return np.ones_like(scores)

        return (scores - min_score) / (max_score - min_score)

    def search(self, query, query_embedding, top_k=10, semantic_weight=0.6, bm25_weight=0.4,
               users=None, folders=None, date_year_min=None, date_year_max=None):
        """Hybrid search combining semantic and BM25 scores with exact match boosting"""
        # get semantic scores
        semantic_scores = np.dot(self.embeddings, query_embedding)

        # get BM25 scores
        query_tokens = self._tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(query_tokens))

        # normalize both
        semantic_scores_norm = self._normalize_scores(semantic_scores)
        bm25_scores_norm = self._normalize_scores(bm25_scores)

        # Extract key terms for exact matching
        key_terms = self._extract_key_terms(query)
        exact_match_boost = np.zeros(len(self.emails_df))
        
        if key_terms:
            for term in key_terms:
                # Check for exact matches in body or subject
                matches = (self.body_lower.str.contains(re.escape(term), regex=True, na=False) | 
                          self.subject_lower.str.contains(re.escape(term), regex=True, na=False))
                exact_match_boost += matches.values * 0.2  # 0.2 boost per matched term
        
        # combine with exact match boost
        hybrid_scores = (
            semantic_weight * semantic_scores_norm +
            bm25_weight * bm25_scores_norm +
            exact_match_boost
        )

        # apply filters
        mask = np.ones(len(self.emails_df), dtype=bool)

        if users:
            mask &= self.emails_df['user'].isin(users).values

        if folders:
            mask &= self.emails_df['folder'].isin(folders).values

        if date_year_min:
            mask &= self.emails_df['date_year'] >= date_year_min

        if date_year_max:
            mask &= self.emails_df['date_year'] <= date_year_max

        hybrid_scores = np.where(mask, hybrid_scores, -np.inf)

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if hybrid_scores[idx] == -np.inf:
                continue

            email = self.emails_df.iloc[idx]
            results.append({
                'path': email['path'],
                'user': email['user'],
                'folder': email.get('folder', ''),
                'subject': email.get('subject', ''),
                'from': email.get('from', ''),
                'to': email.get('to', ''),
                'date': email.get('date', ''),
                'body': email.get('body', ''),
                'score': float(hybrid_scores[idx]),
                'semantic_score': float(semantic_scores[idx]),
                'bm25_score': float(bm25_scores[idx])
            })

        return results

    def get_facet_values(self) -> Dict:
        """Get available filter values"""
        return self.metadata
