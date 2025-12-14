import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.insert(0, str(Path(__file__).parent))
import config


class EmailVectorIndex:
    """Vector index for email search using cosine similarity"""

    def __init__(self):
        self.embeddings = None
        self.metadata = None
        self.emails_df = None  # Full DataFrame for hybrid search
        self.index_path = config.PROCESSED_DIR / "vector_index.pkl"

    def build_index(self, emails_df, embeddings):
        """Build index from emails and embeddings"""
        print("\nBuilding vector index...")

        # Store full DataFrame for hybrid search
        self.emails_df = emails_df.copy()

        # normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / norms

        self.metadata = emails_df[[
            'path', 'user', 'subject', 'from', 'to',
            'body', 'date_year', 'date_month'
        ]].copy()

        self.metadata['folder'] = emails_df['path'].apply(self._extract_folder)

        # handle missing dates
        self.metadata['date_year'] = self.metadata['date_year'].fillna(2000).astype(int)
        self.metadata['date_month'] = self.metadata['date_month'].fillna(1).astype(int)

    def _extract_folder(self, path: str) -> str:
        """Extract folder name from email path."""
        parts = path.split('/')
        return parts[1] if len(parts) >= 2 else "unknown"

    def search(self, query_embedding, top_k=10, filters=None):
        """Search for similar emails"""
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        mask = self._apply_filters(filters)
        filtered_indices = np.where(mask)[0]

        if len(filtered_indices) == 0:
            print("Warning: No emails match the filters")
            return []

        filtered_embeddings = self.embeddings[filtered_indices]
        similarities = np.dot(filtered_embeddings, query_norm)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        actual_indices = filtered_indices[top_indices]

        results = []
        for i, idx in enumerate(actual_indices):
            result = {
                'rank': i + 1,
                'score': float(similarities[top_indices[i]]),
                'path': self.metadata.iloc[idx]['path'],
                'user': self.metadata.iloc[idx]['user'],
                'folder': self.metadata.iloc[idx]['folder'],
                'subject': self.metadata.iloc[idx]['subject'],
                'from': self.metadata.iloc[idx]['from'],
                'to': self.metadata.iloc[idx]['to'],
                'body': self.metadata.iloc[idx]['body'],
                'date_year': int(self.metadata.iloc[idx]['date_year']),
            }
            results.append(result)

        return results

    def _apply_filters(self, filters=None):
        """Apply metadata filters"""
        mask = np.ones(len(self.metadata), dtype=bool)

        if not filters:
            return mask

        if filters.get('users'):
            user_mask = self.metadata['user'].isin(filters['users'])
            mask &= user_mask.values

        if filters.get('folders'):
            folder_mask = self.metadata['folder'].isin(filters['folders'])
            mask &= folder_mask.values

        if filters.get('date_year_min'):
            date_mask = self.metadata['date_year'] >= filters['date_year_min']
            mask &= date_mask.values

        if filters.get('date_year_max'):
            date_mask = self.metadata['date_year'] <= filters['date_year_max']
            mask &= date_mask.values

        return mask

    def save_index(self, path=None):
        """Save index to disk"""
        path = path or self.index_path

        index_data = {
            'embeddings': self.embeddings,
            'metadata': self.metadata,
            'emails_df': self.emails_df  # Save full DataFrame for hybrid search
        }

        with open(path, 'wb') as f:
            pickle.dump(index_data, f)

        print(f"\n✓ Index saved to: {path}")
        print(f"  File size: {Path(path).stat().st_size / 1024 / 1024:.1f} MB")

    def load_index(self, path=None):
        """Load index from disk"""
        path = path or self.index_path

        print(f"Loading index from: {path}")

        with open(path, 'rb') as f:
            index_data = pickle.load(f)

        self.embeddings = index_data['embeddings']
        self.metadata = index_data['metadata']
        # Load emails_df if available (for backward compatibility with old indexes)
        self.emails_df = index_data.get('emails_df', self.metadata.copy())

    def get_facet_values(self) -> Dict:
        """Get unique values for each facet (for UI dropdowns)."""
        return {
            'from': sorted(self.metadata['from'].dropna().unique().tolist()),
            'to': sorted(self.metadata['to'].dropna().unique().tolist()),
            'years': sorted(self.metadata['date_year'].unique().tolist()),
            # Keep for backward compatibility
            'users': sorted(self.metadata['user'].unique().tolist()),
            'folders': sorted(self.metadata['folder'].unique().tolist())
        }

    def test_search(self, query_text: str, embedder, top_k: int = 5):
        """ Test search with a text query."""

        print(f"\nTest search: '{query_text}'")
        print("-" * 60)

        query_embedding = embedder.embed_text(query_text)
        results = self.search(query_embedding, top_k=top_k)

        for result in results:
            print(f"\n[{result['rank']}] Score: {result['score']:.3f}")
            print(f"    Subject: {result['subject'][:60]}")
            print(f"    From: {result['from'][:40]}")
            print(f"    Folder: {result['folder']}")

        return results
