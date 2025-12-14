import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from embedder import EmailEmbedder
from indexer import EmailVectorIndex
from typing import List, Dict, Optional
import os


class EmailSearchEngine:
    """Search engine combining semantic embeddings and optional BM25"""

    def __init__(self, use_hybrid=True):
        self.embedder = None
        self.index = None
        self.hybrid_engine = None
        self.use_hybrid = use_hybrid
        self._initialized = False

    def initialize(self):
        """Load index and embedder"""
        if self._initialized:
            return

        print("Initializing search engine...")
        self.embedder = EmailEmbedder()
        self.index = EmailVectorIndex()
        self.index.load_index()

        if self.use_hybrid:
            try:
                from hybrid_search import HybridSearchEngine
                print("Initializing hybrid search (semantic + BM25)...")
                self.hybrid_engine = HybridSearchEngine(
                    self.index.emails_df,
                    self.index.embeddings,
                    self.index.metadata
                )
            except ImportError as e:
                print(f"Warning: Could not load hybrid search: {e}")
                print("  Falling back to semantic-only search")
                print("  Install rank-bm25: pip install rank-bm25")
                self.use_hybrid = False

        self._initialized = True
        print(f"Search engine ready ({'hybrid' if self.use_hybrid else 'semantic-only'} mode)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        users: Optional[List[str]] = None,
        folders: Optional[List[str]] = None,
        date_year_min: Optional[int] = None,
        date_year_max: Optional[int] = None,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4
    ) -> List[Dict]:
        """
        Search emails with optional faceted filters.

        Args:
            query: Search query text
            top_k: Number of results to return
            users: List of users to filter by
            folders: List of folders to filter by
            date_year_min: Minimum year filter
            date_year_max: Maximum year filter
            semantic_weight: Weight for semantic scores (hybrid mode only)
            bm25_weight: Weight for BM25 scores (hybrid mode only)

        Returns:
            List of result dictionaries
        """
        if not self._initialized:
            self.initialize()

        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)

        # Use hybrid search if available, otherwise fall back to semantic
        if self.use_hybrid and self.hybrid_engine:
            results = self.hybrid_engine.search(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight,
                users=users,
                folders=folders,
                date_year_min=date_year_min,
                date_year_max=date_year_max
            )
        else:
            # Semantic-only search (original behavior)
            filters = {}
            if users:
                filters['users'] = users
            if folders:
                filters['folders'] = folders
            if date_year_min:
                filters['date_year_min'] = date_year_min
            if date_year_max:
                filters['date_year_max'] = date_year_max

            results = self.index.search(
                query_embedding,
                top_k=top_k,
                filters=filters if filters else None
            )

        return results

    def get_facet_values(self) -> Dict:
        """Get available values for each facet."""
        if not self._initialized:
            self.initialize()

        return self.index.get_facet_values()

    def get_stats(self) -> Dict:
        """ Get search index statistics. """
        if not self._initialized:
            self.initialize()

        facets = self.get_facet_values()

        return {
            'total_emails': len(self.index.metadata),
            'unique_users': len(facets['users']),
            'unique_folders': len(facets['folders']),
            'date_range': (min(facets['years']), max(facets['years'])),
            'embedding_dimension': self.index.embeddings.shape[1]
        }
