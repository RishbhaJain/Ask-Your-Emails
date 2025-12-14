from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config


class EmailEmbedder:
    """Generate embeddings for email text"""

    def __init__(self, model_name=None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"Model loaded (dim: {self.model.get_sentence_embedding_dimension()})")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        batch_size = batch_size or 32  # Reduce from config.BATCH_SIZE to 32 for stability

        print(f"Generating embeddings for {len(texts)} texts...")
        print(f"  Batch size: {batch_size}")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device="cpu"  # Explicitly use CPU for stability on Mac
        )

        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"  Shape: {embeddings.shape}")

        return embeddings

    def embed_emails(self, emails_df: pd.DataFrame) -> np.ndarray:
        """
        Generate embeddings for all emails in a DataFrame.

        Uses subject + body for embedding (concatenated).

        Args:
            emails_df: DataFrame with 'subject' and 'body' columns

        Returns:
            Array of embeddings
        """
        print("\nPreparing email texts for embedding...")

        # Combine subject and body for richer embeddings
        texts = []
        for idx, row in tqdm(emails_df.iterrows(), total=len(emails_df), desc="Preparing texts"):
            subject = str(row.get('subject', ''))
            body = str(row.get('body', ''))

            # Truncate body to avoid very long texts (first 512 words)
            body_words = body.split()
            body_truncated = ' '.join(body_words)

            # Combine subject and body
            combined_text = f"{subject} {body_truncated}".strip()
            texts.append(combined_text)

        print(f"Prepared {len(texts)} email texts")
        print(f"  Average length: {np.mean([len(t.split()) for t in texts]):.0f} words")

        # Generate embeddings
        embeddings = self.embed_batch(texts)

        return embeddings

    def save_embeddings(self, embeddings: np.ndarray, output_path: str):
        """
        Save embeddings to disk.

        Args:
            embeddings: Array of embeddings
            output_path: Path to save embeddings (.npy file)
        """
        np.save(output_path, embeddings)
        print(f"✓ Embeddings saved to: {output_path}")
        print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")

    def load_embeddings(self, input_path: str) -> np.ndarray:
        """
        Load embeddings from disk.

        Args:
            input_path: Path to embeddings file (.npy)

        Returns:
            Array of embeddings
        """
        embeddings = np.load(input_path)
        print(f"✓ Loaded embeddings from: {input_path}")
        print(f"  Shape: {embeddings.shape}")
        return embeddings
