import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """
    Handles local generation of text embeddings.
    By default, uses an offline deterministic TF-IDF wrapper as a fallback
    to simulate dense embeddings locally without deep learning dependencies.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=384)
        self.is_fitted = False

    def fit(self, corpus: List[str]):
        """
        Fits the local embedding model on the current knowledge corpus.
        """
        if not corpus:
            return
        self.vectorizer.fit(corpus)
        self.is_fitted = True
        logger.info(f"EmbeddingEngine fitted on corpus of {len(corpus)} documents.")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Produces semantic vectors for a set of texts.
        """
        if not texts:
            return np.array([])
            
        if not self.is_fitted:
            # Fit on the fly if not externally fitted, simulating zero-shot
            self.fit(texts)

        # Standardize returns to simulate typical dense embeddings shape
        vectors = self.vectorizer.transform(texts).toarray()
        logger.info(f"Generated embeddings sequence of shape {vectors.shape}")
        return vectors
