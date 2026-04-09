import logging
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    A lightweight in-memory semantic database for RAG context retrieval.
    """
    def __init__(self):
        self.index_vectors = None
        self.metadata_store = []
        
    def add_documents(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Stores semantic vectors alongside their original source details.
        """
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Embeddings and metadata limits must explicitly match.")

        if self.index_vectors is None:
            self.index_vectors = embeddings
        else:
             # Ensure dimensions match when expanding
            if embeddings.shape[1] != self.index_vectors.shape[1]:
                # Pad to match max features if using mocked TF-IDF across dynamic corpuses
                pad_width = abs(self.index_vectors.shape[1] - embeddings.shape[1])
                if self.index_vectors.shape[1] > embeddings.shape[1]:
                    embeddings = np.pad(embeddings, ((0, 0), (0, pad_width)))
                else:
                    self.index_vectors = np.pad(self.index_vectors, ((0, 0), (0, pad_width)))
                    
            self.index_vectors = np.vstack([self.index_vectors, embeddings])
            
        self.metadata_store.extend(metadata)
        logger.info(f"Stored {len(metadata)} new documents. Total store size: {len(self.metadata_store)}")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs cosine similarity search against the vector index.
        """
        if self.index_vectors is None or len(self.metadata_store) == 0:
            return []

        # If one vector passed, ensure 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Pad query if feature space expanded
        if query_embedding.shape[1] < self.index_vectors.shape[1]:
            pad_width = self.index_vectors.shape[1] - query_embedding.shape[1]
            query_embedding = np.pad(query_embedding, ((0, 0), (0, pad_width)))

        similarities = cosine_similarity(query_embedding, self.index_vectors)[0]
        
        # Get top K indices highest similarity
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            res = self.metadata_store[idx].copy()
            res["similarity_score"] = float(similarities[idx])
            results.append(res)
            
        return results
