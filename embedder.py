from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from typing import List, Dict


class CodeEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []

    def build_index(self, chunks: List[Dict]):
        """
        chunks: output from chunker.py
        """
        texts = [c["retrieval_text"] for c in chunks]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.index.add(embeddings)

        self.metadata = chunks

    def search(self, query: str, k: int = 5):
        query = f"Represent this query for retrieving relevant code: {query}"

        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = self.index.search(query_vec, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            chunk = self.metadata[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results


    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}_meta.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}_meta.pkl", "rb") as f:
            self.metadata = pickle.load(f)
