# 

import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class CodeEmbedder:
    def __init__(
        self,
        model_name="BAAI/bge-large-zh-v1.5",
        store_dir="output"
    ):
        self.model = SentenceTransformer(model_name)
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True)

        self.index_path = self.store_dir / "index.faiss"
        self.chunks_path = self.store_dir / "chunks.json"
        self.id_map_path = self.store_dir / "id_map.json"

        self.index = None
        self.chunks = []
        self.id_map = {}

        self._load_if_exists()

    # -------------------------
    # Load persisted state
    # -------------------------
    def _load_if_exists(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            print("Loaded FAISS index from disk")

        if self.chunks_path.exists():
            self.chunks = json.loads(self.chunks_path.read_text())
            print("Loaded chunks metadata")

        if self.id_map_path.exists():
            self.id_map = json.loads(self.id_map_path.read_text())
            print("Loaded id map")

    # -------------------------
    # Build & persist index
    # -------------------------
    def build_index(self, chunks):
        if self.index is not None:
            print("Index already exists — skipping rebuild")
            return

        texts = [c["retrieval_text"] for c in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.index.add(embeddings)

        self.chunks = chunks
        self.id_map = {str(i): i for i in range(len(chunks))}

        self._persist()

        print(f"FAISS index built with {len(chunks)} vectors")

    # -------------------------
    # Save everything
    # -------------------------
    def _persist(self):
        faiss.write_index(self.index, str(self.index_path))

        self.chunks_path.write_text(
            json.dumps(self.chunks, indent=2)
        )

        self.id_map_path.write_text(
            json.dumps(self.id_map, indent=2)
        )

    # -------------------------
    # Search
    # -------------------------
    def search(self, query, k=5):
        if self.index is None:
            raise RuntimeError("Index not built")

        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, ids = self.index.search(q_emb, k)

        results = []
        for idx, score in zip(ids[0], scores[0]):
            chunk = self.chunks[idx]
            results.append({
                "score": float(score),
                "file_path": chunk["file_path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "retrieval_text": chunk["retrieval_text"],
                "name" : chunk["name"],
                "intent_tags": chunk["intent_tags"]
            })

        return results

    # -------------------------
    # Format chunks for LLM
    # -------------------------
    def format_for_prompt(self, search_results):
        """
        Converts retrieved chunks into a clean, grounded
        context block for LLM prompting.
        """
        formatted = []

        for r in search_results:
            formatted.append(
                f"""File: {r['file_path']}
                Lines: {r['start_line']} - {r['end_line']}
                Name: {r.get('name')}
                Intent: {r.get('intent_tags')}

                {r['retrieval_text']}
          """
            )

        return "\n\n---\n\n".join(formatted)
