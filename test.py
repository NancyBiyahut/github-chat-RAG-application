# from pathlib import Path
# import json

# from chunker import extract_chunks_from_repo
# from embedder import CodeEmbedder


# # ---------------- CONFIG ----------------

# repo_path = Path("D:\\youtube-datalake\\dags").resolve()
# output_path = Path("output/chunks.json")


# # ---------------- CHUNK EXTRACTION ----------------

# chunks = extract_chunks_from_repo(repo_path)

# print(f"Extracted {len(chunks)} chunks")

# output_path.parent.mkdir(exist_ok=True)

# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(chunks, f, indent=2)

# print(f"Chunks written to {output_path}")


# # ---------------- EMBEDDING + INDEX ----------------

# embedder = CodeEmbedder()
# embedder.build_index(chunks)

# print("FAISS index built successfully")


# # ---------------- SEARCH TEST ----------------

# query = "Where are files uploaded to an S3 bucket?"

# results = embedder.search(query, k=3)

# print("\nTop results:\n")

# for r in results:
#     print("File:", r["file_path"])
#     print("Chunk:", r["name"])
#     print("Intent:", r.get("intent_tags"))
#     print("-" * 60)
from pathlib import Path
import json

from chunker import extract_chunks_from_repo
from embedder import CodeEmbedder


# ---------------- CONFIG ----------------

repo_path = Path("D:\\youtube-datalake\\dags").resolve()
output_path = Path("output/chunks.json")


# ---------------- CHUNK EXTRACTION ----------------

chunks = extract_chunks_from_repo(repo_path)

print(f"Extracted {len(chunks)} chunks")

output_path.parent.mkdir(exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"Chunks written to {output_path}")


# ---------------- EMBEDDING + INDEX ----------------

embedder = CodeEmbedder()
embedder.build_index(chunks)

print("FAISS index ready")


# ---------------- SEARCH TEST ----------------

query = "Where are files uploaded to an S3 bucket?"

results = embedder.search(query, k=3)

print("\nTop results:\n")

for r in results:
    print("File      :", r["file_path"])
    print("Chunk     :", r.get("name"))
    print("Lines     :", f'{r.get("start_line")} - {r.get("end_line")}')
    print("Intent    :", r.get("intent_tags"))
    print("Score     :", round(r["score"], 4))
    print("-" * 60)
