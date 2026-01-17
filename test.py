from chunker import extract_chunks_from_repo, build_retrieval_text ,resolve_calls
from embedder import CodeEmbedder
from pathlib import Path
import json

repo_path = Path("D:\\youtube-datalake\\dags" ).resolve() 

chunks = extract_chunks_from_repo(repo_path)
chunks = resolve_calls(chunks)

for chunk in chunks:
    chunk["retrieval_text"] = build_retrieval_text(chunk)
    
output_path = Path("output/chunks.json")
output_path.parent.mkdir(exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"Extracted {len(chunks)} chunks")

embedder = CodeEmbedder()
embedder.build_index(chunks)

results = embedder.search(
    "Where is files loaded to s3 bucket?",
    k=3
)

for r in results:
    print(r["file_path"], r["start_line"], r["end_line"])
