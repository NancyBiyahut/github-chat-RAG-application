import ast
import hashlib
from pathlib import Path

EXCLUDED_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules"}

# ---------- Utilities ----------

def should_skip(path: Path):
    return any(part in EXCLUDED_DIRS for part in path.parts)

def generate_chunk_id(file_path, symbol_name, start, end):
    raw = f"{file_path}:{symbol_name}:{start}:{end}"
    return hashlib.sha1(raw.encode()).hexdigest()


# ---------- Call Graph Visitor ----------

class CallGraphVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.append({
                "name": node.func.id,
                "lineno": node.lineno
            })

        elif isinstance(node.func, ast.Attribute):
            self.calls.append({
                "name": node.func.attr,
                "lineno": node.lineno
            })

        self.generic_visit(node)


# ---------- Chunk Extractor ----------

class PythonChunkExtractor(ast.NodeVisitor):
    def __init__(self, file_path, source_lines):
        self.file_path = file_path
        self.source_lines = source_lines
        self.chunks = []
        self.current_class = None

    def visit_ClassDef(self, node):
        self.chunks.append(
            self._build_chunk(
                symbol_type="class",
                symbol_name=node.name,
                start=node.lineno,
                end=node.end_lineno
            )
        )

        prev = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev

    def visit_FunctionDef(self, node):
        symbol_type = "method" if self.current_class else "function"
        symbol_name = (
            f"{self.current_class}.{node.name}"
            if self.current_class else node.name
        )

        self.chunks.append(
            self._build_chunk(
                symbol_type=symbol_type,
                symbol_name=symbol_name,
                start=node.lineno,
                end=node.end_lineno
            )
        )

        self.generic_visit(node)

    def _build_chunk(self, symbol_type, symbol_name, start, end):
        content = "".join(self.source_lines[start - 1:end])
        return {
            "chunk_id": generate_chunk_id(
                self.file_path, symbol_name, start, end
            ),
            "file_path": self.file_path,
            "symbol_type": symbol_type,
            "symbol_name": symbol_name,
            "start_line": start,
            "end_line": end,
            "content": content
        }


# ---------- Extraction Logic ----------

def extract_chunks_from_file(py_file):
    with open(py_file, "r", encoding="utf-8") as f:
        source = f.read()
        source_lines = source.splitlines(keepends=True)

    tree = ast.parse(source)
    extractor = PythonChunkExtractor(str(py_file), source_lines)
    extractor.visit(tree)

    chunks = extractor.chunks

    # attach raw calls per chunk
    for chunk in chunks:
        call_visitor = CallGraphVisitor()
        subtree = ast.parse(chunk["content"])
        call_visitor.visit(subtree)
        chunk["raw_calls"] = call_visitor.calls

    return chunks


# def extract_chunks_from_repo(repo_path):
#     all_chunks = []

#     for py_file in repo_path.rglob("*.py"):
#         if should_skip(py_file):
#             continue
#         try:
#             all_chunks.extend(extract_chunks_from_file(py_file))
#         except SyntaxError as e:
#             print(f"[SKIPPED - SyntaxError] {py_file}: {e}")


#     return all_chunks

def extract_chunks_from_repo(repo_path: Path):
    all_chunks = []
    found_files = []

    print(f"\n[INFO] Scanning: {repo_path}\n")

    for path in repo_path.rglob("*"):
        if path.is_file() and path.suffix == ".py":
            found_files.append(path)
            print(f"[FOUND] {path}")

            try:
                all_chunks.extend(extract_chunks_from_file(path))
            except SyntaxError as e:
                print(f"[SKIPPED - SyntaxError] {path}: {e}")

    print(f"\n[SUMMARY]")
    print(f"Python files found: {len(found_files)}")
    print(f"Chunks extracted: {len(all_chunks)}")

    return all_chunks



# ---------- Call Resolution (Same File) ----------

def resolve_calls(chunks):
    symbol_index = {}

    for chunk in chunks:
        key = chunk["symbol_name"].split(".")[-1]
        symbol_index[key] = chunk["chunk_id"]

    for chunk in chunks:
        resolved = []
        for call in chunk.get("raw_calls", []):
            target_id = symbol_index.get(call["name"])
            if target_id:
                resolved.append({
                    "called_symbol": call["name"],
                    "called_chunk_id": target_id,
                    "line_number": call["lineno"]
                })

        chunk["calls"] = resolved
        chunk.pop("raw_calls", None)

    return chunks

def build_retrieval_text(chunk):
    lines = chunk["content"].splitlines()

    header = [
        f"Symbol: {chunk['symbol_name']}",
        f"Type: {chunk['symbol_type']}",
        f"File: {chunk['file_path']}"
    ]

    if chunk.get("calls"):
        called = ", ".join(
            c["called_symbol"] for c in chunk["calls"]
        )
        header.append(f"Calls: {called}")

    # keep only first N meaningful lines
    body = "\n".join(lines[:15])

    return "\n".join(header) + "\n\n" + body

# ---------- Entry Point ----------

# if __name__ == "__main__":
#     repo_path = Path("D:\\youtube-datalake\\dags" ).resolve() 

#     chunks = extract_chunks_from_repo(repo_path)
#     chunks = resolve_calls(chunks)

#     for chunk in chunks:
#         chunk["retrieval_text"] = build_retrieval_text(chunk)
        
#     output_path = Path("output/chunks.json")
#     output_path.parent.mkdir(exist_ok=True)

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(chunks, f, indent=2)

#     print(f"Extracted {len(chunks)} chunks")
