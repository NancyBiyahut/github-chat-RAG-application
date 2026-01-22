
import ast
from pathlib import Path
from typing import List, Dict

from uitils.analysis import (
    extract_docstring,
    detect_control_flow,
    infer_intent,
    infer_file_role,
)


class PythonChunkExtractor(ast.NodeVisitor):
    def __init__(self, file_path: str, source_lines: List[str]):
        self.file_path = file_path
        self.source_lines = source_lines

        self.chunks = []

        # Import context
        self.imported_modules = set()
        self.imported_symbols = {}

        # Scope tracking
        self.current_class = None
        self.current_function = None

        # File-level context
        self.file_role = infer_file_role(file_path)

    # ---------------- IMPORTS ----------------

    def visit_Import(self, node):
        for alias in node.names:
            self.imported_modules.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            for alias in node.names:
                self.imported_symbols[alias.asname or alias.name] = node.module
        self.generic_visit(node)

    # ---------------- CLASSES ----------------

    def visit_ClassDef(self, node):
        prev_class = self.current_class
        self.current_class = node.name

        self._create_chunk(
            node=node,
            chunk_type="class",
            name=node.name,
            docstring=extract_docstring(node),
            control_flow=detect_control_flow(node),
            intent_tags=infer_intent(node.name, self.imported_modules),
        )

        self.generic_visit(node)
        self.current_class = prev_class

    # ---------------- FUNCTIONS ----------------

    def visit_FunctionDef(self, node):
        prev_function = self.current_function
        self.current_function = node.name

        self._create_chunk(
            node=node,
            chunk_type="method" if self.current_class else "function",
            name=node.name,
            calls=self._extract_calls(node),
            docstring=extract_docstring(node),
            control_flow=detect_control_flow(node),
            intent_tags=infer_intent(node.name, self.imported_modules),
        )

        self.generic_visit(node)
        self.current_function = prev_function

    # ---------------- CALL EXTRACTION ----------------

    def _extract_calls(self, node):
        calls = []

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    name = child.func.id
                    calls.append({
                        "name": name,
                        "resolved_via": "import"
                        if name in self.imported_symbols else "local_or_unknown",
                        "module": self.imported_symbols.get(name),
                    })

                elif isinstance(child.func, ast.Attribute):
                    calls.append({
                        "name": ast.unparse(child.func),
                        "resolved_via": "attribute",
                        "module": None,
                    })

        return calls

    # ---------------- CHUNK CREATION ----------------

    def _create_chunk(
        self,
        node,
        chunk_type,
        name,
        calls=None,
        docstring=None,
        control_flow=None,
        intent_tags=None,
    ):
        start = node.lineno - 1
        end = node.end_lineno
        code = "".join(self.source_lines[start:end])

        retrieval_text = self._build_retrieval_text(
            chunk_type=chunk_type,
            name=name,
            docstring=docstring,
            intent_tags=intent_tags,
            control_flow=control_flow,
            calls=calls,
        )

        self.chunks.append({
            "type": chunk_type,
            "name": name,
            "file_path": self.file_path,
             "start_line": start + 1,    
             "end_line": end,
            "file_role": self.file_role,
            "class": self.current_class,
            "intent_tags": intent_tags or [],
            "control_flow": control_flow or [],
            "docstring": docstring,
            "imports": {
                "modules": list(self.imported_modules),
                "symbols": self.imported_symbols,
            },
            "calls": calls or [],
            "code": code,
            "retrieval_text": retrieval_text,
        })

    # ---------------- RETRIEVAL TEXT (KEY PART) ----------------

    def _build_retrieval_text(
        self,
        chunk_type,
        name,
        docstring,
        intent_tags,
        control_flow,
        calls,
    ):
        """
        This is the most important function for RAG quality.
        It converts structured metadata into semantic language.
        """

        call_names = [c["name"] for c in calls or []]

        return f"""
{chunk_type.upper()} NAME:
{name}

FILE ROLE:
{self.file_role}

LOCATION:
{self.file_path}

CLASS CONTEXT:
{self.current_class or "N/A"}

INTENT TAGS:
{", ".join(intent_tags or ["general_logic"])}

CONTROL FLOW:
{", ".join(control_flow or ["linear_execution"])}

DOCSTRING SUMMARY:
{docstring or "No docstring provided."}

IMPORT CONTEXT:
Modules → {", ".join(self.imported_modules)}
Imported Symbols → {", ".join(self.imported_symbols.keys())}

FUNCTION CALLS:
{", ".join(call_names) if call_names else "No explicit function calls."}

DESCRIPTION:
This {chunk_type} implements logic related to {name} and is part of the
{self.file_role}. It may interact with other components via imported
modules and function calls listed above.
""".strip()


# ---------------- REPO DRIVER ----------------

def extract_chunks_from_repo(repo_path: str) -> List[Dict]:
    repo_path = Path(repo_path)
    all_chunks = []

    for py_file in repo_path.rglob("*.py"):
        if "venv" in py_file.parts or "__pycache__" in py_file.parts:
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception:
            continue

        extractor = PythonChunkExtractor(
            file_path=str(py_file),
            source_lines=source.splitlines(keepends=True),
        )
        extractor.visit(tree)
        all_chunks.extend(extractor.chunks)

    return all_chunks
