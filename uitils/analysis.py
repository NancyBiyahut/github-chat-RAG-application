import ast


def extract_docstring(node):
    """
    Safely extract docstring from a class or function node.
    """
    return ast.get_docstring(node)


def detect_control_flow(node):
    """
    Detect high-level control flow patterns inside a node.
    This is NOT deep analysis — just semantic hints for retrieval.
    """
    flags = set()

    for child in ast.walk(node):
        if isinstance(child, ast.Try):
            flags.add("exception_handling")
        elif isinstance(child, ast.If):
            flags.add("conditional_logic")
        elif isinstance(child, (ast.For, ast.While)):
            flags.add("looping")

    return list(flags)


def infer_intent(name, imported_modules):
    """
    Infer intent using deterministic, explainable heuristics.
    This is important for interviews and debugging.
    """
    name = name.lower()
    intents = set()

    if any(k in name for k in ["auth", "token", "jwt", "verify", "validate"]):
        intents.add("authentication")

    if any(k in name for k in ["fetch", "get", "read", "load"]):
        intents.add("data_access")

    if any(k in name for k in ["write", "save", "upload", "put"]):
        intents.add("data_persistence")

    if any("boto3" in m or "s3" in m for m in imported_modules):
        intents.add("cloud_storage")

    if any("requests" in m or "http" in m for m in imported_modules):
        intents.add("networking")

    return list(intents)


def infer_file_role(file_path: str):
    """
    Infer the responsibility of a file based on its path.
    This gives global context to every chunk inside it.
    """
    path = file_path.lower()

    if "auth" in path:
        return "authentication / authorization logic"
    if "middleware" in path:
        return "request / response middleware"
    if "etl" in path or "pipeline" in path:
        return "data pipeline orchestration"
    if "utils" in path:
        return "shared utility functions"
    if "service" in path:
        return "business logic service layer"

    return "general application logic"
