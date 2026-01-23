from ollama import chat


MODEL_NAME = "mistral"   


def generate_answer(question: str, retrieved_chunks: list) -> str:

    # --- Build context safely ---
    context_blocks = []

    for c in retrieved_chunks:
        block = f"""
        FILE: {c['file_path']}
        LINES: {c['start_line']} - {c['end_line']}
        TYPE: {c.get('type')}
        INTENT: {c.get('intent_tags')}

        SUMMARY:
        {c['retrieval_text']}
        """
        context_blocks.append(block.strip())

    context = "\n\n---\n\n".join(context_blocks)

    # --- Prompt ---
    prompt = f"""
    You are a senior software engineer helping understand a Python codebase.

    Answer the user's question using ONLY the context below.
    If the answer cannot be found, say "Not found in the codebase".

    Context:
    {context}

    Question:
    {question}
    """.strip()

    # --- Ollama Chat ---
    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response.message.content
