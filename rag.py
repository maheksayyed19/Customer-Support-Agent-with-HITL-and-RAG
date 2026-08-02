"""
Real RAG (Retrieval-Augmented Generation) module for the support agent.

Pipeline:
    1. Load documents from docs/ (policy files, could also be scraped website text)
    2. Split each document into smaller chunks (so retrieval is precise, not whole-doc)
    3. Embed each chunk into a vector using a free HuggingFace sentence-transformer model
    4. Store vectors in ChromaDB (persisted to disk, so we don't re-embed every run)
    5. At query time: embed the customer's question the same way, and ask ChromaDB
       for the most SEMANTICALLY similar chunks -- not keyword matches.
"""
import os
import glob
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "data", "chroma_store")

# Free, local, open-source embedding model -- no API key or cost, unlike OpenAI embeddings.
# Uses ChromaDB's built-in ONNX-based embedding function -- same underlying
# model (all-MiniLM-L6-v2) as sentence-transformers, but runs on ONNX Runtime
# instead of full PyTorch. This cuts memory usage from 400MB+ down to under
# 100MB, which matters on free-tier hosting (e.g. Render's 512MB RAM cap).
_embedding_fn = embedding_functions.DefaultEmbeddingFunction()

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(
    name="support_docs",
    embedding_function=_embedding_fn,
)


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    """
    Split text into overlapping chunks of roughly `chunk_size` characters.
    Overlap keeps context from being cut off awkwardly at chunk boundaries.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def ingest_documents(force: bool = False):
    """
    Reads every .txt file in docs/, chunks it, embeds it, and stores it in ChromaDB.
    Safe to call every time the app starts -- if already ingested, it skips
    unless force=True (useful when you edit the docs and want to re-embed).
    """
    global _collection
    existing_count = _collection.count()
    if existing_count > 0 and not force:
        return existing_count

    if force and existing_count > 0:
        _client.delete_collection("support_docs")
        _collection = _client.get_or_create_collection(
            name="support_docs", embedding_function=_embedding_fn
        )

    doc_paths = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    all_chunks, all_ids, all_metadata = [], [], []

    for path in doc_paths:
        source_name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{source_name}-{i}")
            all_metadata.append({"source": source_name})

    if all_chunks:
        _collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadata)

    return len(all_chunks)


def retrieve(query: str, k: int = 3) -> str:
    """
    Semantic search: embeds the query, finds the k most similar chunks
    from ChromaDB, and returns them formatted with their source file.
    """
    results = _collection.query(query_texts=[query], n_results=k)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        return "No relevant information found in the knowledge base."

    formatted = []
    for doc, meta, dist in zip(docs, metas, distances):
        source = meta.get("source", "unknown")
        formatted.append(f"[From {source}]\n{doc}")

    return "\n\n---\n\n".join(formatted)


if __name__ == "__main__":
    count = ingest_documents(force=True)
    print(f"Ingested {count} chunks from docs/")

    test_queries = [
        "my package never showed up",              # should match shipping, no shared keywords with "delivery"
        "I got charged two times for one order",    # should match payments
        "item came broken can I get my money back", # should match return policy, despite zero word overlap
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print(retrieve(q, k=1))