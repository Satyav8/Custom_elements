import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

_CATALOGUE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalogue.csv")
_CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        import os, sys
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Suppress tqdm stderr output (causes OSError in Streamlit on Windows)
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr
    return _model


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection

    client = chromadb.PersistentClient(path=_CHROMA_PATH)

    # Return existing if already indexed
    existing = [c.name for c in client.list_collections()]
    if "catalogue" in existing:
        _collection = client.get_collection("catalogue")
        return _collection

    # Build index from scratch
    df = pd.read_csv(_CATALOGUE_PATH)
    model = _get_model()

    # Each SKU becomes one document; text combines key fields for semantic search
    texts = (
        df["name"] + " | " +
        df["category"] + " | " +
        df["brand"] + " | " +
        df["vehicle_fitment"] + " | " +
        df["description"]
    ).tolist()

    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    _collection = client.create_collection("catalogue", metadata={"hnsw:space": "cosine"})
    _collection.add(
        ids=df["sku"].tolist(),
        embeddings=embeddings,
        documents=texts,
        metadatas=df.to_dict(orient="records"),
    )
    print(f"[retrieval] Indexed {len(df)} SKUs into ChromaDB.")
    return _collection


def search(query: str, n_results: int = 8) -> list[dict]:
    """Return top-n catalogue records most relevant to query."""
    model = _get_model()
    collection = _get_collection()
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    return results["metadatas"][0]  # list of dicts


def get_by_sku(sku: str) -> dict | None:
    """Fetch a single SKU's metadata directly."""
    collection = _get_collection()
    result = collection.get(ids=[sku], include=["metadatas"])
    if result["metadatas"]:
        return result["metadatas"][0]
    return None


def get_all() -> list[dict]:
    """Return all catalogue records (used for vehicle filtering)."""
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    return result["metadatas"]
