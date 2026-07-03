import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Paths for saved data
CHUNKS_PATH = Path("data/chunks.json")
INDEX_PATH = Path("data/faiss.index")
CHUNKS_METADATA_PATH = Path("data/chunks_metadata.json")

# Model is None until first use — avoids loading at import time
_model = None


def get_model():
    """Load model only when first needed — not at import time"""
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def build_index():
    """Convert all chunks to embeddings and build a FAISS index"""
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)

    print(f"Building index for {len(chunks)} chunks...")
    texts = [chunk["text"] for chunk in chunks]

    # Load model here — only when building the index
    embeddings = get_model().encode(texts, show_progress_bar=True)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with open(CHUNKS_METADATA_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Index built with {index.ntotal} vectors")
    print(f"Saved to {INDEX_PATH}")


def search(query: str, top_k: int = 3) -> list:
    """Search the FAISS index for the most relevant chunks"""
    index = faiss.read_index(str(INDEX_PATH))

    with open(CHUNKS_METADATA_PATH) as f:
        chunks = json.load(f)

    # Load model here — only when searching
    query_embedding = get_model().encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "score": float(distances[0][i]),
            "trial_id": chunks[idx]["trial_id"],
            "title": chunks[idx]["title"],
            "text": chunks[idx]["text"]
        })
    return results


if __name__ == "__main__":
    build_index()
    query = "insulin pump therapy for type 1 diabetes"
    print(f"\nSearching for: '{query}'")
    results = search(query, top_k=3)
    for result in results:
        print(f"\nRank {result['rank']} - {result['title']}")
        print(f"Trial ID: {result['trial_id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text preview: {result['text'][:200]}...")
