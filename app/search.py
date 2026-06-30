import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load the same embedding model used during ingestion
model = SentenceTransformer("all-MiniLM-L6-v2")

# Path for saved data
CHUNKS_PATH = Path("data/chunks.json")
INDEX_PATH = Path("data/faiss.index")
CHUNKS_METADATA_PATH = Path("data/chunks_metadata.json")


def build_index():
    """Convert all chunks to embeddings and build a FAISS index"""
    with open(CHUNKS_PATH) as f:            # Load chunks from disk
        chunks = json.load(f)

    print(f"Building index for {len(chunks)} chunks...")

    # Extract just the text from each chunk for embedding
    texts = [chunk["text"] for chunk in chunks]
    # Convert all texs to embedding s - this returns a numpy array
    embeddings = model.encode(texts, show_progress_bar=True)
    # Get the dimension size (384 for all-MiniLM-l6-v2)
    dimension = embeddings.shape[1]

    # Create a FAISS index using L2 (straight-line) distance
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # Add all embeddings to the index
    # save the index to disk so we don't build every time
    faiss.write_index(index, str(INDEX_PATH))

    # Save metadata separately - FAISS only stores vectors, not text
    with open(CHUNKS_METADATA_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Index built with {index.ntotal} vectors")
    print(f"Saved to {INDEX_PATH}")


def search(query: str, top_k: int = 3) -> list:    # top_k =3 means give the top best 3 matches
    """Search the FAISS index for the most relevant chunks"""
    index = faiss.read_index(
        str(INDEX_PATH))   # Load the saved index from disk
    with open(CHUNKS_METADATA_PATH) as f:
        chunks = json.load(f)
    # Convert the query to an embedding
    query_embedding = model.encode([query])
    # Search FAISS - return distances and indices of top_k matches
    distances, indices = index.search(query_embedding, top_k)
    results = []   # Build reults by matching indices back to metadata
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
    build_index()  # Step 1 - build index
    query = "insulin pump therapy for type 1 diabetes"
    print(f"\nSearching for: '{query}'")
    results = search(query, top_k=3)

    for result in results:
        print(f"\nRank {result['rank']} - {result['title']}")
        print(f"Trial ID: {result['trial_id']}")
        print(f"Score: {result['score']}")
        print(f"Text preview: {result['text']}")
