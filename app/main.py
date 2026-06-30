from fastapi import FastAPI
from pydantic import BaseModel
from app.search import search


# Initialise the FastAPI app
app = FastAPI(title="Clinical RAG API", version="0.1.0")

# Define the shape of a question request


class Question(BaseModel):
    text: str

# Health check endpoint — confirms the API is running


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}

# Main RAG endpoint — will return real answers once pipeline is built


@app.post("/ask")
def ask_question(question: Question):
    # Search the FAISS index for the most relevant 3 chunks
    results = search(question.text, top_k=3)
    top_result = results[0]
    return {
        "question": question.text,
        "answer": top_result["text"],
        "source": {
            "trial_id": top_result["trial_id"],
            "title": top_result["title"],
            "score": top_result["score"]
        }
    }
