from fastapi import FastAPI
from pydantic import BaseModel

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
    return {
        "question": question.text,
        "answer": "RAG pipeline not yet connected",
        "source": None
    }
