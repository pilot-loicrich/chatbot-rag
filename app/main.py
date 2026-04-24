"""FastAPI entrypoint for the RAG chatbot."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import DOCUMENTS_PATH
from app.rag import SUPPORTED_EXTENSIONS, ask_question, load_and_index_documents

app = FastAPI(
    title="Chatbot RAG - Gemini API",
    description="Chatbot intelligent basé sur vos documents via RAG et Gemini",
    version="1.1.0",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str


class SourceRef(BaseModel):
    source: str
    page: Optional[int] = None
    snippet: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceRef]


class UploadResponse(BaseModel):
    message: str
    chunks: int
    filename: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "Chatbot RAG API is running", "status": "ok"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload et indexation d'un document (.pdf, .docx, .txt, .md)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Format non supporté: '{ext}'. "
                f"Formats acceptés: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    os.makedirs(DOCUMENTS_PATH, exist_ok=True)
    file_path = os.path.join(DOCUMENTS_PATH, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks_count = load_and_index_documents(file_path)

    return UploadResponse(
        message=f"Document '{file.filename}' indexé avec succès",
        chunks=chunks_count,
        filename=file.filename,
    )


@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    """Pose une question au chatbot et renvoie la réponse + les sources."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

    result = ask_question(request.question)
    return QuestionResponse(
        question=request.question,
        answer=result["answer"],
        sources=[SourceRef(**s) for s in result["sources"]],
    )
