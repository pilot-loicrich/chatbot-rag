from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from app.rag import load_and_index_documents, ask_question
from app.config import DOCUMENTS_PATH

app = FastAPI(
    title="Chatbot RAG - Gemini API",
    description="Chatbot intelligent basé sur vos documents via RAG et Gemini",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    question: str
    answer: str

@app.get("/")
def read_root():
    return {"message": "Chatbot RAG API is running", "status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload et indexation d'un document PDF"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés")

    os.makedirs(DOCUMENTS_PATH, exist_ok=True)
    file_path = f"{DOCUMENTS_PATH}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks_count = load_and_index_documents(file_path)

    return {
        "message": f"Document '{file.filename}' indexé avec succès",
        "chunks": chunks_count
    }

@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    """Pose une question au chatbot"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

    answer = ask_question(request.question)
    return QuestionResponse(question=request.question, answer=answer)