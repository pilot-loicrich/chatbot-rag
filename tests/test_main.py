"""Tests for the FastAPI layer of the RAG chatbot.

We mock the RAG pipeline (`ask_question`, `load_and_index_documents`) so the
tests don't require a real Gemini API key or a populated Chroma index.
"""

from __future__ import annotations

import io

from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------

def test_ask_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400


def test_ask_returns_answer_and_sources(monkeypatch):
    def fake_ask(_question: str):
        return {
            "answer": "Ceci est une réponse factice.",
            "sources": [
                {"source": "cours1.pdf", "page": 3, "snippet": "extrait 1"},
                {"source": "notes.md", "page": None, "snippet": "extrait 2"},
            ],
        }

    monkeypatch.setattr(main_module, "ask_question", fake_ask)

    response = client.post("/ask", json={"question": "Test ?"})
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "Test ?"
    assert body["answer"] == "Ceci est une réponse factice."
    assert len(body["sources"]) == 2
    assert body["sources"][0] == {
        "source": "cours1.pdf",
        "page": 3,
        "snippet": "extrait 1",
    }
    assert body["sources"][1]["page"] is None


# ---------------------------------------------------------------------------
# /upload
# ---------------------------------------------------------------------------

def test_upload_rejects_unsupported_format():
    response = client.post(
        "/upload",
        files={"file": ("image.png", b"fake", "image/png")},
    )
    assert response.status_code == 400
    assert "Format non supporté" in response.json()["detail"]


def test_upload_accepts_txt(monkeypatch, tmp_path):
    # Redirect DOCUMENTS_PATH to a tmp dir so we don't pollute ./data
    monkeypatch.setattr(main_module, "DOCUMENTS_PATH", str(tmp_path))
    monkeypatch.setattr(
        main_module, "load_and_index_documents", lambda _path: 7
    )

    response = client.post(
        "/upload",
        files={"file": ("cours.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chunks"] == 7
    assert body["filename"] == "cours.txt"
    assert (tmp_path / "cours.txt").exists()
