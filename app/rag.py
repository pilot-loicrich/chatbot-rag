"""RAG pipeline: document ingestion -> ChromaDB -> Gemini LLM (LCEL).

Supports .pdf, .docx, .txt and .md documents. Returns answers together with
their source citations (filename + page when available).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional, TypedDict

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHROMA_DB_PATH, GEMINI_API_KEY

# ---------------------------------------------------------------------------
# Models & constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.0-flash"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
SNIPPET_LEN = 240

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond aux questions en te basant UNIQUEMENT sur "
    "le contexte fourni ci-dessous. Si la réponse n'est pas dans le contexte, "
    "dis clairement que tu ne sais pas. Réponds en français, de manière "
    "concise et structurée.\n\n"
    "Contexte :\n{context}"
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Source(TypedDict):
    source: str
    page: Optional[int]
    snippet: str


class RAGAnswer(TypedDict):
    answer: str
    sources: list[Source]


# ---------------------------------------------------------------------------
# Cached singletons (instantiated once per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=get_embeddings(),
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# Document loading (dispatch by extension)
# ---------------------------------------------------------------------------

def _load_document(file_path: str) -> list[Document]:
    """Dispatch loading by file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    if ext == ".docx":
        return Docx2txtLoader(file_path).load()
    if ext in (".txt", ".md"):
        return TextLoader(file_path, encoding="utf-8").load()
    raise ValueError(
        f"Format non supporté: '{ext}'. Formats acceptés: "
        f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_index_documents(file_path: str) -> int:
    """Load a document, split it into chunks, index them in ChromaDB.

    The original filename is stored in each chunk's metadata (key
    ``source_name``) so it can be surfaced in citations later.
    Returns the number of chunks indexed.
    """
    documents = _load_document(file_path)

    # Stamp the display name on every page before splitting so it propagates.
    display_name = Path(file_path).name
    for doc in documents:
        doc.metadata.setdefault("source_name", display_name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    get_vectorstore().add_documents(chunks)
    return len(chunks)


def _format_context(docs: Iterable[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source_name") or doc.metadata.get("source", "?")
        page = doc.metadata.get("page")
        header = f"[Source {i} — {src}" + (f", page {page + 1}" if isinstance(page, int) else "") + "]"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _doc_to_source(doc: Document) -> Source:
    src = doc.metadata.get("source_name") or Path(
        doc.metadata.get("source", "unknown")
    ).name
    raw_page: Any = doc.metadata.get("page")
    page: Optional[int] = (raw_page + 1) if isinstance(raw_page, int) else None
    snippet = doc.page_content.strip().replace("\n", " ")
    if len(snippet) > SNIPPET_LEN:
        snippet = snippet[:SNIPPET_LEN].rstrip() + "…"
    return {"source": src, "page": page, "snippet": snippet}


def _dedupe_sources(sources: list[Source]) -> list[Source]:
    seen: set[tuple[str, Optional[int]]] = set()
    unique: list[Source] = []
    for s in sources:
        key = (s["source"], s["page"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique


def ask_question(question: str) -> RAGAnswer:
    """Answer a question using the indexed documents, with source citations."""
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    chain = prompt | get_llm() | StrOutputParser()

    answer = chain.invoke(
        {"context": _format_context(docs), "question": question}
    )

    sources = _dedupe_sources([_doc_to_source(d) for d in docs])
    return {"answer": answer, "sources": sources}
