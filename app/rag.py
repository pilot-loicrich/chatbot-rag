"""RAG pipeline: PDF ingestion -> ChromaDB -> Gemini LLM (LCEL)."""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond aux questions en te basant UNIQUEMENT sur le "
    "contexte fourni ci-dessous. Si la réponse n'est pas dans le contexte, "
    "dis clairement que tu ne sais pas. Réponds en français, de manière "
    "concise et structurée.\n\n"
    "Contexte :\n{context}"
)


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
# Public API
# ---------------------------------------------------------------------------

def load_and_index_documents(pdf_path: str) -> int:
    """Load a PDF, split it into chunks, and index them in ChromaDB.

    Returns the number of chunks indexed.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return len(chunks)


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def ask_question(question: str) -> str:
    """Answer a question using the indexed documents via a RAG chain (LCEL)."""
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": TOP_K})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    return chain.invoke(question)
