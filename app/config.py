import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CHROMA_DB_PATH = "./data/chroma_db"
DOCUMENTS_PATH = "./data/documents"