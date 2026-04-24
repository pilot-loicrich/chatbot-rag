# Chatbot RAG — Gemini API

Chatbot de questions/réponses sur tes documents, basé sur une architecture
**RAG** (Retrieval-Augmented Generation) avec **FastAPI**, **ChromaDB**,
**LangChain** et l'**API Gemini** de Google.

## Fonctionnalités

- Upload et indexation de documents `.pdf`, `.docx`, `.txt`, `.md`
- Question / réponse en français sur le contenu indexé
- **Citations des sources** dans chaque réponse (fichier + page)
- Pipeline LCEL moderne, singletons cachés (performance)
- Prêt pour Docker

## Architecture

```
   Document (.pdf / .docx / .txt / .md)
        │
        ▼
   Loader (PyPDFLoader / Docx2txtLoader / TextLoader)
        │
        ▼
   Text Splitter (Recursive, 1000 / 200 overlap)
        │
        ▼
   Embeddings (Gemini text-embedding-004)
        │
        ▼
   ChromaDB (persisté sur disque)
        ▲                                           Réponse + sources
        │                                                   ▲
   Retriever (top-k = 4)                                    │
        │                                                   │
   Question ──────► Prompt (contextualisé) ──► Gemini LLM ──┘
                                               (gemini-2.0-flash)
```

| Composant         | Choix technique                 |
|-------------------|---------------------------------|
| Framework API     | FastAPI                         |
| LLM               | `gemini-2.0-flash`              |
| Embeddings        | `text-embedding-004`            |
| Vector store      | ChromaDB (persistant)           |
| Chaîne RAG        | LangChain LCEL                  |
| Loaders           | PyPDFLoader, Docx2txtLoader, TextLoader |

## Prérequis

- Python 3.10+
- Une clé API Gemini : https://aistudio.google.com/app/apikey

## Installation

```bash
git clone https://github.com/pilot-loicrich/chatbot-rag.git
cd chatbot-rag

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env      # Windows : copy .env.example .env
# puis édite .env et colle ta clé Gemini
```

## Lancement (local)

```bash
uvicorn app.main:app --reload
```

Disponible sur http://localhost:8000 — Swagger UI sur `/docs`.

## Lancement (Docker)

```bash
docker build -t chatbot-rag .
docker run -p 8000:8000 --env-file .env chatbot-rag
```

## Endpoints

| Méthode | Route       | Description                                          |
|---------|-------------|------------------------------------------------------|
| GET     | `/`         | Health check racine                                  |
| GET     | `/health`   | Health check dédié                                   |
| POST    | `/upload`   | Upload + indexation d'un document (pdf/docx/txt/md)  |
| POST    | `/ask`      | Pose une question, retourne la réponse + les sources |

### Exemples

**Upload**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@mon_document.pdf"
```

Réponse :
```json
{
  "message": "Document 'mon_document.pdf' indexé avec succès",
  "chunks": 42,
  "filename": "mon_document.pdf"
}
```

**Poser une question**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quel est le sujet principal du document ?"}'
```

Réponse :
```json
{
  "question": "Quel est le sujet principal du document ?",
  "answer": "Le document traite de…",
  "sources": [
    { "source": "mon_document.pdf", "page": 3, "snippet": "…" },
    { "source": "notes.md",        "page": null, "snippet": "…" }
  ]
}
```

## Tests

```bash
pytest tests/ -v
```

## Structure du projet

```
chatbot-rag/
├── app/
│   ├── __init__.py
│   ├── main.py        # API FastAPI
│   ├── rag.py         # Pipeline RAG (ingestion + retrieval + génération)
│   └── config.py      # Variables d'environnement
├── data/
│   ├── documents/     # Documents uploadés
│   └── chroma_db/     # Index vectoriel (persisté)
├── tests/
│   └── test_main.py
├── Dockerfile
├── .gitlab-ci.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Variables d'environnement

| Variable         | Description                   | Requis |
|------------------|-------------------------------|--------|
| `GEMINI_API_KEY` | Clé API Google Generative AI  | Oui    |

## Roadmap

- [x] **Phase 1** — Modernisation du pipeline (LCEL, `gemini-2.0-flash`, `text-embedding-004`, caching)
- [x] **Phase 2** — Citations des sources, support multi-formats (.pdf, .docx, .txt, .md)
- [ ] **Phase 3** — Historique conversationnel, streaming, UI web
- [ ] **Phase 4** — Évaluation (Ragas), monitoring, auth
- [ ] **Phase 5** — Ingestion Moodle pour assistant de cours

## Licence

Projet pédagogique — usage libre.
