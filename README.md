# Chatbot RAG — Gemini API

Chatbot de questions/réponses sur tes documents PDF, basé sur une architecture
**RAG** (Retrieval-Augmented Generation) avec **FastAPI**, **ChromaDB**,
**LangChain** et l'**API Gemini** de Google.

## Architecture

```
   PDF  ──►  PyPDFLoader ──►  Text Splitter ──►  Embeddings (Gemini) ──►  ChromaDB
                                                                             │
   Question utilisateur ──►  Retriever (top-k) ──►  Prompt ──►  Gemini LLM  ─┘──►  Réponse
```

| Composant            | Choix technique                       |
|----------------------|---------------------------------------|
| Framework API        | FastAPI                               |
| LLM                  | `gemini-2.0-flash`                    |
| Embeddings           | `text-embedding-004`                  |
| Vector store         | ChromaDB (persisté sur disque)        |
| Chaîne RAG           | LangChain LCEL                        |
| Loader de documents  | PyPDFLoader                           |

## Prérequis

- Python 3.10+
- Une clé API Gemini : https://aistudio.google.com/app/apikey

## Installation

```bash
# 1. Cloner le repo
git clone https://github.com/pilot-loicrich/chatbot-rag.git
cd chatbot-rag

# 2. Créer un environnement virtuel
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé API
cp .env.example .env      # (sous Windows : copy .env.example .env)
# puis édite .env et colle ta clé Gemini
```

## Lancement (local)

```bash
uvicorn app.main:app --reload
```

L'API sera disponible sur http://localhost:8000.
La documentation interactive Swagger est sur http://localhost:8000/docs.

## Lancement (Docker)

```bash
docker build -t chatbot-rag .
docker run -p 8000:8000 --env-file .env chatbot-rag
```

## Endpoints

| Méthode | Route       | Description                                      |
|---------|-------------|--------------------------------------------------|
| GET     | `/`         | Health check racine                              |
| GET     | `/health`   | Health check dédié                               |
| POST    | `/upload`   | Upload + indexation d'un document PDF            |
| POST    | `/ask`      | Pose une question au chatbot                     |

### Exemples

**Upload d'un PDF**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@mon_document.pdf"
```

**Poser une question**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quel est le sujet principal du document ?"}'
```

## Tests

```bash
pytest tests/
```

## Structure du projet

```
chatbot-rag/
├── app/
│   ├── __init__.py
│   ├── main.py        # API FastAPI
│   ├── rag.py         # Pipeline RAG (ingestion + retrieval + génération)
│   └── config.py      # Chargement des variables d'environnement
├── data/
│   ├── documents/     # PDFs uploadés
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

| Variable         | Description                         | Requis |
|------------------|-------------------------------------|--------|
| `GEMINI_API_KEY` | Clé API Google Generative AI        | Oui    |

## Roadmap

- [x] **Phase 1** — Modernisation du pipeline (LCEL, `gemini-2.0-flash`, `text-embedding-004`, caching)
- [ ] **Phase 2** — Citations des sources, support multi-formats (.docx, .txt, .md)
- [ ] **Phase 3** — Historique conversationnel, streaming, UI web
- [ ] **Phase 4** — Évaluation (Ragas), monitoring, auth
- [ ] **Phase 5** — Ingestion Moodle pour assistant de cours

## Licence

Projet pédagogique — usage libre.
