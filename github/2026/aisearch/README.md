# AI Search — Rijksoverheid

Hybrid search + RAG system over [rijksoverheid.nl](https://www.rijksoverheid.nl) built on Elastic Cloud Serverless and local Ollama LLMs.

---

## Architecture

```
rijksoverheid.nl
      │
      ▼
┌─────────────────────┐
│  Elastic Web Crawler│  Docker · crawls Q&A + theme pages
│  (config/crawler.yml│  max 5000 URLs · depth 3
└────────┬────────────┘
         │ raw documents
         ▼
┌─────────────────────┐
│  Ingest Pipeline    │  rijksoverheid-clean
│  (setup_pipeline.py)│  strips nav/footer boilerplate → body_clean
└────────┬────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│  Elasticsearch Index: rijksoverheid-qa-v3        │
│                                                  │
│  title          → BM25 (dutch analyzer)          │
│  body_clean     → BM25 (dutch analyzer)          │
│  body_semantic  → semantic_text (auto-chunked)   │
│                   inference: rijksoverheid-       │
│                   embeddings-v2                  │
│                   model: multilingual-e5-small   │
│                   strategy: word 300 tok/50 ovlp │
└────────┬─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Streamlit App                                  │
│                                                 │
│  Page 1: Search (app.py)                        │
│    - Hybrid RRF / BM25 only / Semantic only     │
│    - Field boost sliders (title, body)          │
│    - RRF rank_window_size + rank_constant       │
│    - Auto-translates English → Dutch            │
│                                                 │
│  Page 2: RAG Chat (pages/1_RAG_Chat.py)         │
│    - Two-pass retrieval                         │
│    - Pass 1: RRF hybrid (ranks best docs)       │
│    - Pass 2: semantic inner_hits (matched chunk │
│              extraction for source display)     │
│    - Full body_clean sent to LLM (not chunks)   │
│    - Streams answer from local Ollama LLM       │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Ollama (local)     │  llama3.1:8b / mistral:7b
│  localhost:11434    │  phi3 / qwen3:0.6b
└─────────────────────┘
```

---

## Prerequisites

- Docker Desktop
- Elastic Cloud Serverless account (ES 9.x)
- Ollama running locally (`ollama serve`)
- Python 3.11+
- `pip install eland[pytorch]` (for model deployment)

### Pull at least one LLM

```bash
ollama pull llama3.1:8b    # best quality
ollama pull mistral:7b     # faster
ollama pull qwen3:0.6b     # used for auto-translation
```

---

## Setup

### 1. Configure environment

```bash
cp .env.example .env
# fill in: ES_HOST, ES_API_KEY, ES_INDEX, CRAWL_URL
```

`.env` variables:

| Variable | Description |
|----------|-------------|
| `ES_HOST` | Elastic Cloud endpoint URL |
| `ES_API_KEY` | Elastic API key |
| `ES_INDEX` | Target index name (default: `rijksoverheid-qa-v3`) |
| `CRAWL_URL` | Site to crawl (default: `https://www.rijksoverheid.nl`) |

### 2. Create inference endpoint in Elastic

Via Kibana Dev Tools:

```json
PUT _inference/text_embedding/rijksoverheid-embeddings-v2
{
  "service": "elasticsearch",
  "service_settings": {
    "model_id": ".multilingual-e5-small_linux-x86_64",
    "num_allocations": 1,
    "num_threads": 1,
    "adaptive_allocations": { "enabled": true }
  },
  "chunking_settings": {
    "strategy": "word",
    "max_chunk_size": 300,
    "overlap": 50
  }
}
```

### 3. Create index + ingest pipeline

```bash
python setup_pipeline.py
```

This:
- Creates `rijksoverheid-clean` ingest pipeline (strips nav boilerplate)
- Creates `rijksoverheid-qa-v3` index with BM25 + semantic_text mapping
- Reindexes raw crawl data → clean index (async, monitor in Kibana)

### 4. Crawl the website

```bash
# pull crawler image
docker compose pull

# run crawl (outputs to Elasticsearch)
docker compose run crawler jruby bin/crawler crawl \
  /home/app/config/crawler.yml \
  --es-config=/home/app/config/elasticsearch.yml
```

Crawl targets: `/vraag-en-antwoord`, `/themas`, `/onderwerpen`
Skips: `/documenten`, news, videos, sign-language pages

### 5. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the app

```bash
streamlit run app.py
```

---

## Pages

### Search (`app.py`)

Full-text + semantic search with UI controls:

- **Search modes**: Hybrid RRF · BM25 Only · Semantic Only
- **BM25 boosts**: title (default 2x) · body (default 1x)
- **RRF settings**: rank window size · rank constant
- **Result count**: 5–50
- **Auto-translation**: English queries auto-translated to Dutch via `qwen3:0.6b`

### RAG Chat (`pages/1_RAG_Chat.py`)

Ask natural language questions, get cited answers:

- **Retrieval**: Two-pass hybrid
  - Pass 1: RRF ranks top-K documents
  - Pass 2: semantic `inner_hits` extracts which passages matched (for source display)
- **Context**: Full `body_clean` sent to LLM (not just matched chunks)
- **LLM**: Streams from local Ollama, temperature=0 for deterministic answers
- **Languages**: Dutch and English questions supported
- **Controls**: model, top-K docs, max chars per doc, context window size, response language

---

## Deploy a Better Embedding Model (optional)

Current model: `.multilingual-e5-small` (300 tok/chunk, already in cluster)

To upgrade:

```bash
python deploy_model.py          # interactive menu
python deploy_model.py e5large  # deploy multilingual-e5-large
python deploy_model.py list     # list all deployed models
```

After deploying, create a new inference endpoint pointing to the new model, create a new index, and reindex.

> Note: `jina-embeddings-v3` and `BAAI/bge-m3` require `trust_remote_code=True` and are not compatible with eland import.

---

## File Reference

| File | Purpose |
|------|---------|
| `app.py` | Streamlit search page |
| `pages/1_RAG_Chat.py` | Streamlit RAG chat page |
| `search_utils.py` | Elasticsearch query logic, language detection, translation |
| `setup_pipeline.py` | One-time index + pipeline setup |
| `deploy_model.py` | Deploy HuggingFace models to Elastic via eland |
| `docker-compose.yaml` | Elastic Web Crawler container |
| `config/crawler.yml` | Crawl target, rules, depth |
| `config/elasticsearch.yml` | Crawler → ES connection config |
| `.env` | Secrets (not committed) |

---

## How Retrieval Works

```
User query (Dutch or English)
        │
        ▼
  is_dutch() check
  → if English: translate via qwen3:0.6b
        │
        ▼
  Pass 1 — RRF Hybrid Search
  ┌──────────────────────────────┐
  │ BM25: title^2 + body_clean   │
  │ Semantic: body_semantic       │
  │ RRF fusion → ranked top-K   │
  └──────────┬───────────────────┘
             │ top-K doc URLs
             ▼
  Pass 2 — Semantic inner_hits
  ┌──────────────────────────────┐
  │ semantic query filtered to   │
  │ ranked doc URLs only         │
  │ inner_hits → matched chunks  │
  │ (used for source display)    │
  └──────────┬───────────────────┘
             │
             ▼
  Build LLM context
  → full body_clean per doc (not chunks)
  → up to max_chars per document
             │
             ▼
  Stream answer from Ollama LLM
```

---

## Crawl Config Notes

Allowed paths:
- `/vraag-en-antwoord/*` — Q&A pages (primary content)
- `/themas/*` — theme pages
- `/onderwerpen/*` — topic pages

Denied:
- `/documenten/*` — PDFs, formal documents
- `*.pdf`, `*.mp4`, `*/gebarentaal/*` — media
- `/nieuws/*`, `/actueel/*` — news (frequently changing, low Q&A value)
