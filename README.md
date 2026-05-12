# AbbVie Testing Dashboard

Internal Streamlit app for comparing product testing results, browsing the document library, and running an AI Assistant over study docs.

> **Internal use only.** The `Library/` folder, the testing workbook, and the ILIAD gateway are not redistributable. This repo contains code only.

---

## Prerequisites

- **Python 3.10+**
- **ILIAD API key** (AbbVie internal LLM gateway) — required for the AI Assistant tab and for building the vector index. Without it, the other dashboard tabs still work.
- **Network access** to `iliad-emerging-api.abbvienet.com` (on-network or VPN).
- The **testing workbook** (`testing fpt files.xlsx`) and the **Library/** folder. These are *not* in the repo; obtain them from the shared drive and either:
  - Drop them at the repo root (default), or
  - Point env vars `TESTING_FILE_PATH` / `LIBRARY_PATH` at their real locations.

---

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd Dashboard

# 2. Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS/CML:
source .venv/bin/activate

# 3. Install deps
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
# then edit .env and set ILIAD_API_KEY=...
```

### Environment variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ILIAD_API_KEY` | yes (for AI features) | — | Auth for ILIAD chat + embeddings |
| `TESTING_FILE_PATH` | no | `./testing fpt files.xlsx` | Override workbook path |
| `LIBRARY_PATH` | no | `./Library` | Override document library path |

Place the workbook and `Library/` at the repo root, or set the env vars to their real locations.

---

## Running the dashboard

```bash
streamlit run comparator.py
```

Opens on `http://localhost:8501`. Three tabs:

1. **Product Comparator** — reads the Excel workbook; charts + stats + export.
2. **Document Library** — browser + admin panel over `Library/`. Auto-rebuilds `library_index.json` on launch.
3. **AI Assistant** — chatbot backed by ILIAD `gpt-4o-mini` with keyword + semantic retrieval.

## Vector index maintenance

The AI Assistant needs two files inside `Library/`:

- `library_index.json` — metadata. Built/updated by the app on launch.
- `library_vectors.npz` — embeddings. Built separately.

Rebuild order is **index first, then vectors**.

```bash
# Incremental build (recommended — embeds only new docs)
python build_index.py
```

You can also use the "Build index" button in the AI Assistant tab.

---

## Deploying on CML (Cloudera ML)

1. Create a CML Project from this Git repo.
2. Set **Project Environment Variables**: `ILIAD_API_KEY`, and `TESTING_FILE_PATH` / `LIBRARY_PATH` if the data lives outside the project.
3. Start a Session with Python 3.10+ and run `pip install -r requirements.txt`.
4. Launch the dashboard as a CML **Application** with command:
   ```bash
   streamlit run comparator.py --server.port $CDSW_APP_PORT --server.address 127.0.0.1
   ```

---

## Repository layout

```
.
├── comparator.py           # Streamlit dashboard (main entry point)
├── config.py               # Env-driven path config
├── iliad_client.py         # Shared ILIAD gateway client (chat + embeddings)
├── title_utils.py          # Shared document-title normalisation helpers
├── vector_store.py         # Embedding + semantic search layer
├── doc_text.py             # Text extraction (pdf/docx/pptx)
├── build_index.py          # Standalone incremental vector index builder
├── .streamlit/config.toml  # Streamlit UI config
├── requirements.txt
├── .env.example
└── .gitignore
```

Not in the repo (gitignored): `Library/`, `testing fpt files.xlsx`, `.env`, `settings.json`.

---

## Security notes

- `.env` and `settings.json` are gitignored. **Never commit real API keys.**
- If a key is ever pushed (even to a deleted branch), treat it as compromised and rotate.
- The ILIAD client uses `verify=False` because the corporate proxy uses a self-signed cert; this is intentional and scoped to `iliad-emerging-api.abbvienet.com`.
