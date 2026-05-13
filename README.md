# AbbVie Testing Dashboard

Internal Streamlit app for comparing product testing results, browsing the document library, running an AI Assistant over study docs, and generating polished reports.

> **Internal use only.** The `Library/` folder, the testing workbook, and the ILIAD gateway are not redistributable. This repo contains code only.

---

## Prerequisites

- **Python 3.10+**
- **ILIAD API key** (AbbVie internal LLM gateway) — required for the AI Assistant, Report Generator, and vector index builds. Without it, Product Comparator and Document Library still work.
- **Network access** to `iliad-emerging-api.abbvienet.com` (on-network or VPN).
- The **testing workbook** (`testing fpt files.xlsx`) and the **Library/** folder. Obtain from the shared drive and either:
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
| `DASHBOARD_ADMINS` | no | unset → everyone is admin | Comma-separated usernames allowed to reindex/sync |

Place the workbook and `Library/` at the repo root, or set the env vars to their real locations.

---

## Running the dashboard

```bash
streamlit run comparator.py
```

Opens on `http://localhost:8501`. Four tabs:

1. **Product Comparator** — reads the Excel workbook; charts, stats, export.
2. **Document Library** — browser + admin panel over `Library/`. Auto-rebuilds `library_index.json` on first launch.
3. **AI Assistant** — chatbot backed by ILIAD `claude-4.5-sonnet` with keyword + semantic retrieval.
4. **Report Generator** — upload a PDF/DOCX/PPTX, get a polished summary or full report; admins can save directly to the library.

### Admin vs. teammate

`DASHBOARD_ADMINS` controls who can rewrite shared files. Set it to a comma-separated list of usernames (lowercase, match `$CDSW_USER` / `$USERNAME` on CML):

```
DASHBOARD_ADMINS=piercax,jdoe,asmith
```

**Admins** can:
- Sync / Rebuild the library index
- Build / Update / Rebuild the vector index
- Organize files on disk
- Save Report Generator output directly to the library

**Everyone else** can browse the library, use the AI Assistant, run the comparator, and use the Report Generator (download only). They see informational banners telling them who to ask if something is missing or out of date.

**Default when unset:**
- **Local dev** → current user is treated as admin (frictionless single-user development).
- **On CML** → nobody is admin, everyone sees the read-only view. Deliberately fails closed: the whole point of admin gating is preventing concurrent writes on a shared deployment, so an accidental oversight must not grant everyone write access. Remember to set `DASHBOARD_ADMINS` on the CML project before teammates arrive.

---

## Vector index maintenance

The AI Assistant needs two files inside `Library/`:

- `library_index.json` — metadata. Built/updated by the app on first launch; refreshed via the **Sync** or **Rebuild** buttons in the Library tab.
- `library_vectors.npz` — embeddings. Built / updated from the AI Assistant tab's index controls.

Rebuild order is **index first, then vectors**.

You can also build vectors from the command line:

```bash
python build_index.py   # incremental — only embeds new docs
```

---

## Deploying on CML (Cloudera ML)

1. Create a CML Project from this Git repo.
2. Set **Project Environment Variables**:
   - `ILIAD_API_KEY` — required
   - `DASHBOARD_ADMINS` — e.g. `piercax`
   - `TESTING_FILE_PATH` / `LIBRARY_PATH` if data lives outside the project
3. Start a Session with Python 3.10+ and run `pip install -r requirements.txt`.
4. Launch the dashboard as a CML **Application** with command:
   ```bash
   streamlit run comparator.py --server.port $CDSW_APP_PORT --server.address 127.0.0.1 --server.headless true
   ```

---

## Troubleshooting

### "The workbook is currently open on someone else's machine"

Someone has `testing fpt files.xlsx` open in Excel on the network share. Close it (or ask them to close it), then click **↻ Retry now** in the error banner.

### AI Assistant says "ILIAD rejected the API key"

The `ILIAD_API_KEY` is wrong, expired, or the gateway is rejecting it. Verify the variable is set (terminal: `echo %ILIAD_API_KEY%` on Windows CMD, `$env:ILIAD_API_KEY` in PowerShell). On CML, check **Project Settings → Environment Variables**. If you rotated the key, restart the Streamlit app so it picks up the new value.

### "The document index has not been built yet" (persistent)

The admin hasn't built the vector index yet. Go to the **AI Assistant** tab, open the **📚 Document index** expander, and click **⚡ Build document index**. First build takes ~1–2 minutes per ~60 docs; there's a pre-flight ETA above the button.

### "Library out of sync" warning

The index is missing or has extra entries compared to disk. Click **⚡ Sync (incremental)** inside the warning banner — fast, processes only new/changed/deleted files. If that doesn't resolve it (rare), use **↻ Full rescan** to rebuild the index from scratch.

### A new document isn't showing up in search

Admins: click **⚡ Sync** at the top of the Library tab. That single click runs the metadata scan AND updates the AI vector index in one pass, so the new file joins both the library browser and semantic search together.

(If the vector index hasn't been built yet at all, Sync will skip the AI step and prompt you to use **⚡ Build document index** on the AI Assistant tab first.)

### Vector index looks corrupted (all searches return empty, or loading fails)

Admins:
1. Delete `Library/library_vectors.npz` from the shared drive
2. Restart the Streamlit app
3. Go to AI Assistant tab → **⚡ Build document index**

The file is rebuilt deterministically from `library_index.json`, so nothing is lost.

### Some documents never get embedded (ILIAD hangs on them)

A handful of documents have historically caused the ILIAD embedding endpoint to stall. They're listed in `build_index.py` under `SKIP_ENTRY_IDS`. If you find a new one:

1. Run `python build_index.py` from a terminal — the hung document will be visible in the progress output
2. Add its `entry_id` (shown in the progress line) to `SKIP_ENTRY_IDS`
3. Re-run; it should now skip cleanly

Periodically try removing entries from the list — ILIAD updates sometimes fix the underlying issue.

### AI Assistant responses cite the wrong documents

The retrieval pulled similarly-named docs. Two levers:
- Mention the study number explicitly in your question (e.g. "in study 1745-D76-054, ...") — triggers full-text injection of that specific study
- Ask the admin to run **🔄 Full rebuild** of the vector index; over time the index can pick up noise from removed or renamed files

---

## Rotating the ILIAD API key

1. Get the new key from your AbbVie contact for the ILIAD gateway
2. CML: **Project Settings → Environment Variables** → edit `ILIAD_API_KEY` → Save
3. Restart the Streamlit Application from the Applications tab
4. Local dev: edit `.env`, then restart `streamlit run comparator.py`

---

## For teammates cloning this repo for the first time

```bash
# One-time setup
git clone <repo-url>
cd Dashboard
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Pulling updates
git pull
pip install -r requirements.txt   # in case deps changed

# Running
streamlit run comparator.py
```

If `git pull` prompts for a password, use a GitHub personal access token (not your account password). Create one at [github.com/settings/tokens](https://github.com/settings/tokens) with the `repo` scope.

---

## Repository layout

```
.
├── comparator.py           # Streamlit dashboard (main entry point)
├── config.py               # Env-driven path config
├── auth.py                 # Admin detection for shared-state writes
├── iliad_client.py         # Shared ILIAD gateway client (chat + embeddings)
├── title_utils.py          # Shared document-title normalisation helpers
├── vector_store.py         # Embedding + semantic search layer
├── doc_text.py             # Text extraction (pdf/docx/pptx)
├── build_index.py          # Standalone incremental vector index builder
├── .streamlit/config.toml  # Streamlit UI config
├── requirements.txt
├── .env.example
├── CHANGELOG.md
└── .gitignore
```

Not in the repo (gitignored): `Library/`, `testing fpt files.xlsx`, `.env`, `settings.json`.

---

## Security notes

- `.env` and `settings.json` are gitignored. **Never commit real API keys.**
- If a key is ever pushed (even to a deleted branch), treat it as compromised and rotate.
- The ILIAD client uses `verify=False` because the corporate proxy uses a self-signed cert; this is intentional and scoped to `iliad-emerging-api.abbvienet.com`.
- `DASHBOARD_ADMINS` is a UI guardrail, not a security boundary — anyone with shell access to the machine can edit library files directly. Its job is to prevent well-intentioned teammates from stomping on each other's writes.
