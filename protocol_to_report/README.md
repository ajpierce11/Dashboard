# Protocol → Report Converter

A local web application that converts study protocols (.docx) into study reports by:

1. **Tense conversion** — Automatically converts future/present tense to past tense using spaCy NLP, preserving all document formatting
2. **Report sections** — Prompts you to add Deviations, Results (with figures), and Conclusions sections
3. **Export** — Downloads the final report as a .docx with original formatting intact

**Fully offline — no API calls, no costs, no data leaves your machine.**

## Setup

### Prerequisites
- Python 3.9 or higher

### Quick Start (Windows)
1. Double-click **setup.bat** (first time only)
2. Double-click **run.bat** to launch

### Manual Setup
```bash
cd protocol_to_report
pip install -r requirements.txt
python -m spacy download en_core_web_md
python app.py
```

Open **http://localhost:5000** in your browser.

## How It Works

**Step 1** — Upload your study protocol .docx

**Step 2** — Convert to past tense (review all changes, or skip)

**Step 3** — Add Deviations, Results, and Conclusions sections with optional figures

**Step 4** — Download your completed study report

## Notes
- All processing runs locally via spaCy + pyinflect. No data is sent anywhere.
- Formatting (fonts, bold/italic, tables, images, headers, lists) is preserved.
- Definitions and universal truths are intentionally kept in present tense.
- Review the change log after conversion for any edge cases.
