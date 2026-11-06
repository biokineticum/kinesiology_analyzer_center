# Kinesiology Analyzer Center

**Offline-first desktop suite** for clinical biomechanics PDF/CSV extraction, interactive multi-sensor datasets, local RAG over research PDFs, and local LLM fine-tuning.

Built for physios, sports scientists and labs that need **GDPR-minded, on-prem** workflows (Biodex / Humac / Noraxon-style reports, force data) without sending patient files to the cloud.

## Features

- PDF/table extraction with PyMuPDF (Biodex / Humac–oriented parsers; extensible for Noraxon-style exports)
- Dataset consolidation to Excel/CSV and Plotly analytics in Streamlit
- AI routing: local **Ollama**, plus optional Gemini / OpenAI via REST
- Hybrid TF‑IDF RAG over scientific PDFs; JSONL dataset tools under `finetuner/`
- Native shell: Streamlit + **PySide6** desktop wrapper

## Requirements

- Python 3.10 or 3.11
- Windows 10/11 well supported; macOS/Linux via `run_mac.sh` / Streamlit

## Install

```bash
git clone https://github.com/biokineticum/kinesiology_analyzer_center.git
cd kinesiology_analyzer_center
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Optional GPU fine-tuning: follow Unsloth’s Windows guide, or use the in-app **Stable HF PEFT** path on Apple Silicon.

## Run

**Windows (recommended):** double-click `Uruchom_Kinesiology_Analyzer.vbs`  
**Windows debug:** `run_debug.bat`  
**macOS / Linux desktop wrapper:**

```bash
chmod +x run_mac.sh
./run_mac.sh
```

**Browser / any OS:**

```bash
streamlit run pdf_parser_trainer.py
```

Put raw files in `inbox/` (gitignored). Do not commit patient data.

## Privacy

Prefer **Local Ollama** so parsing, RAG and training stay on your machine. `.gitignore` excludes `inbox/`, `processed/`, and model weights.

---

## Author

**Dariusz Mosler** (Biokineticum) — physiotherapy, biomechanics, sports science  
🌐 [biokineticum.com](https://biokineticum.com) · 📰 [Substack](https://biokineticum.substack.com) · 𝕏 [@Biokineticum](https://x.com/Biokineticum)

