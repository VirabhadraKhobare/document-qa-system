# Document Q&A System

A small Streamlit app that lets you upload searchable PDFs, creates sentence-transformer embeddings, stores them in a FAISS index, and then lets you ask questions over your documents using a generative model (optional Google GenAI/Gemini).

This repository contains a single-file Streamlit app (`app.py`) and a minimal `requirements.txt`.

## What this fixes / why update was needed
- The original `requirements.txt` and `README.md` contained stray markdown code fences which can break some automated installers or confuse users. This repo cleanup removes those fences and adds a clear README with setup and run instructions.

## Prerequisites
- Python 3.10+ recommended (3.11 also fine).
- Windows with PowerShell (commands below are PowerShell-friendly).
- Optional: GPU is not required. `faiss-cpu` is used by default.

## Setup (PowerShell)
Open PowerShell in the repository root and run:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you get issues installing `faiss-cpu` on Windows, you can try a prebuilt wheel or install it via conda. Example (requires miniconda/anaconda):

```powershell
conda create -n docqa python=3.11 -y
conda activate docqa
conda install -c conda-forge faiss-cpu -y
pip install -r requirements.txt
```

## Running the app
Start the Streamlit app with:

```powershell
streamlit run app.py
```

Open the local URL that Streamlit prints (usually http://localhost:8501) in your browser.

## Optional: Google GenAI (Gemini)
If you want to use Google Generative AI models (Gemini) you can put your key in a `.env` file in the repo root or add it to Streamlit secrets. Example `.env`:

```
GOOGLE_API_KEY=your_key_here
```

Note: access to Gemini may require an enabled Google Cloud project and appropriate permissions.

## Files
- `app.py` — main Streamlit application
- `requirements.txt` — pip requirements
- `chat_history.json` — generated at runtime to persist chat history (created automatically)

## Troubleshooting
- If Streamlit shows import errors, ensure the virtual environment is activated and `pip install -r requirements.txt` completed successfully.
- If PDFs extract no text, they might be scanned images. Use OCR before uploading or provide searchable PDFs.

## Next steps (suggested)
- Add unit tests for text extraction and chunking functions.
- Add a small Dockerfile for reproducible deployments.

---

If you want, I can also scan `app.py` for runtime issues and make small fixes (for example, safer FAISS load checks, or removing optional Google GenAI imports when key not present). Tell me if you'd like that.