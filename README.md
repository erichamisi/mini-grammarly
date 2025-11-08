# Mini-Grammarly — Grammar, Style, Rewrite & Plagiarism

A Streamlit app that checks grammar, paraphrases text, reports readability, and detects plagiarism.

## Features
- **Grammar & Spelling**: LanguageTool via Public API (default), Local Java server, or your own Remote server.
- **Paraphrasing**:
  - **Hugging Face Inference API** (recommended for Streamlit Cloud; no heavy installs).
  - Optional **local Transformers** (install `transformers`, `sentencepiece`, `torch`).
- **Readability**: Flesch, F-K Grade, Fog, SMOG, Coleman–Liau, Dale–Chall.
- **Plagiarism**:
  - **Local**: compare against uploaded TXT/PDF/DOCX and/or pasted references (TF-IDF cosine similarity + overlap highlights).
  - **External API**: placeholder stub you can wire to a provider (set `PLAG_API_KEY`).

## Quick Start (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### (Optional) Local paraphrasing
```bash
pip install --upgrade transformers sentencepiece torch
```

## Streamlit Cloud
1. Push this repo to GitHub.
2. Create a new Streamlit app from your repo (`app.py` as the main file).
3. In Streamlit: **Settings → Secrets**  
   - Set `HF_API_TOKEN = <your_hugging_face_token>` (recommended for paraphrasing).
   - (Optional) `LT_REMOTE_SERVER_URL = https://your-languagetool.example.com`
   - (Optional) `PLAG_API_KEY = <your_plagiarism_api_key>`
4. **Clear cache** and restart if you push an update.

## Using the App
1. Paste text.
2. Choose your language and engine in the sidebar.
3. Click **Check grammar & spelling**.
4. Enable **Paraphrase** to rewrite with a selected tone.
5. **Plagiarism**:
   - Turn on “Enable plagiarism checking”.
   - Choose **Local** or **External API**.
   - For **Local**, upload reference files and/or paste text blocks (separate each with a line containing `---`).
   - Click **Run plagiarism check**.
   - Export the report as **CSV**, **DOCX**, or **PDF**.

## External API Plagiarism (stub)
See `run_external_plagiarism_stub()` in `app.py`. Normalize provider response to:
```python
{
  "Source": "Provider URL or title",
  "Similarity": 0.61,  # 0..1
  "OverlapsFound": 2,
  "HighlightedTargetHTML": "<mark>overlap</mark> ...",
  "OverlapSamples": ["sample1", "sample2"]
}
```
Set the API key as `PLAG_API_KEY` in **Secrets**.

## Notes
- Public LanguageTool API is rate-limited. For high volume, run your own LT server or set `LT_REMOTE_SERVER_URL`.
- Local plagiarism **only** compares against references you provide (uploads + pasted). It doesn’t search the web.
- PDF export uses ReportLab; DOCX export uses python-docx.

## License
MIT
