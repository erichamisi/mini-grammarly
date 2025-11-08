# Mini-Grammarly (Streamlit)

Grammar/spell checker using LanguageTool with optional paraphrasing (transformers).

## Local dev
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Notes
- Default engine uses **LanguageTool Public API** (rate-limited). For heavy use, switch to **Local server** in the sidebar (requires Java) or set `LT_REMOTE_SERVER_URL` in Streamlit Secrets to use your own LT server.
- Paraphraser downloads a model on first use; you can skip transformers if you don’t need it.

## Deploy (Streamlit Community Cloud)
- Connect this repo, set **Python 3.12** (via `runtime.txt`), App file = `app.py`.
- (Optional) Add a secret:
  ```
  LT_REMOTE_SERVER_URL = https://your-lt-server.example.com
  ```

## Other deploys
### Render/Railway/Fly (Procfile)
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

## License
MIT (add your own LICENSE file if desired)
