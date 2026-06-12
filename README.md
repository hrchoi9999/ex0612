# ex0612

Streamlit app for PDF and CSV question answering with OpenAI embeddings and ChromaDB.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io/
2. Select repository: `hrchoi9999/ex0612`
3. Branch: `main`
4. Main file path: `app.py`
5. Deploy the app.

The app asks for `OPENAI_API_KEY` on the page, so no API key should be committed to this repository.
