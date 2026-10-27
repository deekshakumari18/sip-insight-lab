# SIP Insight Lab

An AI-assisted discovery engine for exploring public conversations about mutual-fund SIP experiences in India. Built for a Product Management graduation project.

## What it does

- Imports a CSV of public conversations with source metadata.
- Uses the Gemini API in JSON mode to classify reported themes, goals, context, behaviour, workarounds and outcomes.
- Counts themes and allows researchers to inspect classifications and verify exact evidence quotes.
- Provides a lightweight RAG-style question-answering interface using TF-IDF retrieval followed by an answer grounded in retrieved source records.
- Compares AI labels with a manually labelled golden set.
- Exports analysis results for an AI Evidence Log.

## Important limitations

- AI labels are suggestions for researcher review, not verified causes.
- Theme frequency is not proof of importance or causality.
- TF-IDF is lexical retrieval, not embedding-based vector search. It is a deliberately small first version; semantic retrieval can be added after the pipeline is validated.
- Public online feedback is self-selected and is not representative of all investors.
- Synthetic demo rows are for testing only and must not be reported as research evidence.
- The app does not permanently store uploaded data. Download outputs before ending the session.
- Do not upload private financial information, PAN details, account numbers, portfolio screenshots, or identifiable interview notes.

## Run locally

1. Install Python 3.11 or 3.12.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your Gemini API key to an environment variable named `GEMINI_API_KEY`, or create `.streamlit/secrets.toml` containing:

   ```toml
   GEMINI_API_KEY = "paste_your_key_here"
   ```

   Never commit `secrets.toml` or publish the key.

4. Run:

   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `README.md`, and `.gitignore` to a GitHub repository.
2. Open https://share.streamlit.io and select **Create app**.
3. Select your repository, branch, and `app.py` as the main file.
4. Under Advanced settings / Secrets, add:

   ```toml
   GEMINI_API_KEY = "paste_your_key_here"
   ```

5. Deploy and test the deployed app with the synthetic demo first.

## Input CSV format

Required: `text`

Recommended metadata: `id`, `source`, `source_url`, `date`, `platform`

One public conversation or review per row. Preserve original source links. Do not invent missing source metadata.

## Golden set format

Use a separate CSV with columns `id`, `primary_theme`, and optionally `notes`. Human-label the items before comparing them with AI output. Use one of the theme labels displayed in the Evaluate tab.
