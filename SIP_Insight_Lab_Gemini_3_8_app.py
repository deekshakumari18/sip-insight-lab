import io
import json
import os
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_TITLE = "SIP Insight Lab"
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
THEMES = [
    "affordability_or_cashflow",
    "return_expectation_gap",
    "market_uncertainty",
    "payment_or_autopay_issue",
    "product_or_process_confusion",
    "information_or_trust_gap",
    "goal_or_priority_change",
    "habit_or_attention",
    "external_influence",
    "other",
    "insufficient_evidence",
    "not_sip_related",
]

st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")


def get_api_key():
    """Read a Gemini API key from Streamlit secrets or environment variables."""
    try:
        value = st.secrets.get("GEMINI_API_KEY", "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "").strip()


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def template_data() -> pd.DataFrame:
    return pd.DataFrame(columns=["id", "text", "source", "source_url", "date", "platform"])


def golden_template() -> pd.DataFrame:
    return pd.DataFrame(columns=["id", "primary_theme", "notes"])


def synthetic_demo() -> pd.DataFrame:
    # Explicitly synthetic examples for testing the app only. They are not research evidence.
    rows = [
        ("demo_01", "SYNTHETIC DEMO: My salary comes late some months, so the SIP debit failed when there wasn't enough balance.", "Synthetic demo", "", "", "Unknown"),
        ("demo_02", "SYNTHETIC DEMO: I started a SIP because I wanted to save for a home deposit over several years.", "Synthetic demo", "", "", "Unknown"),
        ("demo_03", "SYNTHETIC DEMO: I checked my investment every day because I expected the value to rise quickly, but I did not understand the short-term changes.", "Synthetic demo", "", "", "Unknown"),
        ("demo_04", "SYNTHETIC DEMO: I paused my SIP for two months after an unexpected medical expense; I restarted later.", "Synthetic demo", "", "", "Unknown"),
        ("demo_05", "SYNTHETIC DEMO: I couldn't tell whether my AutoPay mandate was active, so I checked my bank app and contacted support.", "Synthetic demo", "", "", "Unknown"),
        ("demo_06", "SYNTHETIC DEMO: A friend suggested investing, but I wasn't sure how to judge whether my SIP was progressing as expected.", "Synthetic demo", "", "", "Unknown"),
        ("demo_07", "SYNTHETIC DEMO: I stopped making contributions when I needed cash for a change in my living situation.", "Synthetic demo", "", "", "Unknown"),
        ("demo_08", "SYNTHETIC DEMO: I rarely open my investing app now because the SIP runs automatically and I have not needed to change anything.", "Synthetic demo", "", "", "Unknown"),
    ]
    return pd.DataFrame(rows, columns=["id", "text", "source", "source_url", "date", "platform"])


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"url": "source_url", "link": "source_url", "review": "text", "content": "text", "body": "text", "source type": "source"}
    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})
    if "text" not in df.columns:
        raise ValueError("Your CSV must contain a column named 'text' with the review or conversation text.")
    for col in ["source", "source_url", "date", "platform"]:
        if col not in df.columns:
            df[col] = ""
    if "id" not in df.columns:
        df["id"] = [f"item_{i:04d}" for i in range(1, len(df) + 1)]
    else:
        df["id"] = df["id"].fillna("").astype(str).str.strip()
        blank_ids = df["id"].eq("")
        df.loc[blank_ids, "id"] = [f"item_{i:04d}" for i in range(1, int(blank_ids.sum()) + 1)]
    for col in ["text", "source", "source_url", "date", "platform"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["text"], keep="first")
    if df["id"].duplicated().any():
        df["id"] = [f"item_{i:04d}" for i in range(1, len(df) + 1)]
    return df[["id", "text", "source", "source_url", "date", "platform"]].reset_index(drop=True)


def call_gemini(messages, api_key, response_format=None, max_tokens=2500, timeout=90):
    """Call Gemini's generateContent REST API and return response text."""
    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    conversation_messages = [m for m in messages if m.get("role") != "system"]
    contents = []
    for message in conversation_messages:
        role = "model" if message.get("role") == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": str(message.get("content", ""))}]})
    if not contents:
        raise RuntimeError("No user content was supplied to Gemini.")

    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
        },
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    if response_format:
        # Gemini's generateContent API supports JSON-mode output through responseMimeType.
        payload["generationConfig"]["responseMimeType"] = "application/json"

    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    try:
        response = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach Gemini API: {exc}") from exc

    if response.status_code >= 400:
        try:
            detail = response.json()
        except Exception:
            detail = response.text[:1000]
        hint = ""
        if response.status_code == 400:
            hint = " Check that the selected Gemini model supports your API key and request settings."
        elif response.status_code == 401 or response.status_code == 403:
            hint = " Check that Streamlit Secrets contains a valid Gemini API key named GEMINI_API_KEY."
        elif response.status_code == 429:
            hint = " Gemini rate/quota limit reached. Wait before retrying and check Google AI Studio usage limits."
        raise RuntimeError(f"Gemini API returned HTTP {response.status_code}: {detail}.{hint}")

    body = response.json()
    try:
        parts = body["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        if not text:
            raise KeyError("empty text")
        return text
    except (KeyError, IndexError, TypeError) as exc:
        block_reason = body.get("promptFeedback", {}).get("blockReason", "unknown")
        raise RuntimeError(f"Gemini response did not contain text (possible block reason: {block_reason}).") from exc


def safe_json_loads(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def classify_batch(records, api_key):
    system = """You are a careful qualitative research analyst examining public conversations about mutual-fund SIP experiences in India. Classify only what the text supports. Do not invent demographics, motives, causes, quotations, or outcomes. A mention is not proof of causation. If a field is not supported, use 'not stated' or 'insufficient_evidence'. The evidence_quote must be copied exactly from the original text, or be an empty string. Return valid JSON only, with an object containing an 'items' array. Each input id must appear exactly once."""
    schema_instructions = f"""For every input record, return these fields: id, is_sip_relevant (true/false), primary_theme (one of {THEMES}), investor_goal, trigger_or_context, observed_behavior, workaround, outcome_or_consequence, evidence_quote, confidence (high/medium/low). Use short plain-language values. Do not treat a question, hypothetical example, or general opinion as a personal experience unless text supports that. If unrelated to SIPs, use not_sip_related. If there is no identifiable problem/theme, use insufficient_evidence or other as appropriate. The evidence_quote must be an exact substring of the given text. Classify the text itself; do not give investment advice.\nINPUT RECORDS:\n{json.dumps(records, ensure_ascii=False)}"""
    content = call_gemini(
        [{"role": "system", "content": system}, {"role": "user", "content": schema_instructions}],
        api_key,
        response_format={"type": "json_object"},
        max_tokens=3000,
    )
    parsed = safe_json_loads(content)
    items = parsed.get("items", []) if isinstance(parsed, dict) else []
    if not isinstance(items, list):
        raise RuntimeError("Gemini JSON did not contain an 'items' list.")
    by_id = {}
    for item in items:
        if isinstance(item, dict) and item.get("id") is not None:
            by_id[str(item["id"])] = item
    output = []
    for record in records:
        model_item = by_id.get(str(record["id"]), {})
        theme = str(model_item.get("primary_theme", "insufficient_evidence"))
        if theme not in THEMES:
            theme = "other"
        quote = str(model_item.get("evidence_quote", "") or "")
        original_text = str(record.get("text", ""))
        output.append({
            "id": str(record["id"]),
            "is_sip_relevant": bool(model_item.get("is_sip_relevant", False)),
            "primary_theme": theme,
            "investor_goal": str(model_item.get("investor_goal", "not stated")),
            "trigger_or_context": str(model_item.get("trigger_or_context", "not stated")),
            "observed_behavior": str(model_item.get("observed_behavior", "not stated")),
            "workaround": str(model_item.get("workaround", "not stated")),
            "outcome_or_consequence": str(model_item.get("outcome_or_consequence", "not stated")),
            "evidence_quote": quote,
            "quote_verified": bool(quote and quote in original_text),
            "confidence": str(model_item.get("confidence", "low")).lower() if str(model_item.get("confidence", "low")).lower() in ["high", "medium", "low"] else "low",
            "analysis_status": "analysed" if model_item else "model_item_missing",
        })
    return output


def run_analysis(df, api_key, batch_size=5):
    records = df.to_dict(orient="records")
    results = []
    total = len(records)
    progress = st.progress(0, text="Preparing feedback…")
    status = st.empty()
    for start in range(0, total, batch_size):
        batch = records[start:start + batch_size]
        status.write(f"Analysing records {start + 1}–{start + len(batch)} of {total}…")
        results.extend(classify_batch(batch, api_key))
        progress.progress(min(1.0, (start + len(batch)) / max(total, 1)), text=f"Analysed {start + len(batch)} of {total}")
        # Keep consecutive calls spaced out to reduce the chance of hitting free-tier RPM limits.
        if start + batch_size < total:
            time.sleep(1.2)
    progress.empty()
    status.empty()
    result_df = df.merge(pd.DataFrame(results), on="id", how="left")
    return result_df


def retrieve_records(df, query, k=5):
    texts = df["text"].fillna("").astype(str).tolist()
    if not texts or not query.strip():
        return df.iloc[0:0].copy(), []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform(texts + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    except ValueError:
        scores = [0.0] * len(texts)
    ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    selected = [i for i in ranked if scores[i] > 0][:k]
    if not selected:
        # If no terms match, don't pretend retrieval found evidence.
        return df.iloc[0:0].copy(), []
    return df.iloc[selected].copy(), [float(scores[i]) for i in selected]


def answer_with_evidence(query, retrieved_df, scores, api_key):
    evidence = []
    for (_, row), score in zip(retrieved_df.iterrows(), scores):
        evidence.append({
            "id": str(row["id"]),
            "text": str(row["text"])[:1400],
            "source": str(row.get("source", "")),
            "source_url": str(row.get("source_url", "")),
            "date": str(row.get("date", "")),
            "platform": str(row.get("platform", "")),
            "retrieval_score": round(score, 4),
            "theme": str(row.get("primary_theme", "not analysed")),
        })
    system = """You are an evidence-grounded research assistant analysing public conversations about SIP investing in India. Use only the supplied retrieved records. Do not turn frequency into proof of causation or claim the retrieved records represent all investors. If the evidence does not answer the question, say so clearly. Never give investment advice. Return valid JSON only with fields answer (string), supporting_ids (array of record ids), limitations (string), and insufficient_evidence (boolean). supporting_ids must only contain ids from the provided records."""
    prompt = f"QUESTION:\n{query}\n\nRETRIEVED RECORDS:\n{json.dumps(evidence, ensure_ascii=False)}\n\nAnswer with concise research findings, distinguish observations from interpretations, and do not overstate what this small/retrieved sample shows."
    raw = call_gemini(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        api_key,
        response_format={"type": "json_object"},
        max_tokens=1800,
    )
    result = safe_json_loads(raw)
    valid_ids = {str(x) for x in retrieved_df["id"].tolist()}
    ids = [str(x) for x in result.get("supporting_ids", []) if str(x) in valid_ids]
    result["supporting_ids"] = ids
    result["answer"] = str(result.get("answer", "The model did not return an answer."))
    result["limitations"] = str(result.get("limitations", ""))
    result["insufficient_evidence"] = bool(result.get("insufficient_evidence", False))
    return result


def get_analysis_df():
    value = st.session_state.get("analysis_df")
    return value if isinstance(value, pd.DataFrame) and not value.empty else None


st.title("🔎 SIP Insight Lab")
st.caption("AI-assisted discovery for public conversations about mutual-fund SIP experiences")
st.info("Research prototype. AI-generated themes are hypotheses to inspect, not verified causes. Do not upload private financial data, personal identifiers, PAN details or portfolio screenshots.")

api_key = get_api_key()
with st.sidebar:
    st.header("Project status")
    if api_key:
        st.success("Gemini API key detected")
    else:
        st.warning("Gemini API key not configured")
    st.caption(f"Model: {MODEL}")
    st.caption("The app does not save uploaded data to a permanent database. Download your results before ending the session.")
    st.divider()
    st.markdown("**Expected CSV columns**")
    st.code("text, source, source_url, date, platform, id", language="text")
    st.caption("Only `text` is required. Source metadata is strongly recommended so findings can be traced back to evidence.")

input_tab, analysis_tab, rag_tab, eval_tab, log_tab = st.tabs([
    "1 · Import data", "2 · Analyse themes", "3 · Ask the evidence", "4 · Evaluate AI", "5 · Evidence log"
])

with input_tab:
    st.subheader("Import public feedback")
    st.write("Upload a CSV containing public reviews, forum comments or discussions. Keep one conversation per row and preserve its original URL where available.")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download CSV template", data=csv_bytes(template_data()), file_name="sip_feedback_template.csv", mime="text/csv", use_container_width=True)
    with c2:
        st.download_button("Download golden-set template", data=csv_bytes(golden_template()), file_name="golden_set_template.csv", mime="text/csv", use_container_width=True)
    uploaded = st.file_uploader("Upload feedback CSV", type=["csv"], key="source_csv")
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded, dtype=str).fillna("")
            normalized = normalize_input(raw)
            st.session_state["input_df"] = normalized
            st.session_state["input_is_demo"] = False
            st.success(f"Loaded {len(normalized)} non-empty, de-duplicated records.")
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Load synthetic demo data", use_container_width=True):
            st.session_state["input_df"] = synthetic_demo()
            st.session_state["input_is_demo"] = True
            st.session_state.pop("analysis_df", None)
            st.success("Synthetic demo loaded. These examples are fabricated for testing only and must never be reported as research findings.")
    with b2:
        if st.button("Clear current dataset", use_container_width=True):
            st.session_state.pop("input_df", None)
            st.session_state.pop("analysis_df", None)
            st.session_state.pop("input_is_demo", None)
            st.rerun()
    input_df = st.session_state.get("input_df")
    if isinstance(input_df, pd.DataFrame) and not input_df.empty:
        if st.session_state.get("input_is_demo"):
            st.warning("DEMO MODE: all rows are synthetic and not evidence about real investors.")
        st.write(f"**Current dataset:** {len(input_df)} records")
        st.dataframe(input_df.head(25), use_container_width=True, hide_index=True)
        if len(input_df) > 200:
            st.warning("This starter version is designed for small batches. For a large dataset, sample records or process in multiple runs to avoid API limits.")

with analysis_tab:
    st.subheader("Classify conversations and compare themes")
    input_df = st.session_state.get("input_df")
    if not isinstance(input_df, pd.DataFrame) or input_df.empty:
        st.info("Import a CSV or load the synthetic demo in the Import data tab first.")
    else:
        if st.session_state.get("input_is_demo"):
            st.warning("The current data is synthetic. Use it only to test the workflow, not as evidence in your project.")
        st.markdown("The model identifies a **reported theme** and extracts goals, context, behaviour, workarounds and outcomes only when the text supports them.")
        limit = st.slider("Maximum records to analyse in this run", min_value=1, max_value=min(100, len(input_df)), value=min(20, len(input_df)))
        if st.button("Analyse with Gemini", type="primary", disabled=not bool(api_key), use_container_width=True):
            try:
                working = input_df.head(limit).copy()
                with st.spinner("Analysing feedback in small batches…"):
                    st.session_state["analysis_df"] = run_analysis(working, api_key)
                    st.session_state["analysis_model"] = MODEL
                    st.session_state["analysis_timestamp"] = datetime.now(timezone.utc).isoformat()
                st.success("Analysis complete. Review the examples and evidence quotes before trusting the themes.")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
        result_df = get_analysis_df()
        if result_df is not None:
            relevant = result_df[result_df["is_sip_relevant"] == True]
            st.metric("Records analysed", len(result_df))
            st.metric("Marked SIP-relevant by AI", len(relevant))
            theme_counts = (relevant["primary_theme"].value_counts().rename_axis("reported_theme").reset_index(name="records"))
            if not theme_counts.empty:
                theme_counts["share_of_AI_relevant_records_pct"] = (theme_counts["records"] / max(1, len(relevant)) * 100).round(1)
                st.markdown("#### Theme frequency (not a measure of importance or causality)")
                st.bar_chart(theme_counts.set_index("reported_theme")["records"])
                st.dataframe(theme_counts, use_container_width=True, hide_index=True)
            st.markdown("#### Inspect individual classifications")
            st.dataframe(result_df[["id", "source", "source_url", "primary_theme", "investor_goal", "trigger_or_context", "observed_behavior", "workaround", "outcome_or_consequence", "evidence_quote", "quote_verified", "confidence", "analysis_status"]], use_container_width=True, hide_index=True)
            bad_quotes = result_df[~result_df["quote_verified"] & result_df["evidence_quote"].astype(str).str.len().gt(0)]
            if len(bad_quotes):
                st.warning(f"{len(bad_quotes)} generated evidence quote(s) were not exact substrings of the source text. Inspect them; do not use them as direct quotations.")
            st.download_button("Download analysis CSV", data=csv_bytes(result_df), file_name="sip_feedback_analysis.csv", mime="text/csv", use_container_width=True)
            if st.session_state.get("input_is_demo"):
                st.caption("Reminder: these are fabricated demo rows. Do not include their counts or interpretations in your research findings.")

with rag_tab:
    st.subheader("Ask a question about the collected evidence")
    st.write("This starter uses TF-IDF similarity for retrieval. It is a lightweight first RAG-style workflow; semantic embeddings and a vector database can be added later if needed.")
    result_df = get_analysis_df()
    available_df = result_df if result_df is not None else st.session_state.get("input_df")
    if not isinstance(available_df, pd.DataFrame) or available_df.empty:
        st.info("Import feedback first. Analysed records provide richer context than raw text.")
    else:
        query = st.text_area("Research question", placeholder="Example: What difficulties do people describe when trying to keep their SIP going?", height=90)
        top_k = st.slider("Number of records to retrieve", 3, 10, 5, key="rag_top_k")
        if st.button("Retrieve evidence and answer", type="primary", disabled=not bool(api_key), use_container_width=True):
            if not query.strip():
                st.warning("Enter a research question first.")
            else:
                retrieved_df, scores = retrieve_records(available_df, query, k=top_k)
                if retrieved_df.empty:
                    st.warning("The retrieval step did not find a clear text match. Try more specific wording or check whether the dataset contains relevant records. The app will not generate an answer without retrieved evidence.")
                else:
                    try:
                        with st.spinner("Retrieving records and asking Gemini to answer from those records…"):
                            rag_answer = answer_with_evidence(query, retrieved_df, scores, api_key)
                        st.session_state["last_rag_query"] = query
                        st.session_state["last_rag_answer"] = rag_answer
                        st.session_state["last_rag_retrieved"] = retrieved_df.copy()
                        st.session_state["last_rag_scores"] = scores
                    except Exception as exc:
                        st.error(f"Evidence question failed: {exc}")
        answer = st.session_state.get("last_rag_answer")
        if answer:
            st.markdown("#### Evidence-grounded answer")
            if answer.get("insufficient_evidence"):
                st.warning(answer.get("answer", "The evidence is insufficient."))
            else:
                st.write(answer.get("answer", "No answer returned."))
            if answer.get("limitations"):
                st.caption("Limitations: " + answer["limitations"])
            retrieved_df = st.session_state.get("last_rag_retrieved")
            if isinstance(retrieved_df, pd.DataFrame) and not retrieved_df.empty:
                ids = set(answer.get("supporting_ids", []))
                shown = retrieved_df[retrieved_df["id"].astype(str).isin(ids)]
                if shown.empty:
                    st.warning("The model did not provide a valid supporting record ID. Review retrieved records below; don't treat the answer as adequately sourced.")
                    shown = retrieved_df
                st.markdown("#### Supporting records")
                cols = [c for c in ["id", "source", "source_url", "date", "platform", "text"] if c in shown.columns]
                st.dataframe(shown[cols], use_container_width=True, hide_index=True)
                link_rows = shown[shown.get("source_url", pd.Series(index=shown.index, dtype=str)).astype(str).str.startswith(("http://", "https://"))] if "source_url" in shown.columns else shown.iloc[0:0]
                for _, row in link_rows.iterrows():
                    st.markdown(f"- [{row['id']} — {row.get('source', 'Source')}]({row['source_url']})")

with eval_tab:
    st.subheader("Evaluate the AI against a hand-labelled golden set")
    st.write("Label around 20 records yourself before reviewing the AI's labels. This reduces the risk of copying model mistakes into the reference answers.")
    st.markdown("Use exactly these theme labels in `primary_theme`:\n\n" + ", ".join(f"`{t}`" for t in THEMES))
    gold_file = st.file_uploader("Upload your manually labelled golden set CSV", type=["csv"], key="gold_csv")
    result_df = get_analysis_df()
    if gold_file is not None:
        try:
            gold = pd.read_csv(gold_file, dtype=str).fillna("")
            gold.columns = [str(c).strip().lower() for c in gold.columns]
            if not {"id", "primary_theme"}.issubset(gold.columns):
                st.error("Golden-set CSV must contain at least `id` and `primary_theme` columns.")
            elif result_df is None:
                st.info("Run an analysis first so the AI labels are available to compare.")
            else:
                compare = gold[["id", "primary_theme"]].merge(result_df[["id", "primary_theme"]].rename(columns={"primary_theme": "ai_theme"}), on="id", how="inner")
                compare["primary_theme"] = compare["primary_theme"].str.strip()
                compare["ai_theme"] = compare["ai_theme"].str.strip()
                compare["match"] = compare["primary_theme"] == compare["ai_theme"]
                if compare.empty:
                    st.warning("No IDs matched. Use the same IDs in the golden set and analysis CSV.")
                else:
                    accuracy = float(compare["match"].mean()) * 100
                    st.metric("Exact theme agreement", f"{accuracy:.1f}%", help="Percentage of matched hand-labelled items whose primary theme exactly matches the AI label.")
                    st.caption(f"Matched {len(compare)} labelled records. This small-set agreement is not proof that the model is generally accurate.")
                    st.dataframe(compare, use_container_width=True, hide_index=True)
                    disagreements = compare[~compare["match"]]
                    if not disagreements.empty:
                        st.markdown("#### Disagreements to inspect")
                        st.dataframe(disagreements, use_container_width=True, hide_index=True)
                    st.download_button("Download evaluation comparison", data=csv_bytes(compare), file_name="golden_set_evaluation.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Could not evaluate this file: {exc}")
    st.info("Next evaluation improvement: measure agreement by theme, review false positives/negatives, and record every manual correction in the AI Evidence Log.")

with log_tab:
    st.subheader("AI Evidence Log")
    st.write("Record where data came from, how it was labelled, what failed and what you corrected. Never present synthetic demo examples as public feedback.")
    input_df = st.session_state.get("input_df")
    result_df = get_analysis_df()
    c1, c2, c3 = st.columns(3)
    c1.metric("Loaded source records", len(input_df) if isinstance(input_df, pd.DataFrame) else 0)
    c2.metric("Analysed records", len(result_df) if result_df is not None else 0)
    c3.metric("Demo dataset", "Yes" if st.session_state.get("input_is_demo") else "No")
    if result_df is not None:
        st.write("**Model:**", st.session_state.get("analysis_model", MODEL))
        st.write("**Analysis timestamp (UTC):**", st.session_state.get("analysis_timestamp", "Not run yet"))
        st.write("**Quote-verification rate:**", f"{(result_df['quote_verified'].mean() * 100):.1f}%" if len(result_df) else "Not available")
        st.write("**Low-confidence labels:**", int((result_df["confidence"] == "low").sum()))
        st.write("**Missing model records:**", int((result_df["analysis_status"] != "analysed").sum()))
        st.download_button("Download current evidence log data", data=csv_bytes(result_df), file_name="ai_evidence_log_records.csv", mime="text/csv")
    st.markdown("#### Manual log entries to maintain")
    st.markdown("- Source name, collection date, number of items collected, date range and collection method.\n- Golden-set items, human label, AI label, agreement and rationale.\n- At least 2–3 rejected or corrected AI outputs, with the original output, correction and reason.\n- Known bias: public online feedback is self-selected and does not represent all investors.\n- Any source terms, privacy issues, duplicates or missing metadata.")

st.divider()
st.caption("SIP Insight Lab is a research prototype. It does not provide investment advice. Verify sources and manually review AI-generated interpretations before using them in a report.")
