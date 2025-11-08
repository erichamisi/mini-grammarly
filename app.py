# app.py — Your mini “Grammarly‑like” app in Python (Streamlit)
# Run locally with:  streamlit run app.py

import difflib
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st

# ---------- Optional deps ----------
try:
    import language_tool_python as lt
    from language_tool_python.utils import RateLimitError
except Exception:
    lt = None
    RateLimitError = Exception  # fallback type

try:
    import textstat
except Exception:
    textstat = None

# Do NOT import transformers at startup — we lazy-load to avoid ~800MB download
_paraphraser = None
_paraphraser_tokenizer = None
_PARA_MODEL = "Vamsi/T5_Paraphrase_Paws"


def _ensure_paraphraser() -> bool:
    """Load the paraphrase model on demand."""
    global _paraphraser, _paraphraser_tokenizer
    if _paraphraser is not None and _paraphraser_tokenizer is not None:
        return True
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        _paraphraser_tokenizer = AutoTokenizer.from_pretrained(_PARA_MODEL)
        _paraphraser = AutoModelForSeq2SeqLM.from_pretrained(_PARA_MODEL)
        return True
    except Exception:
        return False


SUPPORTED_LANGS = {
    "English (US)": "en-US",
    "English (UK)": "en-GB",
    "German": "de-DE",
    "French": "fr-FR",
    "Spanish": "es",
    "Portuguese": "pt-PT",
}


@dataclass
class Issue:
    rule: str
    message: str
    offset: int
    length: int
    context: str
    replacements: List[str]


# ---------------------- Engine selection ----------------------
st.set_page_config(page_title="Mini‑Grammarly", page_icon="📝")
st.title("📝 Mini‑Grammarly — Grammar, Style & Rewrite")

with st.sidebar:
    st.header("Settings")
    lang = st.selectbox("Language", list(SUPPORTED_LANGS.keys()), index=0)
    lang_code = SUPPORTED_LANGS[lang]

    ENGINE = st.radio(
        "Grammar engine",
        ["Public API (free, rate‑limited)", "Local server (no rate limits; requires Java)"],
        index=0,
        help=(
            "Public API is easiest but throttled. Local server uses Java to run LanguageTool on your machine."
        ),
    )

    # Optional remote LT server (set via Streamlit Cloud Secrets)
    REMOTE = st.secrets.get("LT_REMOTE_SERVER_URL", "")

    do_rewrite = st.checkbox("Enable paraphrase/rewrite (optional)", value=False)
    target_tone = st.selectbox("Target tone (rewrite)", ["Neutral", "Formal", "Friendly", "Concise"], index=0)
    beams = st.slider("Paraphrase quality (beams)", 2, 8, 5)

text = st.text_area("Paste or type your text", height=220, placeholder="Type here…")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    run = st.button("Check grammar & spelling")
with col2:
    rewrite_btn = st.button("Paraphrase / Rewrite")
with col3:
    stats_btn = st.button("Readability report")


# ---------------------- Core helpers ----------------------
def get_tool(lang_code: str):
    """Return a LanguageTool client based on sidebar choice or remote secret."""
    if lt is None:
        return None
    try:
        if REMOTE:
            # Use your own remote LanguageTool server
            return lt.LanguageTool(lang_code, remote_server=REMOTE)
        if ENGINE.startswith("Public"):
            return lt.LanguageToolPublicAPI(lang_code)
        else:
            # Local server mode (requires Java). language_tool_python will manage it.
            return lt.LanguageTool(lang_code)
    except Exception:
        return None


def check_text(text: str, lang_code: str) -> Tuple[str, List[Issue]]:
    tool = get_tool(lang_code)
    if tool is None:
        return text, []

    matches = tool.check(text)
    issues: List[Issue] = []
    for m in matches:
        repl = [r.value for r in getattr(m, "replacements", [])] or []
        issues.append(
            Issue(
                rule=getattr(m.rule, "id", "Rule"),
                message=m.message,
                offset=m.offset,
                length=m.errorLength,
                context=m.context,
                replacements=repl,
            )
        )

    # Auto-apply first suggested replacement for a quick "Fix all"
    corrected = list(text)
    for m in sorted(matches, key=lambda x: x.offset, reverse=True):
        if getattr(m, "replacements", None):
            suggestion = m.replacements[0].value
            start, end = m.offset, m.offset + m.errorLength
            corrected[start:end] = list(suggestion)
    corrected_text = "".join(corrected)

    return corrected_text, issues


@st.cache_data(show_spinner=False, ttl=300)
def check_text_cached(text: str, lang_code: str, engine_key: str, remote_key: str):
    return check_text(text, lang_code)


def highlight_diff(before: str, after: str) -> str:
    """Return HTML with <ins>/<del> markers to visualize changes."""
    diff = difflib.ndiff(before.split(), after.split())
    html_tokens = []
    for token in diff:
        if token.startswith("- "):
            html_tokens.append(f"<del>{token[2:]}</del>")
        elif token.startswith("+ "):
            html_tokens.append(f"<ins>{token[2:]}</ins>")
        elif token.startswith("? "):
            continue
        else:
            html_tokens.append(token[2:])
    return " ".join(html_tokens)


def paraphrase(text: str, beams: int = 5, max_len: int = 256, style_hint: str = "") -> str:
    if not _ensure_paraphraser():
        return "(Paraphraser unavailable — install transformers to enable this feature.)"
    prefix = "paraphrase: " + (style_hint + " " if style_hint else "") + text
    inputs = _paraphraser_tokenizer.encode(prefix, return_tensors="pt", truncation=True)
    outputs = _paraphraser.generate(
        inputs,
        max_length=max_len,
        num_beams=beams,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return _paraphraser_tokenizer.decode(outputs[0], skip_special_tokens=True)


# ---------------------- Actions ----------------------
if run and text.strip():
    try:
        fixed, issues = check_text_cached(text, lang_code, ENGINE, REMOTE)
    except RateLimitError:
        st.error(
            "You’ve hit the free LanguageTool API rate limit. "
            "Switch the engine to **Local server** in the sidebar (requires Java), "
            "configure **LT_REMOTE_SERVER_URL** in Secrets, or try again later."
        )
        st.stop()
    except Exception as e:
        import traceback
        st.error("Grammar check failed:\n\n" + traceback.format_exc())
        st.stop()

    if lt is None:
        st.warning("language-tool-python is not installed. Run: pip install language-tool-python")
    elif not issues:
        st.success("No issues found.")

    # Show diff
    st.subheader("Suggested fixes (auto‑applied preview)")
    st.markdown(highlight_diff(text, fixed), unsafe_allow_html=True)

    # Issues table
    if issues:
        rows = []
        for i, it in enumerate(issues, 1):
            bad = text[it.offset : it.offset + it.length]
            top_suggestion = it.replacements[0] if it.replacements else "(none)"
            rows.append(
                {
                    "#": i,
                    "Rule": it.rule,
                    "Message": it.message,
                    "Text": bad,
                    "Suggestion": top_suggestion,
                    "Pos": f"{it.offset}:{it.offset + it.length}",
                }
            )
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
        except Exception:
            for r in rows:
                st.write(r)

    st.download_button("Download corrected text", fixed.encode("utf-8"), file_name="corrected.txt")

if rewrite_btn and text.strip():
    hint = ""
    if target_tone == "Formal":
        hint = "Rewrite formally:"
    elif target_tone == "Friendly":
        hint = "Rewrite in a friendly tone:"
    elif target_tone == "Concise":
        hint = "Rewrite concisely:"

    out = paraphrase(text, beams=beams, style_hint=hint) if do_rewrite else "(Rewrite disabled in sidebar.)"
    st.subheader("Paraphrased / Rewritten")
    st.write(out)
    if _paraphraser is None and do_rewrite:
        st.info(
            "Install transformers to enable paraphrasing: pip install transformers sentencepiece torch --upgrade"
        )

if stats_btn and text.strip():
    st.subheader("Readability")
    if textstat is None:
        st.info("Install textstat for readability scores: pip install textstat")
    else:
        colA, colB = st.columns(2)
        with colA:
            st.metric("Flesch Reading Ease", f"{textstat.flesch_reading_ease(text):.1f}")
            st.metric("Flesch–Kincaid Grade", f"{textstat.flesch_kincaid_grade(text):.1f}")
            st.metric("Gunning Fog", f"{textstat.gunning_fog(text):.1f}")
        with colB:
            st.metric("SMOG Index", f"{textstat.smog_index(text):.1f}")
            st.metric("Coleman–Liau", f"{textstat.coleman_liau_index(text):.1f}")
            st.metric("Dale–Chall", f"{textstat.dale_chall_readability_score(text):.1f}")

st.caption(
    "Tip: If you get rate limits on the Public API, switch to **Local server** (requires Java).  "
    "To silence the Windows symlink warning from huggingface_hub: setx HF_HUB_DISABLE_SYMLINKS_WARNING 1"
)
