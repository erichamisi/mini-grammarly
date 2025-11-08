# app.py — Mini-Grammarly (Streamlit)
# Run locally:  streamlit run app.py

import difflib
import os
import sys
import warnings
import importlib.util
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional

import streamlit as st

# --------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------
st.set_page_config(page_title="Mini-Grammarly", page_icon="📝")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")  # quiet textstat log noise

# Optional libs (readability)
try:
    import textstat
except Exception:
    textstat = None

# --------------------------------------------------------------------
# Robust import of language_tool_python
#   - auto-install if missing (you can comment out _ensure_pkg(...) if you prefer)
#   - surface real import errors in the sidebar
# --------------------------------------------------------------------
def _ensure_pkg(pkg: str, version: Optional[str] = None):
    """Install a package at runtime if missing, then rerun the app."""
    if importlib.util.find_spec(pkg) is None:
        spec = f"{pkg.replace('_','-')}=={version}" if version else pkg.replace('_','-')
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", spec])
            st.toast(f"Installed {spec}. Reloading…", icon="✅")
            # st.rerun introduced in recent Streamlit; fall back to experimental if needed
            (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()
        except Exception as _e:
            st.sidebar.error(f"Auto-install failed for {spec}: {_e!r}")

# Comment out the next line if you do NOT want auto-install behavior in the cloud.
_ensure_pkg("language_tool_python", "2.7.1")

try:
    import language_tool_python as lt
    from language_tool_python.utils import RateLimitError
    lt_error = None
except Exception as e:
    lt = None
    RateLimitError = Exception  # safe placeholder
    lt_error = repr(e)

# --------------------------------------------------------------------
# Secrets helper (works with/without .streamlit/secrets.toml)
# --------------------------------------------------------------------
def _get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.environ.get(name, default)

REMOTE = _get_secret("LT_REMOTE_SERVER_URL", "")

# --------------------------------------------------------------------
# Paraphraser (lazy-loaded; no 800MB download on startup)
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# App UI
# --------------------------------------------------------------------
st.title("📝 Mini-Grammarly — Grammar, Style & Rewrite")

SUPPORTED_LANGS = {
    "English (US)": "en-US",
    "English (UK)": "en-GB",
    "German": "de-DE",
    "French": "fr-FR",
    "Spanish": "es",
    "Portuguese": "pt-PT",
}

with st.sidebar:
    st.header("Settings")
    # Diagnostics
    st.caption(
        f"LangTool pkg: {'✅' if lt else '❌'}"
        + (f" (import error: {lt_error})" if lt_error else "")
    )
    if REMOTE:
        st.caption("Remote LT server: ✅ (from secret/env)")
    else:
        st.caption("Remote LT server: ❌ (using Public/Local)")

    lang = st.selectbox("Language", list(SUPPORTED_LANGS.keys()), index=0)
    lang_code = SUPPORTED_LANGS[lang]

    ENGINE = st.radio(
        "Grammar engine",
        [
            "Public API (free, rate-limited)",
            "Local server (no rate limits; requires Java)",
            "Remote server (from secret/env)",
        ],
        index=0 if not REMOTE else 2,
        help="Public is easiest; Local needs Java; Remote uses your own LanguageTool server URL.",
    )

    do_rewrite = st.checkbox("Enable paraphrase/rewrite (optional)", value=False)
    target_tone = st.selectbox(
        "Target tone (rewrite)", ["Neutral", "Formal", "Friendly", "Concise"], index=0
    )
    beams = st.slider("Paraphrase quality (beams)", 2, 8, 5)

text = st.text_area("Paste or type your text", height=220, placeholder="Type here…")

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    run = st.button("Check grammar & spelling")
with c2:
    rewrite_btn = st.button("Paraphrase / Rewrite")
with c3:
    stats_btn = st.button("Readability report")

# --------------------------------------------------------------------
# Data classes & helpers
# --------------------------------------------------------------------
@dataclass
class Issue:
    rule: str
    message: str
    offset: int
    length: int
    context: str
    replacements: List[str]

def get_tool(lang_code: str):
    """Pick LT client based on engine choice or remote."""
    if lt is None:
        return None
    try:
        if ENGINE.startswith("Remote"):
            if not REMOTE:
                return None
            return lt.LanguageTool(lang_code, remote_server=REMOTE)
        if ENGINE.startswith("Public"):
            return lt.LanguageToolPublicAPI(lang_code)
        # Local server (requires Java); language_tool_python manages it
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

    # Auto-apply first suggestion for a quick preview
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

# --------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------
if run and text.strip():
    try:
        fixed, issues = check_text_cached(text, lang_code, ENGINE, REMOTE)
    except RateLimitError:
        st.error(
            "You’ve hit the free LanguageTool API rate limit.\n\n"
            "• Switch engine to **Local server** (requires Java) on your PC\n"
            "• Or set **LT_REMOTE_SERVER_URL** (Remote server)\n"
            "• Or try again later"
        )
        st.stop()
    except Exception as e:
        import traceback
        st.error("Grammar check failed:\n\n" + traceback.format_exc())
        st.stop()

    if lt is None:
        st.warning("language-tool-python is not available.")
    elif not issues:
        st.success("No issues found.")

    st.subheader("Suggested fixes (auto-applied preview)")
    st.markdown(highlight_diff(text, fixed), unsafe_allow_html=True)

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
        st.info("Install transformers to enable paraphrasing: pip install transformers sentencepiece torch --upgrade")

if stats_btn and text.strip():
    st.subheader("Readability")
    if textstat is None:
        st.info("Install textstat for readability scores: pip install textstat")
    else:
        cA, cB = st.columns(2)
        with cA:
            st.metric("Flesch Reading Ease", f"{textstat.flesch_reading_ease(text):.1f}")
            st.metric("Flesch–Kincaid Grade", f"{textstat.flesch_kincaid_grade(text):.1f}")
            st.metric("Gunning Fog", f"{textstat.gunning_fog(text):.1f}")
        with cB:
            st.metric("SMOG Index", f"{textstat.smog_index(text):.1f}")
            st.metric("Coleman–Liau", f"{textstat.coleman_liau_index(text):.1f}")
            st.metric("Dale–Chall", f"{textstat.dale_chall_readability_score(text):.1f}")

st.caption(
    "Engine tips: Public API is easiest but rate-limited. Local server needs Java. "
    "Set LT_REMOTE_SERVER_URL to use your own remote LanguageTool server.\n"
    "To silence the Windows symlink warning from huggingface_hub: setx HF_HUB_DISABLE_SYMLINKS_WARNING 1"
)
