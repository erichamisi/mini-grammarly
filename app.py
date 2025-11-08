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
    RateLimitError = Excepti
