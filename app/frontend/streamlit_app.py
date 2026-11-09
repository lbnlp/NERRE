"""Streamlit document upload interface for the NERRE extraction API."""
import hashlib
import json
import os

import requests
import streamlit as st

API_URL = os.environ.get("NERRE_API_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_BYTES = 5 * 1024 * 1024
SAMPLE = b"We report the properties of SrLaMgTaO6 crystal doped with trivalent erbium.\n"


def backend_status():
    try:
        response = requests.get(f"{API_URL}/api/status", timeout=3)
        response.raise_for_status()
        return response.json().get("model_ready", False)
    except (requests.RequestException, ValueError):
        return None


def send_document(name, content):
    try:
        response = requests.post(
            f"{API_URL}/api/extract",
            files={"file": (name, content)},
            timeout=600,
        )
        try:
            data = response.json()
        except ValueError:
            raise ValueError(f"The extraction service returned an invalid response ({response.status_code}).") from None
        if not response.ok:
            detail = data.get("detail") if isinstance(data, dict) else None
            raise ValueError(detail if isinstance(detail, str) else f"Extraction failed ({response.status_code}).")
        if not isinstance(data, dict) or data.get("status") not in ("complete", "partial") or not isinstance(data.get("sentences"), list):
            raise ValueError("The extraction service returned an invalid response.")
        return data
    except requests.RequestException as exc:
        raise ValueError("Could not reach the extraction service. Check that the backend is running.") from exc


def uploaded_changed():
    st.session_state.sample_active = False


st.set_page_config(page_title="NERRE · Document extraction", page_icon="✳", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif;color:#202822}
.stApp {background:#f7f8f4}
[data-testid="stSidebar"] {background:#f0f2eb;border-right:1px solid #e0e4d9}
.block-container {max-width:1280px;padding-top:3rem;padding-bottom:3rem}
h1 {font-weight:600;letter-spacing:-.05em}
h2,h3 {letter-spacing:-.02em}
.eyebrow {font:500 10px 'IBM Plex Mono',monospace;letter-spacing:.1em;color:#6d8f69;margin-bottom:16px}
.brand {font-size:26px;letter-spacing:-1px;font-weight:700;color:#28432e;padding-bottom:36px}
.brand span {color:#80a573}.nav-label {font:500 10px 'IBM Plex Mono',monospace;color:#98a596;letter-spacing:.1em}
.nav-active {background:#dfe9d8;border-radius:6px;padding:14px;color:#355f3b;font-size:13px;font-weight:600;margin-top:12px}
.intro {color:#879187;font-size:14px;margin-top:-12px;margin-bottom:28px}
.caption-label {font:500 10px 'IBM Plex Mono',monospace;color:#899b85;letter-spacing:.08em}
.note {background:#eff5eb;border:1px solid #e1eadc;border-radius:6px;padding:16px;color:#718771;font-size:12px;margin-top:24px}
[data-testid="stVerticalBlockBorderWrapper"] {background:#fff;border-radius:9px}
[data-testid="stFileUploaderDropzone"] {background:#f8fbf6;border:1px dashed #b8ccae;border-radius:6px}
.stButton button[kind="primary"] {background:#345f3d;border-color:#345f3d;color:white}
.stButton button[kind="primary"]:hover {background:#274d30;border-color:#274d30;color:white}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand">✳ nerre<span>.</span></div><div class="nav-label">WORKSPACE</div><div class="nav-active">▤ &nbsp; Document extraction &nbsp; ↗</div>', unsafe_allow_html=True)
    st.markdown("\n\n")
    st.caption("NERRE · Research workspace")

st.markdown('<div class="eyebrow">✳ &nbsp; RESEARCH TOOLS &nbsp; ─── &nbsp; 01 / EXTRACTION</div>', unsafe_allow_html=True)
st.title("Turn documents into data.")
st.markdown('<p class="intro">Extract structured information from scientific text, ready for your next discovery.</p>', unsafe_allow_html=True)

left, right = st.columns(2, gap="medium")
with left:
    with st.container(border=True):
        st.markdown('<div class="caption-label">01 / INPUT</div>', unsafe_allow_html=True)
        st.subheader("Source document")
        st.caption("Start with a document to analyze.")
        uploaded = st.file_uploader("Upload a document", type=["pdf", "txt"], on_change=uploaded_changed, help="Text-based PDF or UTF-8 text, up to 5 MB.")
        sample_clicked = st.button("Try a sample document ↗", use_container_width=True)
        if sample_clicked:
            st.session_state.sample_active = True
        use_sample = st.session_state.get("sample_active", False)
        if use_sample:
            name, content = "sample.txt", SAMPLE
            st.caption("Selected: sample.txt")
        elif uploaded is not None:
            name, content = uploaded.name, uploaded.getvalue()
        else:
            name, content = None, None
        identity = (name, hashlib.sha256(content).hexdigest()) if content is not None else None
        if st.session_state.get("document_identity") != identity:
            st.session_state.document_identity = identity
            st.session_state.pop("extraction_result", None)
            st.session_state.pop("extraction_error", None)
        st.markdown('<div class="caption-label">EXTRACTION SCHEMA</div>', unsafe_allow_html=True)
        st.markdown("**Doping information**  \nMaterials, dopants & their relationships")
        run = st.button("Extract information ↗", type="primary", disabled=content is None, use_container_width=True)
        if run:
            st.session_state.pop("extraction_result", None)
            st.session_state.pop("extraction_error", None)
            if len(content) > MAX_BYTES:
                st.session_state.extraction_error = "File exceeds the 5 MB limit."
            else:
                with st.spinner("Extracting information…"):
                    try:
                        st.session_state.extraction_result = send_document(name, content)
                    except ValueError as exc:
                        st.session_state.extraction_error = str(exc)
        if st.session_state.get("extraction_error"):
            st.error(st.session_state.extraction_error)

with right:
    with st.container(border=True):
        st.markdown('<div class="caption-label">02 / OUTPUT</div>', unsafe_allow_html=True)
        st.subheader("Structured output")
        st.caption("Your results will appear here.")
        result = st.session_state.get("extraction_result")
        if result:
            if result["status"] == "partial":
                count = sum(sentence.get("status") == "error" for sentence in result["sentences"])
                st.warning(f"{count} sentence(s) could not be extracted. See the errors in the JSON output.")
            st.json(result, expanded=True)
            st.download_button("Download JSON ↓", data=json.dumps(result, ensure_ascii=False, indent=2), file_name="extraction.json", mime="application/json", use_container_width=True)
        else:
            st.info("Ready when you are. Upload a document and run extraction to see your structured data here.")
        ready = backend_status()
        st.caption("● Model checkpoint available" if ready else "● Model not configured" if ready is False else "● Extraction service unavailable")

st.markdown('<div class="note">✳ &nbsp; <strong>Built for scientific discovery.</strong> The model extracts doping entities and relationships from each sentence. Review results before using them in your research.</div>', unsafe_allow_html=True)
