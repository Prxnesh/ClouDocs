"""CloudInsight v2 Streamlit dashboard."""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.audio.tts_engine import TTSEngine
from services.ingestion.csv_loader import CSVLoader
from services.ingestion.docx_extractor import DOCXExtractor
from services.ingestion.pdf_extractor import PDFExtractor
from services.ingestion.pptx_extractor import PPTXExtractor
from services.ingestion.xlsx_loader import XLSXLoader
from services.llm.chat_engine import ChatEngine
from services.sentiment_analyzer import analyze_sentiment


SUPPORTED_FILE_TYPES = {
    ".pdf": "PDF",
    ".docx": "DOCX",
    ".pptx": "PPTX",
    ".csv": "CSV",
    ".xlsx": "XLSX",
}

TEXTUAL_TYPES = {"pdf", "docx", "pptx"}
DATA_TYPES = {"csv", "xlsx"}
CHAT_PROMPTS = [
    ("Summarize", "Summarize this file in plain English."),
    ("Key Themes", "What are the key metrics or themes?"),
    ("What Matters", "What stands out as unusual or important?"),
]
RETRIEVAL_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "into",
    "about",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "there",
    "their",
    "they",
    "them",
    "then",
    "than",
    "your",
    "have",
    "has",
    "had",
    "were",
    "been",
    "does",
    "did",
    "just",
    "like",
    "show",
    "tell",
    "give",
    "file",
    "files",
    "document",
    "documents",
    "page",
    "pages",
    "table",
    "tables",
    "section",
    "sections",
}
SUMMARY_STYLE_PHRASES = (
    "summarize",
    "summary",
    "overview",
    "what is this about",
    "what's this about",
    "what happened",
    "main point",
    "key theme",
    "key themes",
    "high level",
    "tl dr",
    "tldr",
)


def configure_page() -> None:
    """Set page-level Streamlit configuration."""
    st.set_page_config(
        page_title="CloudInsight v2",
        page_icon="CI",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_theme(theme_mode: str) -> None:
    """Inject a consistent light and dark theme."""
    is_dark = theme_mode == "Dark"
    palette = {
        "ink": "#ecf2ee" if is_dark else "#20251f",
        "muted": "#9aac9f" if is_dark else "#667162",
        "app_bg": "#0b1211" if is_dark else "#f2eadf",
        "panel": "#14201f" if is_dark else "#fffaf3",
        "panel_soft": "#1b2b29" if is_dark else "#f6ede2",
        "panel_tint": "rgba(255,255,255,0.04)" if is_dark else "rgba(255,255,255,0.72)",
        "line": "#2e4743" if is_dark else "#d8cdbd",
        "line_strong": "#456660" if is_dark else "#bba992",
        "accent": "#6fd0bd" if is_dark else "#0f6c5a",
        "accent_warm": "#e9a270" if is_dark else "#ba7042",
        "accent_soft": "#183735" if is_dark else "#e0f2ea",
        "sidebar_bg": "#101918" if is_dark else "#18231f",
        "sidebar_ink": "#f3f6f3" if is_dark else "#f7f2e9",
        "tab_shell": "rgba(255,255,255,0.05)" if is_dark else "rgba(255,250,243,0.68)",
        "tab_active": "rgba(111,208,189,0.16)" if is_dark else "rgba(15,108,90,0.10)",
        "chip_bg": "rgba(255,255,255,0.08)" if is_dark else "rgba(255,255,255,0.82)",
        "chip_ink": "#dff1eb" if is_dark else "#21453a",
        "hero_side_bg": "linear-gradient(180deg, #13211f 0%, #0d1514 100%)" if is_dark else "linear-gradient(180deg, #1d2b27 0%, #13201c 100%)",
        "hero_side_ink": "rgba(243, 239, 231, 0.82)",
        "shadow": "0 18px 48px rgba(0, 0, 0, 0.34)" if is_dark else "0 18px 48px rgba(56, 41, 20, 0.10)",
        "button_bg": "linear-gradient(135deg, #6fd0bd 0%, #4d83c3 100%)" if is_dark else "linear-gradient(135deg, #0f6c5a 0%, #165d8f 100%)",
        "button_ink": "#0d1716" if is_dark else "#fbf6ef",
        "chat_bg": "#1a2b29" if is_dark else "#fff8f0",
        "audio_bg": "rgba(255,255,255,0.05)" if is_dark else "rgba(255,255,255,0.74)",
        "hero_overlay": "rgba(111,208,189,0.14)" if is_dark else "rgba(15,108,90,0.12)",
        "hero_overlay_warm": "rgba(233,162,112,0.14)" if is_dark else "rgba(186,112,66,0.12)",
    }

    st.markdown(
        f"""
        <style>
        :root {{
            --ink: {palette["ink"]};
            --muted: {palette["muted"]};
            --app-bg: {palette["app_bg"]};
            --panel: {palette["panel"]};
            --panel-soft: {palette["panel_soft"]};
            --panel-tint: {palette["panel_tint"]};
            --line: {palette["line"]};
            --line-strong: {palette["line_strong"]};
            --accent: {palette["accent"]};
            --accent-warm: {palette["accent_warm"]};
            --accent-soft: {palette["accent_soft"]};
            --sidebar-bg: {palette["sidebar_bg"]};
            --sidebar-ink: {palette["sidebar_ink"]};
            --tab-shell: {palette["tab_shell"]};
            --tab-active: {palette["tab_active"]};
            --chip-bg: {palette["chip_bg"]};
            --chip-ink: {palette["chip_ink"]};
            --hero-side-bg: {palette["hero_side_bg"]};
            --hero-side-ink: {palette["hero_side_ink"]};
            --shadow: {palette["shadow"]};
            --button-bg: {palette["button_bg"]};
            --button-ink: {palette["button_ink"]};
            --chat-bg: {palette["chat_bg"]};
            --audio-bg: {palette["audio_bg"]};
            --hero-overlay: {palette["hero_overlay"]};
            --hero-overlay-warm: {palette["hero_overlay_warm"]};
        }}

        html, body, [class*="css"] {{
            font-family: "Avenir Next", "Segoe UI", sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, var(--hero-overlay), transparent 24%),
                radial-gradient(circle at top right, var(--hero-overlay-warm), transparent 22%),
                linear-gradient(180deg, var(--app-bg) 0%, var(--panel-soft) 100%);
            color: var(--ink);
        }}

        .block-container {{
            padding-top: 1.35rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }}

        [data-testid="stSidebar"] {{
            background: var(--sidebar-bg);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--sidebar-ink) !important;
        }}

        [data-testid="stSidebar"] .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }}

        .sidebar-shell {{
            border-radius: 26px;
            padding: 1.1rem 1rem;
            margin-bottom: 1rem;
            background:
                radial-gradient(circle at top right, rgba(111, 208, 189, 0.22), transparent 32%),
                linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
            border: 1px solid rgba(255, 255, 255, 0.09);
        }}

        .sidebar-brand {{
            font-family: Georgia, "Iowan Old Style", serif;
            font-size: 1.85rem;
            line-height: 1;
            letter-spacing: -0.03em;
            margin: 0 0 0.35rem 0;
        }}

        .sidebar-note {{
            color: rgba(247, 242, 233, 0.78);
            font-size: 0.92rem;
            line-height: 1.55;
            margin: 0 0 0.85rem 0;
        }}

        .sidebar-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }}

        .sidebar-badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.6rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.10);
            font-size: 0.76rem;
            font-weight: 700;
        }}

        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
            background: rgba(255,255,255,0.04);
            border: 1px dashed rgba(255,255,255,0.24);
            border-radius: 20px;
            padding: 1rem 0.8rem;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: linear-gradient(180deg, var(--panel-tint), var(--panel));
            border: 1px solid var(--line) !important;
            border-radius: 24px;
            box-shadow: var(--shadow);
        }}

        [data-testid="stVerticalBlockBorderWrapper"] > div:first-child {{
            border-radius: 24px;
        }}

        [data-testid="stMetric"] {{
            background: var(--panel-soft);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.8rem 0.9rem;
        }}

        [data-testid="stTabs"] [data-baseweb="tab-list"] {{
            gap: 0.5rem;
            background: var(--tab-shell);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.35rem;
        }}

        [data-testid="stTabs"] [data-baseweb="tab"] {{
            background: transparent;
            border: 1px solid transparent;
            border-radius: 14px;
            padding: 0.56rem 0.96rem;
            color: var(--ink);
            font-weight: 700;
        }}

        [data-testid="stTabs"] [aria-selected="true"] {{
            background: var(--tab-active);
            border-color: var(--line-strong);
            color: var(--ink);
        }}

        .stButton > button,
        [data-testid="stDownloadButton"] > button {{
            border-radius: 16px;
            border: 1px solid rgba(15,108,90,0.12);
            background: var(--button-bg);
            color: var(--button-ink);
            box-shadow: 0 12px 24px rgba(15,108,90,0.18);
            font-weight: 800;
            min-height: 2.75rem;
        }}

        .stButton > button:hover,
        [data-testid="stDownloadButton"] > button:hover {{
            border-color: var(--line-strong);
        }}

        .stTextInput input,
        .stTextArea textarea,
        [data-baseweb="select"] > div {{
            background: var(--panel-soft);
            color: var(--ink);
            border-radius: 18px;
            border: 1px solid var(--line);
        }}

        .stTextArea textarea {{
            line-height: 1.65;
        }}

        .hero-shell {{
            background:
                radial-gradient(circle at top right, var(--hero-overlay), transparent 28%),
                linear-gradient(135deg, var(--panel) 0%, var(--panel-soft) 100%);
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
            border-radius: 30px;
            padding: 1.8rem 1.8rem 1.5rem 1.8rem;
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
        }}

        .hero-shell::after {{
            content: "";
            position: absolute;
            inset: auto -40px -48px auto;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, var(--hero-overlay-warm), transparent 62%);
            pointer-events: none;
        }}

        .eyebrow {{
            color: var(--accent-warm);
            text-transform: uppercase;
            font-size: 0.76rem;
            letter-spacing: 0.16em;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }}

        .hero-title {{
            color: var(--ink);
            font-family: Georgia, "Iowan Old Style", serif;
            font-size: 3.3rem;
            line-height: 0.96;
            letter-spacing: -0.05em;
            font-weight: 700;
            margin: 0 0 0.65rem 0;
        }}

        .hero-copy {{
            color: var(--muted);
            font-size: 1rem;
            max-width: 42rem;
            line-height: 1.7;
            margin: 0;
        }}

        .hero-steps {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 1.1rem;
        }}

        .hero-step {{
            border-radius: 999px;
            padding: 0.46rem 0.82rem;
            background: var(--chip-bg);
            border: 1px solid var(--line);
            color: var(--chip-ink);
            font-size: 0.86rem;
            font-weight: 700;
        }}

        .hero-sidecard {{
            background: var(--hero-side-bg);
            color: #f3efe7;
            border-radius: 28px;
            padding: 1.3rem 1.2rem;
            border: 1px solid rgba(255,255,255,0.08);
            min-height: 100%;
        }}

        .hero-sidecard h4 {{
            margin: 0 0 0.6rem 0;
            font-size: 1rem;
            font-weight: 800;
        }}

        .hero-sidecard p {{
            margin: 0;
            font-size: 0.94rem;
            line-height: 1.62;
            color: var(--hero-side-ink);
        }}

        .hero-side-list {{
            margin: 0.95rem 0 0 0;
            padding-left: 1rem;
            color: var(--hero-side-ink);
        }}

        .hero-side-list li {{
            margin-bottom: 0.45rem;
        }}

        .hero-side-meter {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 1rem;
        }}

        .hero-meter-card {{
            border-radius: 18px;
            padding: 0.9rem;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
        }}

        .hero-meter-label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: rgba(243,239,231,0.56);
            margin-bottom: 0.35rem;
            font-weight: 800;
        }}

        .hero-meter-value {{
            font-size: 1.55rem;
            line-height: 1;
            font-weight: 800;
        }}

        .kpi-card {{
            background: linear-gradient(180deg, var(--panel), var(--panel-soft));
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 1rem 1.1rem;
            min-height: 122px;
            position: relative;
            overflow: hidden;
        }}

        .kpi-card::before {{
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, var(--accent), var(--accent-warm));
        }}

        .kpi-label {{
            color: var(--muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.45rem;
            font-weight: 800;
        }}

        .kpi-value {{
            color: var(--ink);
            font-family: Georgia, "Iowan Old Style", serif;
            font-size: 2.2rem;
            line-height: 0.95;
            letter-spacing: -0.04em;
            font-weight: 700;
            margin-bottom: 0.36rem;
        }}

        .kpi-note,
        .section-copy,
        .file-meta,
        .small-muted {{
            color: var(--muted);
            font-size: 0.94rem;
        }}

        .section-kicker {{
            color: var(--accent-warm);
            text-transform: uppercase;
            font-size: 0.76rem;
            letter-spacing: 0.16em;
            font-weight: 800;
            margin-bottom: 0.26rem;
        }}

        .section-title {{
            color: var(--ink);
            font-family: Georgia, "Iowan Old Style", serif;
            font-size: 1.75rem;
            line-height: 1;
            letter-spacing: -0.03em;
            font-weight: 700;
            margin-bottom: 0.42rem;
        }}

        .insight-chip,
        .mini-stat,
        .prompt-chip {{
            display: inline-block;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: var(--chip-bg);
            color: var(--chip-ink);
            font-size: 0.84rem;
            font-weight: 700;
        }}

        .insight-chip {{
            padding: 0.42rem 0.74rem;
            margin: 0 0.42rem 0.42rem 0;
        }}

        .mini-stat {{
            margin: 0 0.42rem 0.42rem 0;
            padding: 0.38rem 0.7rem;
        }}

        .prompt-chip {{
            margin: 0 0.46rem 0.46rem 0;
            padding: 0.44rem 0.78rem;
        }}

        .file-card {{
            background: linear-gradient(180deg, var(--panel), var(--panel-soft));
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            margin-bottom: 0.9rem;
            position: relative;
            overflow: hidden;
        }}

        .file-card::before {{
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 6px;
            background: linear-gradient(180deg, var(--accent), var(--accent-warm));
        }}

        .file-card h4 {{
            margin: 0 0 0.3rem 0;
            font-size: 1.02rem;
            font-weight: 800;
            color: var(--ink);
        }}

        .list-tight {{
            margin: 0;
            padding-left: 1rem;
            color: var(--muted);
        }}

        .list-tight li {{
            margin-bottom: 0.3rem;
        }}

        [data-testid="stChatMessage"] {{
            background: var(--chat-bg);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 0.58rem 0.74rem;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.06);
        }}

        [data-testid="stExpander"] {{
            border: 1px solid var(--line);
            border-radius: 18px;
            overflow: hidden;
        }}

        [data-testid="stAudio"] {{
            background: var(--audio-bg);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.4rem 0.55rem;
        }}

        @media (max-width: 900px) {{
            .hero-title {{
                font-size: 2.5rem;
            }}

            .hero-shell {{
                padding: 1.45rem 1.25rem 1.2rem 1.25rem;
            }}

            .section-title {{
                font-size: 1.45rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    """Initialize dashboard session state."""
    st.session_state.setdefault("analyses", [])
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("theme_mode", "Light")
    st.session_state.setdefault("chat_scope", "All files")
    st.session_state.setdefault(
        "editor_text",
        "Paste a draft here, then use CloudInsight actions to tighten tone, summarize, expand, or smooth grammar.",
    )
    st.session_state.setdefault("tts_audio_bytes", b"")
    st.session_state.setdefault("tts_audio_format", "audio/wav")
    st.session_state.setdefault("tts_audio_label", "")
    st.session_state.setdefault("tts_audio_name", "cloudinsight-readout.wav")


def has_structured_data(results: list[dict[str, Any]]) -> bool:
    """Return whether any uploaded result is a tabular dataset."""
    return any(result.get("file_type") in DATA_TYPES for result in results)


def save_uploaded_file(uploaded_file: Any) -> Path:
    """Persist uploaded content to a temporary file path."""
    suffix = Path(uploaded_file.name).suffix.lower()
    temp_dir = Path(tempfile.mkdtemp(prefix="cloudinsight_upload_"))
    temp_path = temp_dir / f"ingest{suffix}"
    temp_path.write_bytes(uploaded_file.getbuffer())
    return temp_path


def route_file(file_path: Path) -> dict[str, Any]:
    """Dispatch a file to the correct ingestion component."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return PDFExtractor().extract(file_path)
    if suffix == ".docx":
        return DOCXExtractor().extract(file_path)
    if suffix == ".pptx":
        return PPTXExtractor().extract(file_path)
    if suffix == ".csv":
        return CSVLoader(preview_rows=100).load(file_path)
    if suffix == ".xlsx":
        return XLSXLoader(preview_rows=100).load(file_path)

    return {
        "status": "error",
        "file_name": file_path.name,
        "file_path": str(file_path.resolve()),
        "file_type": "unsupported",
        "detected_type": suffix.lstrip(".") or "unknown",
        "metadata": {},
        "content": {"text": "", "sections": [], "tables": []},
        "warnings": [],
        "errors": [f"Unsupported file type: {suffix or 'unknown'}"],
    }


def infer_record_count(result: dict[str, Any]) -> int:
    """Infer a best-effort count of extracted records."""
    metadata = result.get("metadata", {})
    for key in ("row_count", "page_count", "paragraph_count", "slide_count", "sheet_count"):
        value = metadata.get(key)
        if isinstance(value, int):
            return value
    return len(result.get("content", {}).get("sections", []))


def normalize_token(token: str) -> str:
    """Normalize a token for more forgiving lexical retrieval."""
    normalized = token.lower().strip()
    for suffix in (
        "ization",
        "ations",
        "ation",
        "ments",
        "ment",
        "ingly",
        "edly",
        "iness",
        "iness",
        "lessly",
        "lessly",
        "ships",
        "ingly",
        "ness",
        "less",
        "tion",
        "sion",
        "ings",
        "ing",
        "edly",
        "edly",
        "ers",
        "ies",
        "ied",
        "ers",
        "er",
        "ed",
        "es",
        "s",
    ):
        if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 4:
            if suffix in {"ies", "ied"}:
                return normalized[: -len(suffix)] + "y"
            return normalized[: -len(suffix)]
    return normalized


def tokenize(text: str) -> list[str]:
    """Convert text into a normalized token list for retrieval and keywords."""
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    tokens: list[str] = []
    for raw_token in cleaned.split():
        normalized = normalize_token(raw_token)
        if len(normalized) <= 2 or normalized in RETRIEVAL_STOP_WORDS:
            continue
        tokens.append(normalized)
    return tokens


def is_summary_style_question(question: str) -> bool:
    """Detect broad summary prompts that should still work without exact token overlap."""
    normalized = " ".join(question.lower().split())
    return any(phrase in normalized for phrase in SUMMARY_STYLE_PHRASES)


def top_keywords(text: str, limit: int = 8) -> list[str]:
    """Return simple high-signal keywords from extracted text."""
    stop_words = RETRIEVAL_STOP_WORDS | {
        "rows",
        "columns",
        "contains",
        "sheet",
        "slide",
        "sample",
        "text",
    }
    counts = Counter(token for token in tokenize(text) if token not in stop_words)
    return [token for token, _ in counts.most_common(limit)]


def build_overview_metrics(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build high-level metric cards for the landing section."""
    total_files = len(results)
    successful = sum(1 for item in results if item.get("status") == "success")
    total_records = sum(infer_record_count(item) for item in results if item.get("status") == "success")
    data_files = sum(1 for item in results if item.get("file_type") in DATA_TYPES)
    return [
        {
            "label": "Files Analyzed",
            "value": str(total_files),
            "note": f"{successful} ingestion flows completed cleanly",
        },
        {
            "label": "Records Surfaced",
            "value": f"{total_records:,}",
            "note": "Pages, paragraphs, rows, slides, and sheets unified",
        },
        {
            "label": "Data Assets",
            "value": str(data_files),
            "note": "Structured datasets ready for visual exploration",
        },
    ]


def section_header(kicker: str, title: str, copy: str) -> None:
    """Render a consistent section heading block."""
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-copy">{copy}</div>', unsafe_allow_html=True)


def summarize_result(result: dict[str, Any]) -> str:
    """Generate a concise story for an ingested file."""
    metadata = result.get("metadata", {})
    file_type = result.get("file_type", "unknown").upper()
    file_name = result.get("file_name", "Unnamed")

    if result.get("file_type") == "pdf":
        return f"{file_name} is a {metadata.get('page_count', 0)} page PDF with page-wise text extraction ready for downstream NLP."
    if result.get("file_type") == "docx":
        return (
            f"{file_name} contains {metadata.get('paragraph_count', 0)} paragraphs and "
            f"{metadata.get('table_count', 0)} embedded tables."
        )
    if result.get("file_type") == "pptx":
        return f"{file_name} includes {metadata.get('slide_count', 0)} slides summarized into a presentation overview."
    if result.get("file_type") == "csv":
        return (
            f"{file_name} exposes {metadata.get('row_count', 0)} rows across "
            f"{metadata.get('column_count', 0)} columns for automated analytics."
        )
    if result.get("file_type") == "xlsx":
        return f"{file_name} spans {metadata.get('sheet_count', 0)} workbook sheets with tabular previews by sheet."
    return f"{file_name} was processed as {file_type}."


def result_stat_chips(result: dict[str, Any]) -> list[str]:
    """Return small summary chips for a result."""
    metadata = result.get("metadata", {})
    chips: list[str] = []
    for label, key in (
        ("Pages", "page_count"),
        ("Paragraphs", "paragraph_count"),
        ("Tables", "table_count"),
        ("Slides", "slide_count"),
        ("Rows", "row_count"),
        ("Columns", "column_count"),
        ("Sheets", "sheet_count"),
    ):
        value = metadata.get(key)
        if isinstance(value, int):
            chips.append(f"{label}: {value}")
    return chips[:4]


def routing_label(file_type: str) -> str:
    """Return the platform pipeline that handled a file."""
    if file_type in {"pdf", "docx", "txt"}:
        return "NLP ingestion pipeline"
    if file_type == "pptx":
        return "Slide summarization pipeline"
    if file_type in {"csv", "xlsx"}:
        return "Data analysis ingestion pipeline"
    return "Unknown pipeline"


def feature_highlights(result: dict[str, Any]) -> list[str]:
    """Return visible ingestion capabilities for the selected file."""
    file_type = result.get("file_type")
    if file_type == "pdf":
        return [
            "Page-wise text extraction",
            "Empty-page warning detection",
            "Normalized text sections for RAG",
        ]
    if file_type == "docx":
        return [
            "Paragraph-level extraction",
            "Embedded table previews",
            "Table text flattened for retrieval",
        ]
    if file_type == "pptx":
        return [
            "Slide-wise extraction",
            "Slide title capture",
            "Grouped shape and table text support",
        ]
    if file_type == "csv":
        return [
            "Schema and dtype detection",
            "Missing-value profiling",
            "Numeric summary and preview rows",
        ]
    if file_type == "xlsx":
        return [
            "Workbook sheet routing",
            "Per-sheet table previews",
            "Numeric summary by sheet",
        ]
    return ["Standardized ingestion payload"]


def render_badges(items: list[str], css_class: str = "mini-stat") -> None:
    """Render a compact badge row."""
    if not items:
        return
    html = "".join(f'<span class="{css_class}">{item}</span>' for item in items)
    st.markdown(html, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_chat_engine() -> ChatEngine:
    """Return a cached chat engine for grounded responses."""
    return ChatEngine()


def recent_conversation(limit: int = 6) -> list[dict[str, str]]:
    """Return the most recent chat turns for conversational continuity."""
    turns: list[dict[str, str]] = []
    for message in st.session_state.get("chat_messages", [])[-limit:]:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and content:
            turns.append({"role": str(role), "content": str(content)})
    return turns


def chat_mode_note() -> tuple[str, str]:
    """Return the active chat mode label and supporting copy."""
    engine = get_chat_engine()
    if engine.is_remote_enabled:
        return engine.mode_label, "Full conversational mode is active with grounded retrieval."
    return engine.mode_label, "A real API-backed model is not configured yet, so answers use local grounded mode."


def audio_mime_type(path: Path) -> str:
    """Infer the correct MIME type for a generated audio asset."""
    return {
        ".aiff": "audio/aiff",
        ".aif": "audio/aiff",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
    }.get(path.suffix.lower(), "audio/wav")


def synthesize_audio_to_session(text: str, label: str) -> None:
    """Generate a TTS clip and store it in session state for playback."""
    spoken_text = text_for_audio(text)
    if not spoken_text:
        st.warning("There was no clean text available to read aloud.")
        return

    audio_path = TTSEngine.synthesize(spoken_text)
    if not audio_path:
        st.warning("Audio generation was unavailable for that text.")
        return

    path = Path(audio_path)
    if not path.exists():
        st.warning("The generated audio file could not be found.")
        return

    st.session_state.tts_audio_bytes = path.read_bytes()
    st.session_state.tts_audio_format = audio_mime_type(path)
    st.session_state.tts_audio_label = label
    st.session_state.tts_audio_name = path.name


def text_for_audio(text: str) -> str:
    """Convert markdown-ish UI text into cleaner spoken text."""
    normalized_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            line = line.lstrip(">").strip()
        if line.startswith("- "):
            line = line[2:].strip()
        line = line.replace("**", "").replace("*", "").replace("`", "")
        normalized_lines.append(line)
    return " ".join(normalized_lines)


def render_tts_panel(key_prefix: str) -> None:
    """Render the latest generated audio clip if one exists."""
    if not st.session_state.get("tts_audio_bytes"):
        return

    with st.container(border=True):
        section_header(
            "Audio",
            st.session_state.get("tts_audio_label", "Generated narration"),
            "Listen to the latest narration generated from the chatbot or editor, then download it if you want to keep a copy.",
        )
        st.audio(
            st.session_state.tts_audio_bytes,
            format=st.session_state.get("tts_audio_format", "audio/wav"),
        )
        action_left, action_right = st.columns((0.7, 0.3), gap="small")
        with action_left:
            st.download_button(
                "Download Audio",
                data=st.session_state.tts_audio_bytes,
                file_name=st.session_state.get("tts_audio_name", "cloudinsight-readout.wav"),
                mime=st.session_state.get("tts_audio_format", "audio/wav"),
                use_container_width=True,
                key=f"{key_prefix}_download_audio",
            )
        with action_right:
            if st.button("Clear Audio", use_container_width=True, key=f"{key_prefix}_clear_audio"):
                st.session_state.tts_audio_bytes = b""
                st.session_state.tts_audio_label = ""
                st.session_state.tts_audio_name = "cloudinsight-readout.wav"
                st.session_state.tts_audio_format = "audio/wav"
                st.rerun()


def scope_results(results: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    """Filter results to the selected chat scope."""
    if scope == "All files":
        return results
    return [result for result in results if result.get("file_name") == scope]


def retrieve_relevant_sections(question: str, results: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    """Retrieve relevant sections with direct matching first and representative fallbacks second."""
    question_tokens = set(tokenize(question))
    summary_style = is_summary_style_question(question)
    direct_matches: list[dict[str, Any]] = []
    fallback_matches: list[dict[str, Any]] = []

    for result in results:
        sections = list(result.get("content", {}).get("sections", []))
        if not sections:
            fallback_text = str(result.get("content", {}).get("text", "")).strip()
            if fallback_text:
                sections = [{"id": "summary", "type": "summary", "text": fallback_text}]

        file_name = result.get("file_name", "Unnamed")
        file_name_tokens = set(tokenize(file_name))

        for index, section in enumerate(sections):
            text = str(section.get("text", "")).strip()
            if not text:
                continue

            section_id = str(section.get("id", "section"))
            section_type = str(section.get("type", "section"))
            section_tokens = set(tokenize(text))
            section_prefixes = {token[:4] for token in section_tokens if len(token) >= 4}
            text_lower = text.lower()

            overlap = len(question_tokens & section_tokens)
            partial_hits = len(
                {
                    token
                    for token in question_tokens
                    if len(token) >= 4 and token[:4] in section_prefixes
                }
            )
            phrase_hits = sum(1 for token in question_tokens if token in text_lower)
            file_name_hits = len(question_tokens & file_name_tokens)

            score = (overlap * 8) + (max(0, partial_hits - overlap) * 3) + phrase_hits + (file_name_hits * 2)

            if summary_style:
                if index == 0:
                    score += 3
                if "summary" in section_id.lower() or "summary" in section_type.lower():
                    score += 6
                if section_id.lower().startswith(("page_1", "paragraph_1", "slide_1", "sheet_1")):
                    score += 2

            match_payload = {
                "score": score,
                "file_name": file_name,
                "section_id": section_id,
                "text": text,
                "match_strength": "direct" if score > 0 else "fallback",
            }

            if score > 0:
                direct_matches.append(match_payload)
                continue

            fallback_score = min(len(section_tokens), 24)
            if "summary" in section_id.lower() or "summary" in section_type.lower():
                fallback_score += 10
            if section_type in {"dataset_summary", "summary"}:
                fallback_score += 6
            if index == 0:
                fallback_score += 4
            if section_id.lower().startswith(("page_1", "paragraph_1", "slide_1", "sheet_1")):
                fallback_score += 3
            if len(text) >= 240:
                fallback_score += 2

            fallback_matches.append(
                {
                    **match_payload,
                    "score": max(1, fallback_score),
                    "match_strength": "fallback",
                }
            )

    if direct_matches:
        return sorted(direct_matches, key=lambda item: item["score"], reverse=True)[:limit]

    return sorted(fallback_matches, key=lambda item: item["score"], reverse=True)[:limit]


def run_chat_query(question: str, scoped_results: list[dict[str, Any]], scope_label: str) -> None:
    """Append a grounded chat exchange to session state."""
    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": question,
            "scope": scope_label,
        }
    )
    answer = simple_chat_response(question, scoped_results, scope_label)
    st.session_state.chat_messages.append(answer)


def fallback_chat_message(matches: list[dict[str, Any]], scope_label: str) -> str:
    """Return a simple bullet-based fallback when the chat engine cannot answer cleanly."""
    if not matches:
        return f"I couldn't find a direct text match in {scope_label.lower()}, and the local chat engine could not synthesize a stronger answer."

    lead = (
        f"Across {scope_label.lower()}, the strongest evidence points to these ideas:"
        if scope_label == "All files"
        else f"In {scope_label}, the strongest evidence points to these ideas:"
    )
    bullets = []
    for match in matches[:3]:
        snippet = match["text"].replace("\n", " ").strip()
        bullets.append(f"- {snippet[:220]}")
    return lead + "\n" + "\n".join(bullets)


def build_dataset_from_result(result: dict[str, Any]) -> pd.DataFrame:
    """Build a dataframe preview from CSV or XLSX ingestion output."""
    tables = result.get("content", {}).get("tables", [])
    if not tables:
        return pd.DataFrame()
    preview_rows = tables[0].get("preview_rows", [])
    return pd.DataFrame(preview_rows)


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric column names from a dataframe."""
    return frame.select_dtypes(include=["number"]).columns.tolist()


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    """Return non-numeric column names from a dataframe."""
    return frame.select_dtypes(exclude=["number"]).columns.tolist()


def detect_data_insights(frame: pd.DataFrame) -> list[str]:
    """Produce rule-based insight statements for a dataset preview."""
    if frame.empty:
        return ["No preview data is available yet for this dataset."]

    insights: list[str] = []
    num_cols = numeric_columns(frame)
    cat_cols = categorical_columns(frame)

    if num_cols:
        missing = int(frame[num_cols].isna().sum().sum())
        insights.append(f"Numeric surface: {len(num_cols)} measurable columns are available with {missing} missing numeric values in preview.")

        if len(num_cols) >= 2:
            corr = frame[num_cols].corr(numeric_only=True)
            best_pair: tuple[str, str] | None = None
            best_value = 0.0
            for left in corr.columns:
                for right in corr.columns:
                    if left == right:
                        continue
                    value = corr.loc[left, right]
                    if pd.notna(value) and abs(float(value)) > abs(best_value):
                        best_pair = (left, right)
                        best_value = float(value)
            if best_pair:
                direction = "positive" if best_value >= 0 else "negative"
                insights.append(
                    f"Strongest visible relationship: {best_pair[0]} vs {best_pair[1]} shows a {direction} correlation of {best_value:.2f} in the preview."
                )

        spread_col = max(num_cols, key=lambda col: float(frame[col].std(ddof=0) or 0))
        insights.append(f"Highest variation appears in '{spread_col}', making it a good candidate for trend and outlier analysis.")

    if cat_cols:
        dominant_col = cat_cols[0]
        dominant_value = frame[dominant_col].astype(str).value_counts().index[0]
        insights.append(
            f"Category snapshot: '{dominant_col}' is led by '{dominant_value}' in the current preview slice."
        )

    return insights[:4]


def simple_chat_response(question: str, results: list[dict[str, Any]], scope_label: str) -> dict[str, Any]:
    """Return a grounded chat-engine response with evidence and fallbacks."""
    if not results:
        return {
            "role": "assistant",
            "content": "Upload and analyze at least one file first so I have context to answer against.",
            "sources": [],
            "matches": [],
            "scope": scope_label,
        }

    summary_style = is_summary_style_question(question)
    question_tokens = set(tokenize(question))
    if not question_tokens and not summary_style:
        return {
            "role": "assistant",
            "content": "Try a more specific question, such as asking about key metrics, dominant themes, or one uploaded file by name.",
            "sources": [],
            "matches": [],
            "scope": scope_label,
        }

    matches = retrieve_relevant_sections(question, results)
    if not matches:
        return {
            "role": "assistant",
            "content": (
                "I couldn't find direct evidence for that in the selected files. "
                "Try different keywords, narrow the scope to one file, or ask about a phrase that appears in the document."
            ),
            "sources": [],
            "matches": [],
            "scope": scope_label,
        }

    context_matches = matches
    sources = list(dict.fromkeys(match["file_name"] for match in matches))

    try:
        response_text = get_chat_engine().generate_response(
            question,
            context_matches,
            conversation=recent_conversation(),
        )
    except Exception:
        response_text = fallback_chat_message(context_matches, scope_label)

    if matches and all(match.get("match_strength") == "fallback" for match in matches):
        response_text = (
            "I didn't find an exact phrase match, so I answered from the most representative sections in the selected files.\n\n"
            + response_text
        )

    return {
        "role": "assistant",
        "content": response_text,
        "sources": sources,
        "matches": matches,
        "scope": scope_label,
    }


def summarize_editor_text(text: str) -> str:
    """Create a short summary of the editor text."""
    sentences = [part.strip() for part in text.replace("\n", " ").split(".") if part.strip()]
    if not sentences:
        return "No substantial text detected."
    if len(sentences) == 1:
        return sentences[0]
    return ". ".join(sentences[:2]) + "."


def rewrite_editor_text(text: str) -> str:
    """Rewrite text into a sharper executive tone."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return text
    return (
        "Executive rewrite: "
        + cleaned.replace(" very ", " notably ").replace(" really ", " strategically ")
        + " Focus the narrative on outcomes, signals, and next decisions."
    )


def expand_editor_text(text: str) -> str:
    """Expand text with a more developed analytical frame."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return text
    return (
        f"{cleaned}\n\nExpanded context:\n"
        "1. What changed and why it matters.\n"
        "2. The strongest supporting evidence from the source material.\n"
        "3. Recommended next action for the team."
    )


def improve_grammar(text: str) -> str:
    """Apply lightweight grammar cleanup."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return text
    cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def run_editor_action(action: str) -> None:
    """Mutate editor text according to the selected action."""
    current_text = st.session_state.get("editor_text", "")
    if action == "Rewrite":
        st.session_state.editor_text = rewrite_editor_text(current_text)
    elif action == "Summarize":
        st.session_state.editor_text = summarize_editor_text(current_text)
    elif action == "Expand":
        st.session_state.editor_text = expand_editor_text(current_text)
    elif action == "Improve Grammar":
        st.session_state.editor_text = improve_grammar(current_text)


def render_sentiment_panel(text: str) -> None:
    """Render sentiment analysis for the current editor draft."""
    sentiment = analyze_sentiment(text)

    tone_left, tone_mid, tone_right = st.columns(3)
    tone_left.metric("Sentiment", sentiment["label"])
    tone_mid.metric("Polarity", f"{sentiment['polarity_score']:+.2f}")
    tone_right.metric("Confidence", f"{sentiment['confidence']:.0%}")

    st.caption(sentiment["guidance"])

    cue_left, cue_right = st.columns(2, gap="large")
    with cue_left:
        st.markdown("**Positive cues**")
        if sentiment["positive_hits"]:
            st.markdown(
                " ".join(
                    f'<span class="mini-stat">{word} × {count}</span>'
                    for word, count in sorted(sentiment["positive_hits"].items())
                ),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No strong positive cues detected yet.")

    with cue_right:
        st.markdown("**Negative cues**")
        if sentiment["negative_hits"]:
            st.markdown(
                " ".join(
                    f'<span class="mini-stat">{word} × {count}</span>'
                    for word, count in sorted(sentiment["negative_hits"].items())
                ),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No strong negative cues detected yet.")


def render_sidebar() -> list[Any]:
    """Render the control sidebar."""
    badges = "".join(
        f'<span class="sidebar-badge">{label}</span>' for label in ("Docs", "Tables", "Chat", "Audio")
    )
    st.sidebar.markdown(
        f"""
        <div class="sidebar-shell">
            <div class="sidebar-brand">CloudInsight</div>
            <p class="sidebar-note">
                A smaller local workspace for uploading files, checking what was extracted, and chatting against that evidence.
            </p>
            <div class="sidebar-badges">{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    dark_mode = st.sidebar.toggle(
        "Dark mode",
        value=st.session_state.theme_mode == "Dark",
        help="Switch between light and dark workspace modes.",
    )
    st.session_state.theme_mode = "Dark" if dark_mode else "Light"
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Supported Files")
    for suffix, label in SUPPORTED_FILE_TYPES.items():
        st.sidebar.markdown(f"- `{suffix}` {label}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### This V1 Does")
    st.sidebar.markdown("- Upload PDF, DOCX, PPTX, CSV, and XLSX files")
    st.sidebar.markdown("- Show extracted text, sections, tables, and warnings")
    st.sidebar.markdown("- Answer questions with local Ollama chat")
    st.sidebar.markdown("- Read assistant answers aloud")
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Workspace", use_container_width=True):
        st.session_state.analyses = []
        st.session_state.chat_messages = []
        st.session_state.tts_audio_bytes = b""
        st.session_state.tts_audio_label = ""
        st.session_state.tts_audio_name = "cloudinsight-readout.wav"
        st.session_state.tts_audio_format = "audio/wav"
        st.rerun()
    return st.sidebar.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=[item.lstrip(".") for item in SUPPORTED_FILE_TYPES],
        help="Drop reports, decks, spreadsheets, or source documents.",
    )


def handle_uploads(uploaded_files: list[Any]) -> None:
    """Process uploaded files and persist results in session state."""
    if not uploaded_files:
        return

    existing_names = {item.get("file_name") for item in st.session_state.analyses}
    new_files = [uploaded for uploaded in uploaded_files if uploaded.name not in existing_names]
    if not new_files:
        return

    progress = st.progress(0.0, text="Routing files into the CloudInsight ingestion layer...")
    total = len(new_files)
    for index, uploaded_file in enumerate(new_files, start=1):
        temp_path = save_uploaded_file(uploaded_file)
        result = route_file(temp_path)
        result["file_name"] = uploaded_file.name
        st.session_state.analyses.append(result)
        progress.progress(index / total, text=f"Processed {uploaded_file.name}")

    progress.empty()


def render_hero(results: list[dict[str, Any]]) -> None:
    """Render the dashboard hero and metric strip."""
    successful = sum(1 for item in results if item.get("status") == "success")
    data_assets = sum(1 for item in results if item.get("file_type") in DATA_TYPES)
    hero_left, hero_right = st.columns((1.45, 0.8), gap="large")
    with hero_left:
        st.markdown(
            """
            <div class="hero-shell">
                <div class="eyebrow">Document Intelligence Studio</div>
                <div class="hero-title">CloudInsight</div>
                <p class="hero-copy">
                    This rescue build focuses on one job: upload files, verify the extraction, then ask grounded questions against the evidence.
                </p>
                <div class="hero-steps">
                    <span class="hero-step">Upload mixed files</span>
                    <span class="hero-step">Inspect extraction</span>
                    <span class="hero-step">Ask grounded questions</span>
                    <span class="hero-step">Listen to answers</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_right:
        st.markdown(
            f"""
            <div class="hero-sidecard">
                <h4>Current Scope</h4>
                <p>
                    {successful} file(s) are live in the workspace. This version intentionally trims the surface area
                    so the files view and the chat view are the parts that have to work.
                </p>
                <div class="hero-side-meter">
                    <div class="hero-meter-card">
                        <div class="hero-meter-label">Files Ready</div>
                        <div class="hero-meter-value">{successful}</div>
                    </div>
                    <div class="hero-meter-card">
                        <div class="hero-meter-label">Data Assets</div>
                        <div class="hero-meter-value">{data_assets}</div>
                    </div>
                </div>
                <ul class="hero-side-list">
                    <li>Use <strong>Workspace</strong> to see what was loaded and any parser warnings.</li>
                    <li>Use <strong>Files</strong> to inspect actual extracted text and table structure.</li>
                    <li>Use <strong>Ask</strong> when you want evidence-backed answers only.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    metric_columns = st.columns(3)
    for column, metric in zip(metric_columns, build_overview_metrics(results), strict=False):
        column.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{metric['label']}</div>
                <div class="kpi-value">{metric['value']}</div>
                <div class="kpi-note">{metric['note']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_overview_tab(results: list[dict[str, Any]]) -> None:
    """Render the executive overview tab."""
    left, right = st.columns((1.35, 1), gap="large")

    with left:
        with st.container(border=True):
            section_header(
                "Workspace",
                "Files and extraction status",
                "Check which parser handled each file and whether anything important was missed.",
            )
            if not results:
                st.info("Upload one or more files from the sidebar to start the rescue build workflow.")
            else:
                for result in results:
                    chips = "".join(f'<span class="mini-stat">{chip}</span>' for chip in result_stat_chips(result))
                    warning_copy = ""
                    if result.get("warnings"):
                        warning_copy = f"<div class='file-meta'>Warnings: {'; '.join(result['warnings'])}</div>"
                    st.markdown(
                        f"""
                        <div class="file-card">
                            <h4>{result.get('file_name', 'Unnamed')}</h4>
                            <div class="file-meta">
                                {result.get('file_type', 'unknown').upper()} · {result.get('status', 'unknown').upper()} · {routing_label(result.get('file_type', 'unknown'))}
                            </div>
                            <div>{summarize_result(result)}</div>
                            <div style="margin-top:0.7rem;">{chips}</div>
                            {warning_copy}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with right:
        with st.container(border=True):
            section_header(
                "Topics",
                "Current workspace signals",
                "Keywords are generated from the extracted text already inside the workspace.",
            )
            combined_text = " ".join(result.get("content", {}).get("text", "") for result in results)
            keywords = top_keywords(combined_text)
            if keywords:
                render_badges(keywords, css_class="insight-chip")
            else:
                st.caption("Keywords appear after the first successful extraction.")

        with st.container(border=True):
            section_header(
                "What This App Does Well",
                "The parts worth trusting in this build",
                "This smaller v1 is centered on extraction visibility, grounded chat, and local audio playback.",
            )
            st.markdown(
                """
                <ul class="list-tight">
                    <li>PDFs show page-level extraction and warnings for empty pages.</li>
                    <li>DOCX and PPTX files surface paragraph and slide content for inspection.</li>
                    <li>CSV and XLSX files keep schema, preview rows, and sheet-level summaries.</li>
                    <li>The Ask tab uses local Ollama chat with grounded evidence snippets.</li>
                </ul>
                """,
                unsafe_allow_html=True,
            )


def render_intelligence_tab(results: list[dict[str, Any]]) -> None:
    """Render file-level intelligence cards."""
    if not results:
        st.info("Analyze a file to unlock document and dataset intelligence cards.")
        return

    selected_name = st.selectbox(
        "Choose a file to inspect",
        options=[result["file_name"] for result in results],
        index=0,
    )
    result = next(item for item in results if item["file_name"] == selected_name)

    top_left, top_right = st.columns((1.15, 1), gap="large")
    with top_left:
        with st.container(border=True):
            section_header(
                "File Details",
                selected_name,
                summarize_result(result),
            )
            render_badges(result_stat_chips(result))
            render_badges(feature_highlights(result), css_class="insight-chip")
            if result.get("warnings"):
                st.warning("; ".join(result["warnings"]))
            if result.get("errors"):
                st.error("; ".join(result["errors"]))

    with top_right:
        with st.container(border=True):
            section_header(
                "Routing",
                "Pipeline and extraction contract",
                "This file now exposes the normalized ingestion payload used by the downstream platform.",
            )
            st.markdown(f"**Pipeline:** {routing_label(result.get('file_type', 'unknown'))}")
            st.json(result.get("metadata", {}))

    bottom_left, bottom_right = st.columns((1.2, 0.9), gap="large")
    with bottom_left:
        with st.container(border=True):
            section_header(
                "Content Preview",
                "Extracted text",
                "Use this to validate that the parser captured the source material cleanly.",
            )
            preview_text = result.get("content", {}).get("text", "")
            st.text_area(
                "Extracted Preview",
                value=preview_text[:2400] if preview_text else "No text preview available.",
                height=360,
                label_visibility="collapsed",
                disabled=True,
                key=f"preview_{selected_name}",
            )

    with bottom_right:
        with st.container(border=True):
            section_header(
                "Structured Elements",
                "Sections and tables emitted by ingestion",
                "These are the structured objects now available to later phases such as retrieval and analytics.",
            )
            sections = result.get("content", {}).get("sections", [])
            tables = result.get("content", {}).get("tables", [])
            metric_left, metric_right = st.columns(2)
            metric_left.metric("Section Count", len(sections))
            metric_right.metric("Table Objects", len(tables))
            if tables:
                first_table = tables[0]
                rows = first_table.get("preview_rows") or first_table.get("rows") or []
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            elif sections:
                preview_sections = pd.DataFrame(sections[:8])
                st.dataframe(preview_sections, use_container_width=True, hide_index=True)

    with st.container(border=True):
        section_header(
            "Visible Features",
            "What this file now contributes to the platform",
            "This section makes the new ingestion capabilities explicit in the product surface.",
        )
        info_left, info_right = st.columns((0.9, 1.1), gap="large")
        with info_left:
            st.markdown("**Enabled outputs**")
            for item in feature_highlights(result):
                st.markdown(f"- {item}")
        with info_right:
            tables = result.get("content", {}).get("tables", [])
            sections = result.get("content", {}).get("sections", [])
            if tables:
                table = tables[0]
                st.markdown("**First table schema**")
                schema_frame = pd.DataFrame(
                    {
                        "column": table.get("column_names", []),
                        "dtype": [table.get("dtypes", {}).get(name, "") for name in table.get("column_names", [])],
                        "missing": [table.get("missing_values", {}).get(name, 0) for name in table.get("column_names", [])],
                    }
                )
                st.dataframe(schema_frame, use_container_width=True, hide_index=True)
            elif sections:
                st.markdown("**Section preview**")
                st.dataframe(pd.DataFrame(sections[:10]), use_container_width=True, hide_index=True)


def render_visualization_tab(results: list[dict[str, Any]]) -> None:
    """Render auto-generated visualizations for structured data."""
    data_results = [result for result in results if result.get("file_type") in DATA_TYPES]
    if not data_results:
        st.info("Upload a CSV or XLSX file to generate charts and dataset insights.")
        return

    selected_name = st.selectbox(
        "Dataset",
        options=[item["file_name"] for item in data_results],
        help="Choose which structured file to visualize.",
    )
    result = next(item for item in data_results if item["file_name"] == selected_name)
    frame = build_dataset_from_result(result)
    num_cols = numeric_columns(frame)
    cat_cols = categorical_columns(frame)

    with st.container(border=True):
        section_header(
            "Auto Insights",
            "What stands out in the current dataset preview",
            "These observations are generated from the preview rows currently loaded into the dashboard.",
        )
        metric_one, metric_two, metric_three = st.columns(3)
        metric_one.metric("Preview Rows", len(frame))
        metric_two.metric("Numeric Columns", len(num_cols))
        metric_three.metric("Categorical Columns", len(cat_cols))
        for insight in detect_data_insights(frame):
            st.markdown(f"- {insight}")

    if frame.empty:
        st.warning("Preview rows are empty for this dataset, so there is nothing to chart yet.")
        return

    chart_left, chart_right = st.columns(2, gap="large")

    with chart_left:
        with st.container(border=True):
            section_header(
                "Numeric Profile",
                "Measured values across the preview",
                "This chart compares numeric columns side by side so you can spot scale differences quickly.",
            )
            if num_cols:
                selected_metrics = st.multiselect(
                    "Metrics to compare",
                    options=num_cols,
                    default=num_cols[: min(3, len(num_cols))],
                    key=f"metric_pick_{selected_name}",
                )
                if selected_metrics:
                    st.bar_chart(frame[selected_metrics], use_container_width=True)
                else:
                    st.caption("Choose at least one numeric column to compare.")
            else:
                st.caption("No numeric columns detected in the preview.")

    with chart_right:
        with st.container(border=True):
            section_header(
                "Category Distribution",
                "How labels are distributed",
                "The first categorical column is summarized here to provide a quick segmentation view.",
            )
            if cat_cols:
                category_column = st.selectbox(
                    "Category field",
                    options=cat_cols,
                    key=f"category_pick_{selected_name}",
                )
                counts = (
                    frame[category_column]
                    .astype(str)
                    .value_counts()
                    .rename_axis(category_column)
                    .reset_index(name="count")
                )
                st.dataframe(counts, use_container_width=True, hide_index=True)
                st.bar_chart(counts.set_index(category_column), use_container_width=True)
            else:
                st.caption("No categorical columns detected in the preview.")

    if num_cols:
        histogram_target = st.selectbox(
            "Distribution field",
            options=num_cols,
            key=f"distribution_pick_{selected_name}",
        )
        hist_counts, _ = pd.cut(
            frame[histogram_target],
            bins=min(8, max(2, len(frame))),
            retbins=True,
            include_lowest=True,
        )
        histogram = hist_counts.value_counts(sort=False).reset_index()
        histogram.columns = ["range", "count"]
        histogram["range"] = histogram["range"].astype(str)

        with st.container(border=True):
            section_header(
                "Distribution View",
                f"Histogram for {histogram_target}",
                "This bins the selected numeric field into ranges so you can see spread and concentration.",
            )
            st.bar_chart(histogram.set_index("range"), use_container_width=True)

        if len(num_cols) >= 2:
            with st.container(border=True):
                section_header(
                    "Correlation Matrix",
                    "Relationships between numeric fields",
                    "This table shows pairwise correlations computed from the current preview slice.",
                )
                st.dataframe(frame[num_cols].corr(numeric_only=True), use_container_width=True)


def render_chat_tab(results: list[dict[str, Any]]) -> None:
    """Render the conversational query panel."""
    if not results:
        st.info("Upload and analyze files first, then come back here to ask grounded questions.")
        return

    scope_options = ["All files"] + [result["file_name"] for result in results]
    selected_scope = st.selectbox(
        "Chat scope",
        options=scope_options,
        index=scope_options.index(st.session_state.chat_scope) if st.session_state.chat_scope in scope_options else 0,
        help="Ground the conversation in all files or narrow it to one document.",
    )
    st.session_state.chat_scope = selected_scope
    scoped_results = scope_results(results, selected_scope)
    mode_label, mode_copy = chat_mode_note()

    with st.container(border=True):
        section_header(
            "Chat",
            "Talk to your uploaded files",
            "Choose a scope, ask a question in plain language, and CloudInsight will answer from retrieved evidence in the selected files.",
        )

        guide_left, guide_right = st.columns((1.2, 0.8), gap="large")
        with guide_left:
            st.markdown(
                f"""
                <div class="small-muted">
                    <strong>Mode:</strong> {mode_label}<br/>
                    <strong>Scope:</strong> {selected_scope}. {mode_copy}
                </div>
                """,
                unsafe_allow_html=True,
            )
            prompt_columns = st.columns(len(CHAT_PROMPTS))
            chosen_prompt = None
            for column, (label, prompt) in zip(prompt_columns, CHAT_PROMPTS, strict=False):
                if column.button(label, use_container_width=True, key=f"prompt_{label}_{selected_scope}"):
                    chosen_prompt = prompt

        with guide_right:
            with st.container(border=True):
                st.markdown("**Scope summary**")
                st.write(f"{len(scoped_results)} file(s) in context")
                for result in scoped_results[:3]:
                    st.markdown(f'<span class="mini-stat">{result.get("file_name", "Unnamed")}</span>', unsafe_allow_html=True)
                if st.button("Clear Chat", key="clear_chat_thread", use_container_width=True):
                    st.session_state.chat_messages = []
                    st.rerun()

        if not st.session_state.chat_messages:
            with st.chat_message("assistant"):
                st.markdown(
                    "I’m ready. Ask for a summary, comparisons, unusual findings, or details from one specific file."
                )

        render_tts_panel("chat")

        for index, message in enumerate(st.session_state.chat_messages[-10:]):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    st.caption("Sources: " + ", ".join(message["sources"]))
                if message.get("matches"):
                    with st.expander("Evidence used", expanded=False):
                        for match in message["matches"][:3]:
                            st.markdown(f"**{match['file_name']} · {match['section_id']}**")
                            st.write(match["text"][:280])
                if message["role"] == "assistant":
                    if st.button("Read Aloud", key=f"chat_tts_{index}", use_container_width=False):
                        synthesize_audio_to_session(message["content"], "Chatbot narration")
                        st.rerun()

        if chosen_prompt:
            run_chat_query(chosen_prompt, scoped_results, selected_scope)
            st.rerun()

        prompt = st.chat_input("Ask about the selected files")
        if prompt and prompt.strip():
            run_chat_query(prompt.strip(), scoped_results, selected_scope)
            st.rerun()


def render_editor_tab() -> None:
    """Render the AI editor experience."""
    with st.container(border=True):
        section_header(
            "Writing Studio",
            "Refine your draft with AI actions",
            "Paste a rough draft, then use the actions below to compress, expand, sharpen tone, or clean grammar.",
        )
        cleaned_words = [word for word in st.session_state.editor_text.replace("\n", " ").split(" ") if word]
        stat_left, stat_right, stat_third = st.columns(3)
        stat_left.metric("Words", len(cleaned_words))
        stat_right.metric("Characters", len(st.session_state.editor_text))
        stat_third.metric("Paragraphs", max(1, len([block for block in st.session_state.editor_text.split("\n\n") if block.strip()])))

        st.session_state.editor_text = st.text_area(
            "Draft",
            value=st.session_state.editor_text,
            height=360,
            label_visibility="collapsed",
        )

        with st.container(border=True):
            section_header(
                "Tone Analysis",
                "How the draft currently reads",
                "This sentiment pass is a lightweight editorial signal to help you check whether the draft feels positive, neutral, or negative.",
            )
            render_sentiment_panel(st.session_state.editor_text)

        action_columns = st.columns(4)
        actions = ["Rewrite", "Summarize", "Expand", "Improve Grammar"]
        for column, action in zip(action_columns, actions, strict=False):
            if column.button(action, use_container_width=True):
                run_editor_action(action)
                st.rerun()

        utility_left, utility_right = st.columns((0.4, 0.6), gap="small")
        with utility_left:
            if st.button("Listen Draft", use_container_width=True):
                synthesize_audio_to_session(st.session_state.editor_text, "Draft narration")
                st.rerun()
        with utility_right:
            st.caption("Generate an audio readout of the current draft and play it back below.")

        render_tts_panel("editor")

        st.download_button(
            "Download Draft",
            data=st.session_state.editor_text,
            file_name="cloudinsight_draft.txt",
            mime="text/plain",
            use_container_width=True,
        )


def main() -> None:
    """Run the Streamlit dashboard."""
    configure_page()
    initialize_state()
    uploaded_files = render_sidebar()
    apply_theme(st.session_state.theme_mode)
    handle_uploads(uploaded_files or [])

    results = st.session_state.analyses
    render_hero(results)

    workspace_tab, files_tab, chat_tab = st.tabs(["Workspace", "Files", "Ask"])

    with workspace_tab:
        render_overview_tab(results)
    with files_tab:
        render_intelligence_tab(results)
        if has_structured_data(results):
            st.divider()
            render_visualization_tab(results)
    with chat_tab:
        render_chat_tab(results)


if __name__ == "__main__":
    main()
