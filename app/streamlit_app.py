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

from services.ingestion.csv_loader import CSVLoader
from services.ingestion.docx_extractor import DOCXExtractor
from services.ingestion.pdf_extractor import PDFExtractor
from services.ingestion.pptx_extractor import PPTXExtractor
from services.ingestion.xlsx_loader import XLSXLoader


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


def configure_page() -> None:
    """Set page-level Streamlit configuration."""
    st.set_page_config(
        page_title="CloudInsight v2",
        page_icon="CI",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_theme(theme_mode: str) -> None:
    """Inject theme-aware styling and interaction polish."""
    is_dark = theme_mode == "Dark"
    palette = {
        "ink": "#eef4ff" if is_dark else "#16263a",
        "muted": "#a7b8cd" if is_dark else "#5f738a",
        "paper": "rgba(15, 23, 36, 0.84)" if is_dark else "rgba(255, 250, 242, 0.92)",
        "paper_soft": "rgba(21, 31, 48, 0.72)" if is_dark else "rgba(255, 255, 255, 0.68)",
        "line": "rgba(167, 184, 205, 0.16)" if is_dark else "rgba(22, 38, 58, 0.10)",
        "line_strong": "rgba(167, 184, 205, 0.28)" if is_dark else "rgba(22, 38, 58, 0.18)",
        "teal": "#57d4c6" if is_dark else "#187d76",
        "gold": "#ffc76a" if is_dark else "#ffb347",
        "coral": "#ff8f75" if is_dark else "#d86a4a",
        "sidebar_text": "#eef4fb" if is_dark else "#10253c",
        "sidebar_bg": "linear-gradient(180deg, rgba(9, 16, 28, 0.99), rgba(10, 62, 77, 0.98))"
        if is_dark
        else "linear-gradient(180deg, rgba(250, 244, 234, 0.98), rgba(244, 235, 222, 0.98))",
        "app_bg": (
            "radial-gradient(circle at top left, rgba(87, 212, 198, 0.14), transparent 24%),"
            "radial-gradient(circle at top right, rgba(255, 199, 106, 0.12), transparent 22%),"
            "linear-gradient(180deg, #0b1420 0%, #101c2b 100%)"
        )
        if is_dark
        else (
            "radial-gradient(circle at top left, rgba(255, 179, 71, 0.18), transparent 26%),"
            "radial-gradient(circle at top right, rgba(24, 125, 118, 0.16), transparent 24%),"
            "linear-gradient(180deg, #fbf6ee 0%, #f3ede3 100%)"
        ),
        "shadow": "0 20px 50px rgba(0, 0, 0, 0.28)" if is_dark else "0 18px 50px rgba(22, 38, 58, 0.10)",
        "hero_side": "rgba(9, 16, 28, 0.96)" if is_dark else "rgba(22, 38, 58, 0.96)",
        "hero_side_text": "rgba(238, 244, 255, 0.82)" if is_dark else "rgba(255, 255, 255, 0.82)",
        "tab_bg": "rgba(255, 255, 255, 0.04)" if is_dark else "rgba(22, 38, 58, 0.04)",
        "tab_active": "linear-gradient(135deg, rgba(87, 212, 198, 0.22), rgba(255, 199, 106, 0.16))"
        if is_dark
        else "linear-gradient(135deg, rgba(24, 125, 118, 0.12), rgba(255, 179, 71, 0.12))",
        "button_bg": "linear-gradient(135deg, #122033, #176d68)" if is_dark else "linear-gradient(135deg, #16263a, #187d76)",
        "button_text": "#f7fbff",
    }

    st.markdown(
        f"""
        <style>
        :root {{
            --ink: {palette["ink"]};
            --muted: {palette["muted"]};
            --paper: {palette["paper"]};
            --paper-soft: {palette["paper_soft"]};
            --line: {palette["line"]};
            --line-strong: {palette["line_strong"]};
            --teal: {palette["teal"]};
            --gold: {palette["gold"]};
            --coral: {palette["coral"]};
            --shadow: {palette["shadow"]};
            --button-bg: {palette["button_bg"]};
            --button-text: {palette["button_text"]};
            --tab-bg: {palette["tab_bg"]};
            --tab-active: {palette["tab_active"]};
        }}

        @keyframes riseIn {{
            from {{ opacity: 0; transform: translateY(18px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        @keyframes shimmerPulse {{
            0% {{ transform: translateX(-10px); opacity: 0.55; }}
            50% {{ transform: translateX(10px); opacity: 1; }}
            100% {{ transform: translateX(-10px); opacity: 0.55; }}
        }}

        .stApp {{
            background: {palette["app_bg"]};
            color: var(--ink);
            transition: background 0.35s ease, color 0.35s ease;
        }}

        .block-container {{
            padding-top: 1.55rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }}

        [data-testid="stSidebar"] {{
            background: {palette["sidebar_bg"]};
            border-right: 1px solid var(--line);
        }}

        [data-testid="stSidebar"] * {{
            color: {palette["sidebar_text"]} !important;
        }}

        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
            background: {palette["paper_soft"]};
            border: 1px dashed var(--line-strong);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}

        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {{
            transform: translateY(-1px);
            border-color: var(--teal);
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--paper);
            border: 1px solid var(--line) !important;
            border-radius: 24px;
            box-shadow: var(--shadow);
            animation: riseIn 0.45s ease;
        }}

        [data-testid="stMetric"] {{
            background: var(--paper-soft);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.8rem 0.9rem;
        }}

        [data-testid="stTabs"] [data-baseweb="tab-list"] {{
            gap: 0.4rem;
            background: transparent;
        }}

        [data-testid="stTabs"] [data-baseweb="tab"] {{
            background: var(--tab-bg);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.52rem 0.95rem;
            color: var(--ink);
            transition: transform 0.18s ease, background 0.25s ease, border-color 0.25s ease;
        }}

        [data-testid="stTabs"] [aria-selected="true"] {{
            background: var(--tab-active);
            border-color: var(--teal);
            transform: translateY(-1px);
        }}

        .stButton > button,
        [data-testid="stDownloadButton"] > button {{
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.04);
            background: var(--button-bg);
            color: var(--button-text);
            transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.12);
            font-weight: 700;
        }}

        .stButton > button:hover,
        [data-testid="stDownloadButton"] > button:hover {{
            transform: translateY(-1px);
            filter: brightness(1.04);
            box-shadow: 0 14px 28px rgba(0, 0, 0, 0.16);
        }}

        .stTextInput input,
        .stTextArea textarea,
        [data-baseweb="select"] > div {{
            background: var(--paper-soft);
            color: var(--ink);
            border-radius: 16px;
            border: 1px solid var(--line);
        }}

        .hero-shell {{
            background:
                linear-gradient(135deg, var(--paper), var(--paper-soft)),
                linear-gradient(120deg, rgba(24, 125, 118, 0.10), rgba(255, 179, 71, 0.12));
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
            border-radius: 30px;
            padding: 2rem 2rem 1.6rem 2rem;
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
            isolation: isolate;
            animation: riseIn 0.45s ease;
        }}

        .hero-shell:before {{
            content: "";
            position: absolute;
            inset: auto -60px -90px auto;
            width: 240px;
            height: 240px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255, 179, 71, 0.28), transparent 62%);
            z-index: -1;
        }}

        .hero-shell:after {{
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 180px;
            height: 3px;
            background: linear-gradient(90deg, transparent, var(--teal), transparent);
            animation: shimmerPulse 7s ease-in-out infinite;
        }}

        .eyebrow {{
            color: var(--teal);
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.18em;
            font-weight: 700;
            margin-bottom: 0.7rem;
        }}

        .hero-title {{
            color: var(--ink);
            font-size: 3.15rem;
            line-height: 1;
            font-weight: 900;
            margin: 0 0 0.6rem 0;
        }}

        .hero-copy {{
            color: var(--muted);
            font-size: 1rem;
            max-width: 44rem;
            margin: 0 0 1rem 0;
        }}

        .hero-steps {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }}

        .hero-step {{
            border-radius: 999px;
            padding: 0.48rem 0.8rem;
            background: var(--paper-soft);
            border: 1px solid var(--line);
            color: var(--ink);
            font-size: 0.9rem;
            font-weight: 600;
            transition: transform 0.18s ease, border-color 0.18s ease;
        }}

        .hero-step:hover {{
            transform: translateY(-1px);
            border-color: var(--teal);
        }}

        .hero-sidecard {{
            background: {palette["hero_side"]};
            color: white;
            border-radius: 24px;
            padding: 1.2rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            min-height: 100%;
            animation: riseIn 0.55s ease;
        }}

        .hero-sidecard h4 {{
            margin: 0 0 0.55rem 0;
            font-size: 1.02rem;
            font-weight: 800;
        }}

        .hero-sidecard p {{
            margin: 0;
            font-size: 0.93rem;
            line-height: 1.5;
            color: {palette["hero_side_text"]};
        }}

        .kpi-card {{
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 1rem 1.05rem 0.95rem 1.05rem;
            min-height: 128px;
            transition: transform 0.18s ease, border-color 0.18s ease;
            animation: riseIn 0.5s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
            border-color: var(--teal);
        }}

        .kpi-label {{
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.45rem;
            font-weight: 700;
        }}

        .kpi-value {{
            color: var(--ink);
            font-size: 2.05rem;
            line-height: 1;
            font-weight: 900;
            margin-bottom: 0.4rem;
        }}

        .kpi-note,
        .section-copy,
        .file-meta,
        .small-muted {{
            color: var(--muted);
            font-size: 0.94rem;
        }}

        .section-kicker {{
            color: var(--teal);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.14em;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }}

        .section-title {{
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 900;
            margin-bottom: 0.35rem;
        }}

        .insight-chip,
        .mini-stat,
        .prompt-chip {{
            display: inline-block;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: var(--paper-soft);
            color: var(--ink);
            font-size: 0.88rem;
            font-weight: 600;
            transition: transform 0.18s ease, border-color 0.18s ease;
        }}

        .insight-chip {{
            padding: 0.45rem 0.75rem;
            margin: 0 0.45rem 0.45rem 0;
        }}

        .mini-stat {{
            margin: 0 0.45rem 0.45rem 0;
            padding: 0.45rem 0.72rem;
        }}

        .prompt-chip {{
            margin: 0 0.5rem 0.5rem 0;
            padding: 0.5rem 0.8rem;
        }}

        .insight-chip:hover,
        .mini-stat:hover,
        .prompt-chip:hover {{
            transform: translateY(-1px);
            border-color: var(--teal);
        }}

        .file-card {{
            background: var(--paper-soft);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            transition: transform 0.18s ease, border-color 0.18s ease;
        }}

        .file-card:hover {{
            transform: translateY(-2px);
            border-color: var(--teal);
        }}

        .file-card h4 {{
            margin: 0 0 0.35rem 0;
            font-size: 1rem;
            font-weight: 800;
            color: var(--ink);
        }}

        [data-testid="stChatMessage"] {{
            background: var(--paper-soft);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.3rem 0.35rem;
        }}

        [data-testid="stExpander"] {{
            border: 1px solid var(--line);
            border-radius: 18px;
            overflow: hidden;
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
    st.session_state.setdefault("chat_question", "")
    st.session_state.setdefault(
        "editor_text",
        "Paste a draft here, then use CloudInsight actions to tighten tone, summarize, expand, or smooth grammar.",
    )


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


def tokenize(text: str) -> list[str]:
    """Convert text into a light-weight token list for retrieval and keywords."""
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return [token for token in cleaned.split() if len(token) > 2]


def top_keywords(text: str, limit: int = 8) -> list[str]:
    """Return simple high-signal keywords from extracted text."""
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "into",
        "rows",
        "columns",
        "contains",
        "sheet",
        "slide",
        "page",
        "table",
        "document",
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


def scope_results(results: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    """Filter results to the selected chat scope."""
    if scope == "All files":
        return results
    return [result for result in results if result.get("file_name") == scope]


def retrieve_relevant_sections(question: str, results: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    """Retrieve the highest-overlap sections for a question."""
    question_tokens = set(tokenize(question))
    matches: list[dict[str, Any]] = []

    for result in results:
        for section in result.get("content", {}).get("sections", []):
            text = str(section.get("text", "")).strip()
            if not text:
                continue
            section_tokens = set(tokenize(text))
            overlap = len(question_tokens & section_tokens)
            if overlap == 0:
                continue
            matches.append(
                {
                    "score": overlap,
                    "file_name": result.get("file_name", "Unnamed"),
                    "section_id": section.get("id", "section"),
                    "text": text,
                }
            )

    return sorted(matches, key=lambda item: item["score"], reverse=True)[:limit]


def run_chat_query(question: str, scoped_results: list[dict[str, Any]], scope_label: str) -> None:
    """Append a grounded chat exchange to session state."""
    answer = simple_chat_response(question, scoped_results, scope_label)
    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": question,
            "scope": scope_label,
        }
    )
    st.session_state.chat_messages.append(answer)
    st.session_state.chat_question = ""


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
    """Return a grounded local retrieval response with evidence."""
    if not results:
        return {
            "role": "assistant",
            "content": "Upload and analyze at least one file first so I have context to answer against.",
            "sources": [],
            "matches": [],
            "scope": scope_label,
        }

    question_tokens = set(tokenize(question))
    if not question_tokens:
        return {
            "role": "assistant",
            "content": "Try a more specific question, such as asking about key metrics, dominant themes, or one uploaded file by name.",
            "sources": [],
            "matches": [],
            "scope": scope_label,
        }

    matches = retrieve_relevant_sections(question, results)
    if not matches:
        summaries = " ".join(summarize_result(item) for item in results[:3])
        return {
            "role": "assistant",
            "content": f"I couldn't find a close textual match in {scope_label.lower()}, so here is the current platform context: {summaries}",
            "sources": [item.get("file_name", "Unnamed") for item in results[:3]],
            "matches": [],
            "scope": scope_label,
        }

    sources = list(dict.fromkeys(match["file_name"] for match in matches))
    lead = (
        f"Across {scope_label.lower()}, the strongest evidence points to these ideas:"
        if scope_label == "All files"
        else f"In {scope_label}, the strongest evidence points to these ideas:"
    )
    bullets = []
    for match in matches[:3]:
        snippet = match["text"].replace("\n", " ").strip()
        bullets.append(f"- {snippet[:220]}")

    return {
        "role": "assistant",
        "content": lead + "\n" + "\n".join(bullets),
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


def render_sidebar() -> list[Any]:
    """Render the control sidebar."""
    st.sidebar.markdown("## CloudInsight")
    st.sidebar.markdown("Upload files, inspect extracted structure, then move into insights, charts, chat, or editing.")
    dark_mode = st.sidebar.toggle(
        "Dark mode",
        value=st.session_state.theme_mode == "Dark",
        help="Switch between light and dark presentation modes.",
    )
    st.session_state.theme_mode = "Dark" if dark_mode else "Light"
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Input Formats")
    for suffix, label in SUPPORTED_FILE_TYPES.items():
        st.sidebar.markdown(f"- `{suffix}`  {label}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Workflow")
    st.sidebar.markdown("1. Upload one or more files")
    st.sidebar.markdown("2. Review extraction quality")
    st.sidebar.markdown("3. Explore insights and ask questions")
    st.sidebar.markdown("4. Refine narrative in the editor")
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Workspace", use_container_width=True):
        st.session_state.analyses = []
        st.session_state.chat_messages = []
        st.session_state.chat_question = ""
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
    hero_left, hero_right = st.columns((1.45, 0.8), gap="large")
    with hero_left:
        st.markdown(
            """
            <div class="hero-shell">
                <div class="eyebrow">AI Document & Data Intelligence</div>
                <div class="hero-title">CloudInsight v2</div>
                <p class="hero-copy">
                    Turn PDFs, decks, spreadsheets, and reports into a single intelligence surface.
                    This dashboard is organized around one simple flow: ingest, understand, visualize, and ask.
                </p>
                <div class="hero-steps">
                    <span class="hero-step">1. Upload mixed files</span>
                    <span class="hero-step">2. Inspect extracted structure</span>
                    <span class="hero-step">3. Generate dataset signals</span>
                    <span class="hero-step">4. Query and rewrite with AI</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_right:
        st.markdown(
            """
            <div class="hero-sidecard">
                <h4>What this page is showing</h4>
                <p>
                    Overview gives you the portfolio view. Intelligence shows file-by-file extraction.
                    Visualizations focuses on structured data. Chat is the retrieval workspace.
                    Editor is the writing surface for AI-assisted drafting.
                </p>
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
                "Portfolio View",
                "Recent files and extraction status",
                "Use this panel to confirm that each file was routed through the right ingestion path before moving deeper.",
            )
            if not results:
                st.info("Upload one or more files from the sidebar to populate the workspace.")
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
                            <div class="file-meta">{result.get('file_type', 'unknown').upper()} · {result.get('status', 'unknown').upper()}</div>
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
                "Orientation",
                "What CloudInsight is surfacing",
                "The keywords below are derived from extracted text across all successful files and give you a fast sense of the dominant topics.",
            )
            combined_text = " ".join(result.get("content", {}).get("text", "") for result in results)
            keywords = top_keywords(combined_text)
            if keywords:
                chips = "".join(f'<span class="insight-chip">{keyword}</span>' for keyword in keywords)
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.caption("Keywords appear after the first successful extraction.")

        with st.container(border=True):
            section_header(
                "Next Step",
                "How to move through the dashboard",
                "If you want to verify extraction quality, go to Intelligence. If you want data patterns, go to Visualizations. If you want Q&A, use Chat. If you want to refine language, use Editor.",
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
                "Extraction Summary",
                selected_name,
                summarize_result(result),
            )
            for chip in result_stat_chips(result):
                st.markdown(f'<span class="mini-stat">{chip}</span>', unsafe_allow_html=True)
            if result.get("warnings"):
                st.warning("; ".join(result["warnings"]))
            if result.get("errors"):
                st.error("; ".join(result["errors"]))

    with top_right:
        with st.container(border=True):
            section_header(
                "Metadata",
                "Technical extraction details",
                "This is the normalized metadata payload returned by the ingestion layer.",
            )
            st.json(result.get("metadata", {}))

    bottom_left, bottom_right = st.columns((1.2, 0.9), gap="large")
    with bottom_left:
        with st.container(border=True):
            section_header(
                "Content Preview",
                "Extracted text",
                "Review this panel to check whether the parser captured the source material correctly.",
            )
            preview_text = result.get("content", {}).get("text", "")
            st.text_area(
                "Extracted Preview",
                value=preview_text[:1800] if preview_text else "No text preview available.",
                height=360,
                label_visibility="collapsed",
                disabled=True,
                key=f"preview_{selected_name}",
            )

    with bottom_right:
        with st.container(border=True):
            section_header(
                "Structured Elements",
                "Sections and tables",
                "This view highlights the structured objects emitted by the extractor, such as pages, slides, paragraphs, and table previews.",
            )
            sections = result.get("content", {}).get("sections", [])
            st.metric("Section Count", len(sections))
            tables = result.get("content", {}).get("tables", [])
            st.metric("Table Objects", len(tables))
            if tables:
                first_table = tables[0]
                rows = first_table.get("preview_rows") or first_table.get("rows") or []
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            elif sections:
                preview_sections = pd.DataFrame(sections[:8])
                st.dataframe(preview_sections, use_container_width=True, hide_index=True)


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

    with st.container(border=True):
        section_header(
            "Retrieval Workspace",
            "Ask grounded questions about your documents",
            "Choose a scope, ask a question, and CloudInsight will answer using the most relevant extracted sections from the selected files.",
        )

        guide_left, guide_right = st.columns((1.15, 0.85), gap="large")
        with guide_left:
            st.caption(
                f"Current scope: {selected_scope}. Matching sections are ranked by token overlap against your question."
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

        for message in st.session_state.chat_messages[-10:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    st.caption("Sources: " + ", ".join(message["sources"]))
                if message.get("matches"):
                    with st.expander("Evidence used", expanded=False):
                        for match in message["matches"][:3]:
                            st.markdown(f"**{match['file_name']} · {match['section_id']}**")
                            st.write(match["text"][:280])

        ask_left, ask_right = st.columns((1, 0.22), gap="small")
        with ask_left:
            question = st.text_input(
                "Ask about the uploaded materials",
                key="chat_question",
                placeholder="What are the main themes across the uploaded files?",
            )
        with ask_right:
            submit = st.button("Ask", use_container_width=True)

        if chosen_prompt:
            run_chat_query(chosen_prompt, scoped_results, selected_scope)
            st.rerun()
        if submit and question.strip():
            run_chat_query(question.strip(), scoped_results, selected_scope)
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

        action_columns = st.columns(4)
        actions = ["Rewrite", "Summarize", "Expand", "Improve Grammar"]
        for column, action in zip(action_columns, actions, strict=False):
            if column.button(action, use_container_width=True):
                run_editor_action(action)
                st.rerun()

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

    overview_tab, intelligence_tab, visualization_tab, chat_tab, editor_tab = st.tabs(
        ["Overview", "Intelligence", "Visualizations", "Chat", "Editor"]
    )

    with overview_tab:
        render_overview_tab(results)
    with intelligence_tab:
        render_intelligence_tab(results)
    with visualization_tab:
        render_visualization_tab(results)
    with chat_tab:
        render_chat_tab(results)
    with editor_tab:
        render_editor_tab()


if __name__ == "__main__":
    main()
