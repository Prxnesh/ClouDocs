"""PDF ingestion utilities for CloudInsight v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency fallback
    try:
        from PyPDF2 import PdfReader  # type: ignore[assignment]
    except ImportError:  # pragma: no cover - handled at runtime
        PdfReader = None  # type: ignore[assignment]


class PDFExtractor:
    """Extract structured text content from PDF documents."""

    SUPPORTED_SUFFIXES = {".pdf"}

    def __init__(self, *, max_pages: int | None = None, max_chars_per_page: int | None = 20000) -> None:
        self.max_pages = max_pages
        self.max_chars_per_page = max_chars_per_page

    def extract(self, file_path: str | Path) -> dict[str, Any]:
        """Extract PDF content and return a standardized payload."""
        path = Path(file_path).expanduser()
        detected_type = self._detect_file_type(path)

        try:
            self._validate(path, detected_type)
            self._ensure_dependency()

            reader = PdfReader(str(path))
            total_pages = len(reader.pages)
            page_limit = total_pages if self.max_pages is None else min(total_pages, self.max_pages)

            sections: list[dict[str, Any]] = []
            text_parts: list[str] = []
            warnings: list[str] = []

            for page_index in range(page_limit):
                extracted = reader.pages[page_index].extract_text() or ""
                normalized_text = " ".join(extracted.split())

                if self.max_chars_per_page and len(normalized_text) > self.max_chars_per_page:
                    normalized_text = normalized_text[: self.max_chars_per_page].rstrip()
                    warnings.append(
                        f"Page {page_index + 1} exceeded {self.max_chars_per_page} characters and was truncated."
                    )

                sections.append(
                    {
                        "id": f"page_{page_index + 1}",
                        "type": "page",
                        "page_number": page_index + 1,
                        "text": normalized_text,
                        "char_count": len(normalized_text),
                    }
                )
                if normalized_text:
                    text_parts.append(normalized_text)

            if total_pages > page_limit:
                warnings.append(
                    f"Processed {page_limit} of {total_pages} pages. Increase max_pages to ingest the full document."
                )

            metadata = {
                "file_size_bytes": path.stat().st_size,
                "page_count": total_pages,
                "pages_processed": page_limit,
                "truncated": total_pages > page_limit or bool(warnings),
            }
            content = {
                "text": "\n\n".join(text_parts),
                "sections": sections,
                "tables": [],
            }
            return self._success_response(path, detected_type, metadata, content, warnings)
        except Exception as exc:  # pragma: no cover - exercised through integration
            return self._error_response(path, detected_type, exc)

    def _detect_file_type(self, path: Path) -> str:
        return path.suffix.lower().lstrip(".") or "unknown"

    def _validate(self, path: Path, detected_type: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Expected a file path, received: {path}")
        if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type for PDF extractor: {detected_type}")

    def _ensure_dependency(self) -> None:
        if PdfReader is None:
            raise ImportError("PDF extraction requires 'pypdf' or 'PyPDF2' to be installed.")

    def _success_response(
        self,
        path: Path,
        detected_type: str,
        metadata: dict[str, Any],
        content: dict[str, Any],
        warnings: list[str],
    ) -> dict[str, Any]:
        return {
            "status": "success",
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "file_type": "pdf",
            "detected_type": detected_type,
            "metadata": metadata,
            "content": content,
            "warnings": warnings,
            "errors": [],
        }

    def _error_response(self, path: Path, detected_type: str, exc: Exception) -> dict[str, Any]:
        return {
            "status": "error",
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "file_type": "pdf",
            "detected_type": detected_type,
            "metadata": {},
            "content": {"text": "", "sections": [], "tables": []},
            "warnings": [],
            "errors": [str(exc)],
        }


def extract_pdf(file_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper for PDF extraction."""
    return PDFExtractor(**kwargs).extract(file_path)
