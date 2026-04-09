"""PDF ingestion service for CloudInsight."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from services.ingestion.common import (
    IngestionValidationError,
    build_error_result,
    build_success_result,
    clean_text,
    count_words,
    join_text_fragments,
    validate_file,
)


logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract structured page-wise content from PDF documents."""

    def __init__(self, include_empty_pages: bool = False) -> None:
        self.include_empty_pages = include_empty_pages

    def extract(self, file_path: str | Path) -> dict[str, Any]:
        """Extract text and metadata from a PDF file."""
        try:
            path = validate_file(file_path, {".pdf"})
        except IngestionValidationError as exc:
            return build_error_result(file_path, "pdf", errors=[str(exc)])

        try:
            try:
                from pypdf import PdfReader
            except ImportError:
                from PyPDF2 import PdfReader
        except ImportError:
            return build_error_result(
                path,
                "pdf",
                errors=["Missing dependency 'pypdf' or 'PyPDF2'. Install one before ingesting PDF files."],
            )

        try:
            reader = PdfReader(str(path))
            sections: list[dict[str, Any]] = []
            warnings: list[str] = []
            empty_page_count = 0

            for page_number, page in enumerate(reader.pages, start=1):
                page_text = clean_text(page.extract_text() or "")
                if not page_text:
                    empty_page_count += 1
                    if self.include_empty_pages:
                        sections.append(
                            {
                                "id": f"page_{page_number}",
                                "type": "page",
                                "page_number": page_number,
                                "text": "",
                                "char_count": 0,
                                "word_count": 0,
                            }
                        )
                    continue

                sections.append(
                    {
                        "id": f"page_{page_number}",
                        "type": "page",
                        "page_number": page_number,
                        "text": page_text,
                        "char_count": len(page_text),
                        "word_count": count_words(page_text),
                    }
                )

            if empty_page_count:
                warnings.append(
                    f"{empty_page_count} page(s) had no extractable text. This can happen with scanned PDFs."
                )

            metadata = {
                "page_count": len(reader.pages),
                "extracted_page_count": sum(1 for section in sections if section["text"]),
                "empty_page_count": empty_page_count,
            }
            full_text = join_text_fragments(section["text"] for section in sections)

            logger.info("Extracted PDF content from %s", path)
            return build_success_result(
                path,
                "pdf",
                metadata=metadata,
                text=full_text,
                sections=sections,
                warnings=warnings,
            )
        except Exception as exc:
            logger.exception("PDF extraction failed for %s", path)
            return build_error_result(path, "pdf", errors=[f"PDF extraction failed: {exc}"])


def extract_pdf(file_path: str | Path, **_: Any) -> dict[str, Any]:
    """Convenience wrapper for PDF extraction."""
    return PDFExtractor().extract(file_path)
