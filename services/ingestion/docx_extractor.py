"""DOCX ingestion utilities for CloudInsight v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from docx import Document
except ImportError:  # pragma: no cover - handled at runtime
    Document = None  # type: ignore[assignment]


class DOCXExtractor:
    """Extract paragraph and table content from DOCX files."""

    SUPPORTED_SUFFIXES = {".docx"}

    def __init__(
        self,
        *,
        max_paragraphs: int | None = None,
        max_table_rows: int = 25,
        max_chars_per_section: int | None = 10000,
    ) -> None:
        self.max_paragraphs = max_paragraphs
        self.max_table_rows = max_table_rows
        self.max_chars_per_section = max_chars_per_section

    def extract(self, file_path: str | Path) -> dict[str, Any]:
        """Extract DOCX content and return a standardized payload."""
        path = Path(file_path).expanduser()
        detected_type = self._detect_file_type(path)

        try:
            self._validate(path, detected_type)
            self._ensure_dependency()

            document = Document(str(path))
            total_paragraphs = len(document.paragraphs)
            paragraph_limit = total_paragraphs if self.max_paragraphs is None else min(total_paragraphs, self.max_paragraphs)

            sections: list[dict[str, Any]] = []
            tables: list[dict[str, Any]] = []
            text_parts: list[str] = []
            warnings: list[str] = []

            for index, paragraph in enumerate(document.paragraphs[:paragraph_limit], start=1):
                normalized_text = " ".join(paragraph.text.split())
                if not normalized_text:
                    continue

                if self.max_chars_per_section and len(normalized_text) > self.max_chars_per_section:
                    normalized_text = normalized_text[: self.max_chars_per_section].rstrip()
                    warnings.append(
                        f"Paragraph {index} exceeded {self.max_chars_per_section} characters and was truncated."
                    )

                style_name = paragraph.style.name if paragraph.style else "Normal"
                sections.append(
                    {
                        "id": f"paragraph_{index}",
                        "type": "paragraph",
                        "paragraph_number": index,
                        "style": style_name,
                        "text": normalized_text,
                        "char_count": len(normalized_text),
                    }
                )
                text_parts.append(normalized_text)

            for table_index, table in enumerate(document.tables, start=1):
                rows: list[list[str]] = []
                for row_index, row in enumerate(table.rows):
                    if row_index >= self.max_table_rows:
                        warnings.append(
                            f"Table {table_index} exceeded {self.max_table_rows} rows and was truncated."
                        )
                        break
                    rows.append([" ".join(cell.text.split()) for cell in row.cells])

                tables.append(
                    {
                        "id": f"table_{table_index}",
                        "type": "table",
                        "table_number": table_index,
                        "row_count": len(table.rows),
                        "column_count": len(table.columns),
                        "rows": rows,
                    }
                )

            if total_paragraphs > paragraph_limit:
                warnings.append(
                    f"Processed {paragraph_limit} of {total_paragraphs} paragraphs. Increase max_paragraphs to ingest more content."
                )

            metadata = {
                "file_size_bytes": path.stat().st_size,
                "paragraph_count": total_paragraphs,
                "paragraphs_processed": paragraph_limit,
                "table_count": len(document.tables),
                "truncated": total_paragraphs > paragraph_limit or bool(warnings),
            }
            content = {
                "text": "\n\n".join(text_parts),
                "sections": sections,
                "tables": tables,
            }
            return self._success_response(path, detected_type, metadata, content, warnings)
        except Exception as exc:  # pragma: no cover - exercised through integration
            return self._error_response(path, detected_type, exc)

    def _detect_file_type(self, path: Path) -> str:
        return path.suffix.lower().lstrip(".") or "unknown"

    def _validate(self, path: Path, detected_type: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"DOCX file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Expected a file path, received: {path}")
        if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type for DOCX extractor: {detected_type}")

    def _ensure_dependency(self) -> None:
        if Document is None:
            raise ImportError("DOCX extraction requires 'python-docx' to be installed.")

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
            "file_type": "docx",
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
            "file_type": "docx",
            "detected_type": detected_type,
            "metadata": {},
            "content": {"text": "", "sections": [], "tables": []},
            "warnings": [],
            "errors": [str(exc)],
        }


def extract_docx(file_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper for DOCX extraction."""
    return DOCXExtractor(**kwargs).extract(file_path)
