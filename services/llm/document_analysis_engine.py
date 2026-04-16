"""Document analysis engine with Ollama and local fallback support."""

from __future__ import annotations

import json
from typing import Any
from urllib import request

from services.document_mode_detector import DocumentModeDetector
from services.llm.chat_engine import ChatEngine


class DocumentAnalysisEngine(ChatEngine):
    """Generate structured document analysis from ingested CloudInsight results."""

    def __init__(self) -> None:
        super().__init__()
        self.mode_detector = DocumentModeDetector()

    @property
    def ollama_label(self) -> str:
        """Return the configured Ollama backend label."""
        return f"Ollama · {self.ollama_model}"

    def analyze_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Analyze a single ingested result using Ollama or the local fallback."""
        if result.get("status") != "success":
            return {
                "status": "skipped",
                "backend": self.mode_label,
                "summary": "Analysis is only available for successfully ingested files.",
                "insights": [],
                "recommended_questions": [],
                "evidence": [],
                "warning": "; ".join(result.get("errors", []) or []) or "File ingestion did not complete cleanly.",
                "mode_detection": {},
            }

        text = self._result_text(result)
        if not text:
            return {
                "status": "skipped",
                "backend": self.mode_label,
                "summary": "No extracted text was available for analysis.",
                "insights": [],
                "recommended_questions": [],
                "evidence": [],
                "warning": "The ingestion pipeline did not produce textual evidence for this file.",
                "mode_detection": {},
            }

        detection = self.mode_detector.detect(
            text,
            file_name=result.get("file_name"),
            metadata=result.get("metadata", {}),
        )
        evidence = self._build_evidence(result)
        if self.is_ollama_available():
            try:
                payload = self._analyze_with_ollama(result, detection, evidence)
                return self._finalize_analysis(payload, detection, evidence, backend_label=self.ollama_label)
            except Exception as exc:
                ollama_warning = f"Ollama analysis failed. Details: {exc}"
        else:
            ollama_warning = ""

        local_payload = self._build_local_analysis(result, detection, evidence)
        final_warning = (
            ollama_warning + " CloudInsight used the local fallback instead."
            if ollama_warning
            else ""
        )
        return self._finalize_analysis(
            local_payload,
            detection,
            evidence,
            backend_label="Local grounded mode",
            warning=final_warning,
        )

    def _analyze_with_ollama(
        self,
        result: dict[str, Any],
        detection: dict[str, Any],
        evidence: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Run structured document analysis through the Ollama chat API."""
        system_prompt = (
            "You are a precise document analyst inside CloudInsight. "
            "Use only the supplied evidence. Return valid JSON only."
        )
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._analysis_prompt(result, detection, evidence)},
            ],
            "stream": False,
        }
        req = request.Request(
            f"{self.ollama_host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = str(data.get("message", {}).get("content", "")).strip()
        if not content:
            raise ValueError("No content returned from Ollama analysis.")
        return self._parse_analysis_json(content)

    def _analysis_prompt(
        self,
        result: dict[str, Any],
        detection: dict[str, Any],
        evidence: list[dict[str, str]],
    ) -> str:
        """Build the grounded analysis prompt for Ollama."""
        metadata = result.get("metadata", {})
        evidence_text = "\n\n".join(
            (
                f"Section: {item.get('section_id', 'section')}\n"
                f"Evidence: {item.get('text', '')}"
            )
            for item in evidence
        )
        return (
            "Analyze the uploaded file using only the supplied metadata and evidence.\n"
            "Return strict JSON with this exact shape:\n"
            '{'
            '"summary": "2-3 sentence summary", '
            '"insights": ["exactly 3 concise insights"], '
            '"recommended_questions": ["exactly 3 useful follow-up questions"]'
            '}\n'
            "Rules:\n"
            "- Do not invent facts.\n"
            "- Mention uncertainty inside the summary or insights when evidence is incomplete.\n"
            "- Keep each insight to one sentence.\n"
            "- Keep each recommended question specific to this file.\n\n"
            f"File name: {result.get('file_name', 'Unnamed')}\n"
            f"File type: {result.get('file_type', 'unknown')}\n"
            f"Detected mode: {detection.get('display_name', 'Unknown')}\n"
            f"Mode confidence: {detection.get('confidence', 0)}\n"
            f"Matched signals: {', '.join(detection.get('matched_signals', [])) or 'None'}\n"
            f"Recommended analyses: {', '.join(detection.get('recommended_analyses', [])) or 'None'}\n"
            f"Metadata: {json.dumps(metadata, default=str)}\n\n"
            f"Evidence:\n{evidence_text}"
        )

    def _parse_analysis_json(self, raw_text: str) -> dict[str, Any]:
        """Parse a JSON response from a remote backend, tolerating code fences."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("The model did not return a JSON object.")
        payload = json.loads(cleaned[start : end + 1])
        return {
            "summary": self._clean_text(payload.get("summary", "")),
            "insights": self._normalize_string_list(payload.get("insights", []), limit=3),
            "recommended_questions": self._normalize_string_list(payload.get("recommended_questions", []), limit=3),
        }

    def _build_local_analysis(
        self,
        result: dict[str, Any],
        detection: dict[str, Any],
        evidence: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Create a deterministic local analysis when no LLM backend is available."""
        metadata = result.get("metadata", {})
        combined = " ".join(item.get("text", "") for item in evidence[:3])
        summary = self.summarizer.summarize(combined)
        if not summary:
            summary = self._clean_text(combined)[:420].rstrip()
        if summary and not summary.endswith("."):
            summary += "."

        structure_note = self._structure_note(result)
        mode_name = detection.get("display_name", "General Document")
        confidence = detection.get("confidence", 0)
        insights = [
            f"CloudInsight classifies this as {mode_name.lower()} content with {confidence:.0%} confidence.",
            structure_note,
        ]

        extracted_fields = detection.get("extracted_fields", {})
        field_note = self._field_note(extracted_fields)
        if field_note:
            insights.append(field_note)
        else:
            insights.append("The strongest evidence comes from the leading extracted sections surfaced in the file preview.")

        summary_prefix = "This local analysis uses extracted evidence only."
        return {
            "summary": f"{summary_prefix} {summary}".strip(),
            "insights": insights[:3],
            "recommended_questions": self._default_questions(result, detection),
        }

    def _finalize_analysis(
        self,
        payload: dict[str, Any],
        detection: dict[str, Any],
        evidence: list[dict[str, str]],
        *,
        backend_label: str,
        warning: str | None = None,
    ) -> dict[str, Any]:
        """Normalize an analysis payload for UI rendering."""
        return {
            "status": "ready",
            "backend": backend_label,
            "summary": self._clean_text(payload.get("summary", "")),
            "insights": self._normalize_string_list(payload.get("insights", []), limit=3),
            "recommended_questions": self._normalize_string_list(payload.get("recommended_questions", []), limit=3),
            "evidence": evidence,
            "warning": warning or "",
            "mode_detection": detection,
        }

    def _build_evidence(self, result: dict[str, Any], limit: int = 4) -> list[dict[str, str]]:
        """Extract concise evidence snippets from the normalized ingestion result."""
        sections = list(result.get("content", {}).get("sections", []))
        evidence: list[dict[str, str]] = []

        for section in sections[:limit]:
            text = self._clean_text(section.get("text", ""))
            if not text:
                continue
            evidence.append(
                {
                    "section_id": str(section.get("id", "section")),
                    "text": text[:700],
                }
            )

        if evidence:
            return evidence

        fallback_text = self._clean_text(result.get("content", {}).get("text", ""))
        if fallback_text:
            evidence.append({"section_id": "summary", "text": fallback_text[:1200]})
        return evidence

    def _result_text(self, result: dict[str, Any]) -> str:
        """Return the best available text body from an ingested result."""
        content = result.get("content", {})
        text = self._clean_text(content.get("text", ""))
        if text:
            return text

        tables = content.get("tables", [])
        table_fragments: list[str] = []
        for table in tables[:2]:
            if table.get("name"):
                table_fragments.append(str(table["name"]))
            if table.get("column_names"):
                table_fragments.append(", ".join(str(item) for item in table["column_names"][:12]))
        return self._clean_text(" ".join(table_fragments))

    def _normalize_string_list(self, values: Any, *, limit: int) -> list[str]:
        """Normalize a list-like payload into a clean string list."""
        if isinstance(values, str):
            items = [item.strip(" -") for item in values.split("\n") if item.strip()]
        else:
            items = [self._clean_text(item) for item in values if self._clean_text(item)]
        return items[:limit]

    def _structure_note(self, result: dict[str, Any]) -> str:
        """Generate a concise structural insight from file metadata."""
        metadata = result.get("metadata", {})
        file_type = result.get("file_type")
        if file_type == "pdf":
            return f"The document spans {metadata.get('page_count', 0)} pages with page-level extraction available for review."
        if file_type == "docx":
            return (
                f"The file includes {metadata.get('paragraph_count', 0)} paragraphs and "
                f"{metadata.get('table_count', 0)} embedded tables."
            )
        if file_type == "pptx":
            return f"The deck contains {metadata.get('slide_count', 0)} slides and can be inspected slide by slide."
        if file_type == "csv":
            return (
                f"The dataset contains {metadata.get('row_count', 0)} rows and "
                f"{metadata.get('column_count', 0)} columns for numeric or categorical analysis."
            )
        if file_type == "xlsx":
            return (
                f"The workbook contains {metadata.get('sheet_count', 0)} sheets with "
                f"{metadata.get('table_object_count', 0)} structured table objects."
            )
        return "The file has been normalized into sections and metadata for downstream analysis."

    def _field_note(self, extracted_fields: dict[str, Any]) -> str:
        """Summarize detected structured fields when available."""
        if not extracted_fields:
            return ""

        fragments: list[str] = []
        for key, value in extracted_fields.items():
            if isinstance(value, list) and value:
                fragments.append(f"{key.replace('_', ' ')} include {', '.join(str(item) for item in value[:3])}")
            elif isinstance(value, str) and value.strip():
                cleaned = self._clean_text(value)
                fragments.append(f"{key.replace('_', ' ')} points to {cleaned[:120]}")

        if not fragments:
            return ""
        return fragments[0] + "."

    def _default_questions(self, result: dict[str, Any], detection: dict[str, Any]) -> list[str]:
        """Return follow-up questions tailored to the file type and detected mode."""
        file_name = result.get("file_name", "this file")
        mode = detection.get("mode", "general_document")

        if mode == "resume":
            return [
                f"What are the candidate's strongest skills in {file_name}?",
                f"Summarize the work experience progression in {file_name}.",
                f"What gaps or missing details should I verify in {file_name}?",
            ]
        if mode == "invoice":
            return [
                f"What payment terms and totals appear in {file_name}?",
                f"Are there any unusual charges or missing fields in {file_name}?",
                f"Pull out the vendor, amount, and date from {file_name}.",
            ]
        if mode == "research_paper":
            return [
                f"What methodology and findings are emphasized in {file_name}?",
                f"What limitations or unanswered questions appear in {file_name}?",
                f"Summarize the abstract and conclusion from {file_name}.",
            ]
        if result.get("file_type") in {"csv", "xlsx"}:
            return [
                f"What trends and outliers stand out in {file_name}?",
                f"Which columns in {file_name} are most useful for analysis?",
                f"What follow-up chart or aggregation should I run on {file_name}?",
            ]
        return [
            f"Give me the key takeaways from {file_name}.",
            f"What feels most important or unusual in {file_name}?",
            f"What should I inspect next in {file_name}?",
        ]
