"""Grounded chat engine with Ollama-first local LLM support."""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib import request

from services.nlp.summarizer import ExtractiveSummarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatEngine:
    """Answer questions from retrieved context using Ollama or a local fallback."""

    def __init__(self) -> None:
        self.summarizer = ExtractiveSummarizer(num_sentences=2)
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2").strip()
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", os.getenv("OPENAI_TIMEOUT_SECONDS", "45")))

    @property
    def mode_label(self) -> str:
        """Return the currently active chat backend label."""
        if self.is_ollama_available():
            return f"Ollama · {self.ollama_model}"
        return "Local grounded mode"

    @property
    def is_remote_enabled(self) -> bool:
        """Return whether the Ollama backend is reachable."""
        return self.is_ollama_available()

    def generate_response(
        self,
        query: str,
        retrieved_context: list[dict[str, Any]],
        conversation: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate a grounded answer from retrieved context and recent chat history."""
        logger.info("Generating chat response for query: %s", query)

        if not retrieved_context:
            return "I couldn't find any relevant context in the selected files."

        ranked_context = sorted(
            retrieved_context,
            key=lambda item: int(item.get("score", 0)),
            reverse=True,
        )[:4]
        top_score = int(ranked_context[0].get("score", 0))
        if top_score <= 0:
            return "I don't have enough direct evidence in the selected files to answer that reliably."

        if self.is_ollama_available():
            try:
                return self._generate_ollama_response(query, ranked_context, conversation or [])
            except Exception as exc:
                logger.warning("Ollama chat failed, falling back: %s", exc)

        return self._generate_local_response(query, ranked_context)

    def is_ollama_available(self) -> bool:
        """Return whether the local Ollama endpoint is reachable."""
        try:
            req = request.Request(f"{self.ollama_host}/api/tags", method="GET")
            with request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except Exception:
            return False

    def _generate_local_response(self, query: str, ranked_context: list[dict[str, Any]]) -> str:
        """Generate a more conversational local fallback from retrieved snippets."""
        if self._looks_like_summary_request(query):
            combined = " ".join(
                self._clean_text(item.get("text", "")) for item in ranked_context[:3]
            ).strip()
            summary = self.summarizer.summarize(combined)
            if summary:
                sources = ", ".join(
                    f"{item.get('file_name', 'Unnamed')} {item.get('section_id', 'section')}"
                    for item in ranked_context[:2]
                )
                return f"Here’s the short version: {summary}\n\nI grounded that in {sources}."

        lead = self._build_local_lead(query)
        details = []
        for item in ranked_context[:3]:
            snippet = self._clean_text(item.get("text", ""))
            if not snippet:
                continue
            distilled = self.summarizer.summarize(snippet)
            source = item.get("file_name", "Unnamed file")
            section = item.get("section_id", "section")
            details.append(f"- {distilled} ({source}, {section})")

        if not details:
            return "I found matching sections, but they were too thin to restate cleanly."

        return lead + "\n\n" + "\n".join(details)

    def _generate_ollama_response(
        self,
        query: str,
        ranked_context: list[dict[str, Any]],
        conversation: list[dict[str, str]],
    ) -> str:
        """Call the local Ollama chat API."""
        system_prompt = (
            "You are a grounded document assistant inside a Streamlit app. "
            "Answer naturally and clearly, but only use the retrieved evidence. "
            "If the evidence is incomplete, say that plainly. Keep answers concise and useful."
        )
        evidence_block = "\n\n".join(
            (
                f"Source: {item.get('file_name', 'Unnamed')} | Section: {item.get('section_id', 'section')}\n"
                f"Evidence: {self._clean_text(item.get('text', ''))}"
            )
            for item in ranked_context
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation[-6:])
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    f"Retrieved evidence:\n{evidence_block}\n\n"
                    "Answer using only this evidence."
                ),
            }
        )
        payload = {
            "model": self.ollama_model,
            "messages": messages,
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
        content = data.get("message", {}).get("content", "")
        if not content:
            raise ValueError("No content returned from Ollama chat.")
        return str(content).strip()

    def _looks_like_summary_request(self, query: str) -> bool:
        """Detect when the user is asking for a summary-style answer."""
        normalized = query.lower()
        return any(
            phrase in normalized
            for phrase in ("summary", "summarize", "overview", "what happened", "what is this about")
        )

    def _build_local_lead(self, query: str) -> str:
        """Build a less robotic lead-in for local answers."""
        normalized = query.lower()
        if "why" in normalized:
            return "From the matching sections, the clearest explanation is:"
        if "how" in normalized:
            return "Here’s how the selected files answer that:"
        if "what" in normalized:
            return "Here’s what the matching evidence says:"
        return "Here’s the closest grounded answer I can give from the selected files:"

    def _clean_text(self, text: Any) -> str:
        """Normalize whitespace in retrieved snippets."""
        return " ".join(str(text).split())
