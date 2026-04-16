import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.llm.document_analysis_engine import DocumentAnalysisEngine


def test_document_analysis_local_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")

    engine = DocumentAnalysisEngine()
    result = {
        "status": "success",
        "file_name": "candidate_resume.docx",
        "file_type": "docx",
        "metadata": {
            "paragraph_count": 12,
            "table_count": 1,
        },
        "content": {
            "text": (
                "Skills\nPython, SQL, Streamlit, AWS\n\n"
                "Experience\nBuilt analytics tooling for document intelligence.\n\n"
                "Education\nB.Tech in Computer Science"
            ),
            "sections": [
                {"id": "paragraph_1", "type": "section", "text": "Skills\nPython, SQL, Streamlit, AWS"},
                {
                    "id": "paragraph_2",
                    "type": "section",
                    "text": "Experience\nBuilt analytics tooling for document intelligence.",
                },
            ],
            "tables": [],
        },
        "warnings": [],
        "errors": [],
    }

    analysis = engine.analyze_result(result)

    assert analysis["status"] == "ready"
    assert analysis["backend"] == "Local grounded mode"
    assert analysis["mode_detection"]["mode"] == "resume"
    assert analysis["summary"]
    assert len(analysis["insights"]) == 3
    assert len(analysis["recommended_questions"]) == 3
    assert analysis["evidence"]
