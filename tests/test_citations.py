import sys
from pathlib import Path

from pydantic import ValidationError
import pytest


backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.schemas.research import QueryResponse


def test_query_response_accepts_citations():
    response = QueryResponse.model_validate(
        {
            "answer": "The paper used Adam.",
            "sources": [
                {
                    "source_file": "paper.pdf",
                    "heading": "Optimization",
                    "page": 5,
                    "snippet": "We used Adam with a learning rate of 0.001.",
                    "score": 0.91,
                }
            ],
        }
    )

    assert response.sources[0].source_file == "paper.pdf"
    assert response.sources[0].page == 5


def test_query_response_rejects_invalid_page():
    with pytest.raises(ValidationError):
        QueryResponse.model_validate(
            {
                "answer": "Answer",
                "sources": [
                    {
                        "source_file": "paper.pdf",
                        "snippet": "Text",
                        "page": 0,
                    }
                ],
            }
        )
