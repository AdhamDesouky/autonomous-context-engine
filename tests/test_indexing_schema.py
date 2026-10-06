import sys
from pathlib import Path

import pytest
from pydantic import ValidationError


backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.schemas.research import IndexResponse


def test_index_response_accepts_successful_indexing():
    response = IndexResponse.model_validate(
        {
            "source_file": "manual.pdf",
            "indexed": True,
            "chunks_indexed": 12,
            "detail": "Indexed 12 chunks.",
        }
    )

    assert response.chunks_indexed == 12


def test_index_response_rejects_negative_chunk_count():
    with pytest.raises(ValidationError):
        IndexResponse.model_validate(
            {
                "source_file": "manual.pdf",
                "indexed": True,
                "chunks_indexed": -1,
                "detail": "Invalid",
            }
        )
