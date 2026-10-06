import sys
from pathlib import Path

import httpx
import pytest


backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.main import app


class FakeStore:
    def __init__(self):
        self.indexed_files = set()
        self.calls = []

    def get_indexed_files(self):
        return self.indexed_files

    def index_document(self, file_path: str, source_file: str):
        self.calls.append((file_path, source_file))
        self.indexed_files.add(source_file)
        return 2


class FakeAgent:
    def __init__(self):
        self.store = FakeStore()


@pytest.mark.asyncio
async def test_upload_indexes_original_source_filename():
    agent = FakeAgent()
    app.state.research_agent = agent

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/research/documents",
            files={"file": ("manual.pdf", b"%PDF-test", "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json()["source_file"] == "manual.pdf"
    assert agent.store.calls[0][1] == "manual.pdf"


@pytest.mark.asyncio
async def test_upload_rejects_non_pdf():
    app.state.research_agent = FakeAgent()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/research/documents",
            files={"file": ("manual.txt", b"not a PDF", "text/plain")},
        )

    assert response.status_code == 415
