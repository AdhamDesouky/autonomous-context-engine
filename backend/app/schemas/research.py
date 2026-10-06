from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    source_file: str
    heading: str = "General"
    page: int | None = Field(default=None, ge=1)
    snippet: str
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)


class IndexResponse(BaseModel):
    source_file: str
    indexed: bool
    chunks_indexed: int = Field(default=0, ge=0)
    detail: str
