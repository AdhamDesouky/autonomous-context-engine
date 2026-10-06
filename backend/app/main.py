# backend/app/main.py
import sys
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool

# Ensure backend directory is in sys.path before local imports
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agent.graph import ResearchAgent
from app.schemas.research import IndexResponse, QueryResponse


# 1. Request / Response Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, description="The research question to execute.")

# 2. Application Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] Server Startup: Loading Vector Models and LangGraph Agent into memory...")
    agent_instance = ResearchAgent()
    app.state.research_agent = agent_instance
    print("[+] System Ready. Accepting connections.")
    
    yield
    
    print("[*] Server Shutdown: Releasing resources.")
    app.state.research_agent = None


# 3. FastAPI Initialization
app = FastAPI(
    title="Agentic RAG Engine API",
    description="Autonomous AI Research-to-Code Engine",
    version="1.0.0",
    lifespan=lifespan,
)


# 4. Endpoints
@app.post("/api/v1/research/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, req: Request):
    agent: ResearchAgent | None = getattr(req.app.state, "research_agent", None)
    if not agent:
        raise HTTPException(status_code=503, detail="Agent subsystem offline.")

    try:
        result = await run_in_threadpool(agent.run_with_sources, request.query)
        return QueryResponse.model_validate(result)
    except Exception as e:
        print(f"[!] Error during execution: {e}")
        raise HTTPException(status_code=500, detail="Internal agent execution failed.")


@app.post("/api/v1/research/documents", response_model=IndexResponse)
async def index_document(
    req: Request,
    file: UploadFile = File(..., description="A PDF document to index."),
):
    agent: ResearchAgent | None = getattr(req.app.state, "research_agent", None)
    if not agent:
        raise HTTPException(status_code=503, detail="Agent subsystem offline.")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF documents are supported.")

    source_file = Path(file.filename).name
    if source_file in agent.store.get_indexed_files():
        return IndexResponse(
            source_file=source_file,
            indexed=False,
            detail="Document is already indexed.",
        )

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
            while chunk := await file.read(1024 * 1024):
                temporary_file.write(chunk)

        chunks_indexed = await run_in_threadpool(
            agent.store.index_document, str(temporary_path), source_file
        )
        if chunks_indexed == 0:
            return IndexResponse(
                source_file=source_file,
                indexed=False,
                detail="No indexable text was found in the PDF.",
            )
        return IndexResponse(
            source_file=source_file,
            indexed=True,
            chunks_indexed=chunks_indexed,
            detail=f"Indexed {chunks_indexed} chunks.",
        )
    except Exception as exc:
        print(f"[!] Error while indexing {source_file}: {exc}")
        raise HTTPException(status_code=500, detail="Document indexing failed.")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        await file.close()


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(req: Request):
    agent: ResearchAgent | None = getattr(req.app.state, "research_agent", None)
    if not agent:
        raise HTTPException(status_code=503, detail="Agent initializing")

    try:
        # Queries Qdrant collection info through the agent instance store
        info = agent.store.client.get_collection(agent.store.collection_name)
        return {
            "status": "ready",
            "collection": agent.store.collection_name,
            "points_count": info.points_count,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)