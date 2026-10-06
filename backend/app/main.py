# backend/app/main.py
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool

# Ensure backend directory is in sys.path before local imports
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agent.graph import ResearchAgent


# 1. Request / Response Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, description="The research question to execute.")

class QueryResponse(BaseModel):
    answer: str


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
        final_answer = await run_in_threadpool(agent.run, request.query)
        return QueryResponse(answer=final_answer)
    except Exception as e:
        print(f"[!] Error during execution: {e}")
        raise HTTPException(status_code=500, detail="Internal agent execution failed.")


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