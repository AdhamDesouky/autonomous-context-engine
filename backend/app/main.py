# backend/app/main.py
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool

# Ensure backend directory is in the Python search path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agent.graph import ResearchAgent

# 1. Define Request/Response Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, description="The research question to execute.")

class QueryResponse(BaseModel):
    answer: str

# Global pointer for the agent
research_agent: ResearchAgent = None

# 2. Manage Application Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global research_agent
    print("[*] Server Startup: Loading Vector Models and LangGraph Agent in memory...")
    research_agent = ResearchAgent()
    print("[+] System Ready. Accepting connections.")
    yield
    print("[*] Server Shutdown: Releasing resources.")
    research_agent = None

# 3. Initialize FastAPI
app = FastAPI(
    title="Agentic RAG Engine API",
    description="Autonomous AI Research-to-Code Engine",
    version="1.0.0",
    lifespan=lifespan
)

# 4. Define Endpoints
@app.post("/api/v1/research/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    if not research_agent:
        raise HTTPException(status_code=503, detail="Agent subsystem offline.")
    
    try:
        # Offload the synchronous graph execution to a background thread
        final_answer = await run_in_threadpool(research_agent.run, request.query)
        return QueryResponse(answer=final_answer)
    except Exception as e:
        print(f"[!] Error during execution: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal agent execution failed.")

if __name__ == "__main__":
    import uvicorn
    # Local execution entry point
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)