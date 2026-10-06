# backend/app/agent/graph.py
import re
import sys
from typing import Any, Dict, List, TypedDict
from pathlib import Path
from pydantic import BaseModel, Field

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph
from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.retrieval.vector_store import HybridVectorStore
from app.core.llm import configured_provider, create_chat_model

load_dotenv()

def clean_reasoning(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# 1. Semantically Correct State Schema
class AgentState(TypedDict):
    original_question: str
    current_query: str
    context: List[str]
    sources: List[Dict[str, Any]]
    answer: str
    retry_count: int
    generation_retries: int
    critique: str

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses the user question, 'yes' or 'no'")

class ResearchAgent:
    def __init__(self):
        print("[*] Initializing Production Self-Correcting Agent...")
        
        self.generation_provider = configured_provider("generation")
        self.critique_provider = configured_provider("critique")
        self.generation_llm = create_chat_model(
            self.generation_provider,
            role="generation",
        )
        self.critique_llm = create_chat_model(
            self.critique_provider,
            role="critique",
        )
        self.store = HybridVectorStore()
        self.reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
        
        # Graph Construction
        self.workflow = StateGraph(AgentState)
        
        # Nodes
        self.workflow.add_node("retrieve", self.retrieve_node)
        self.workflow.add_node("grade_documents", self.grade_documents_node)
        self.workflow.add_node("rewrite_query", self.rewrite_query_node)
        self.workflow.add_node("generate", self.generate_node)
        
        # Edges
        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")
        
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.decide_after_grading,
            {
                "generate": "generate",
                "rewrite_query": "rewrite_query"
            }
        )
        
        self.workflow.add_edge("rewrite_query", "retrieve")
        
        self.workflow.add_conditional_edges(
            "generate",
            self.evaluate_generation,
            {
                "regenerate_grounded": "generate",     # Hallucination fix: regenerate with strict critique
                "rewrite_and_search": "rewrite_query",  # Utility fix: fetch better context
                "useful": END                           # Exit
            }
        )
        
        self.app = self.workflow.compile()

    def retrieve_node(self, state: AgentState):
        query = state["current_query"]
        print(f"\n[*] [Node: Retrieve] Fetching context for: '{query}'")
        
        results = self.store.hybrid_search(query, limit=15)
        candidates = []
        for result in results:
            payload = result.payload or {}
            text = payload.get("text", "")
            if text:
                candidates.append({
                    "text": text,
                    "source_file": payload.get("source_file", "Unknown source"),
                    "heading": payload.get("heading", "General"),
                    "page": payload.get("page"),
                })

        raw_chunks = [candidate["text"] for candidate in candidates]
        
        if not raw_chunks:
            return {"context": [], "sources": []}
            
        print(f"[*] [Node: Retrieve] Reranking {len(raw_chunks)} chunks...")
        scores = list(self.reranker.rerank(query, raw_chunks))
        if len(scores) == 1 and hasattr(scores[0], '__len__'):
            scores = scores[0]
            
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top_candidates = scored[:3]
        top_chunks = [candidate["text"] for candidate, _ in top_candidates]
        sources = [
            {
                "source_file": candidate["source_file"],
                "heading": candidate["heading"],
                "page": candidate["page"],
                "snippet": candidate["text"][:500],
                "score": float(score),
            }
            for candidate, score in top_candidates
        ]
        
        return {"context": top_chunks, "sources": sources}

    def grade_documents_node(self, state: AgentState):
        print("[*] [Node: Grade Documents] Evaluating chunk relevance...")
        original_question = state["original_question"]
        raw_chunks = state.get("context", [])
        
        grader_prompt = PromptTemplate.from_template(
            """You are an evaluator assessing whether an excerpt from a research paper is relevant to a user question.
            If the excerpt contains ANY information that helps answer at least one part of the question, mark it as relevant.
            Respond with ONLY 'yes' or 'no'. No explanation, no punctuation.

            Question: {question}
            Excerpt: {excerpt}

            Relevant (yes/no):"""
        )
        grader_chain = grader_prompt | self.critique_llm | StrOutputParser()
        
        relevant = []
        relevant_sources = []
        sources = state.get("sources", [])
        for i, chunk in enumerate(raw_chunks, 1):
            eval_res = clean_reasoning(grader_chain.invoke({"question": original_question, "excerpt": chunk})).lower()
            if "yes" in eval_res:
                print(f"    -> Chunk {i}: [RELEVANT]")
                relevant.append(chunk)
                if i <= len(sources):
                    relevant_sources.append(sources[i - 1])
            else:
                print(f"    -> Chunk {i}: [IRRELEVANT - DROPPED]")
                
        return {"context": relevant, "sources": relevant_sources}

    def rewrite_query_node(self, state: AgentState):
        current_retries = state.get("retry_count", 0) + 1
        base_query = state["original_question"]
        print(f"[*] [Node: Rewrite Query] Attempt {current_retries}: Reformulating '{base_query}'")

        rewrite_prompt = PromptTemplate.from_template(
            "Rewrite this question into a targeted academic search query to find the missing details in an arXiv paper:\n{question}\nTargeted Query:"
        )
        chain = rewrite_prompt | self.generation_llm | StrOutputParser()
        rewritten = clean_reasoning(chain.invoke({"question": base_query})).strip('"\n ')
        print(f"    -> Reformulated Query: '{rewritten}'")
        
        return {"current_query": rewritten, "retry_count": current_retries}

    def decide_after_grading(self, state: AgentState) -> str:
        if len(state.get("context", [])) > 0:
            return "generate"
        if state.get("retry_count", 0) >= 2:
            print("[*] Max retrieval retries reached. Forcing fallback generation.")
            return "generate"
        return "rewrite_query"

    def generate_node(self, state: AgentState) -> Dict[str, Any]:
        print("[*] [Node: Generate] Synthesizing answer...")
        question = state["original_question"]
        context_chunks = state.get("context", [])
        retries = state.get("generation_retries", 0)
        critique = state.get("critique", "")

        if not context_chunks:
            return {
                "answer": "I do not have sufficient relevant context from the document to answer this query accurately.",
                "generation_retries": retries + 1,
                "critique": ""
            }

        context_str = "\n\n---\n\n".join(context_chunks)
        
        prompt_str = "Answer the user question using ONLY the provided verified context.\n"
        if critique:
            prompt_str += f"ATTENTION: Your previous attempt failed because: {critique}. Strict adherence to context is required.\n"
        prompt_str += "\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        chain = PromptTemplate.from_template(prompt_str) | self.generation_llm | StrOutputParser()
        answer = clean_reasoning(chain.invoke({"question": question, "context": context_str}))
        
        return {"answer": answer, "generation_retries": retries + 1, "critique": ""}

    def evaluate_generation(self, state: AgentState) -> str:
        question = state["original_question"]
        documents = state["context"]
        generation = state["answer"]
        gen_retries = state.get("generation_retries", 0)
        search_retries = state.get("retry_count", 0)

        if gen_retries >= 3 or not documents:
            return "useful"

        print("[*] [Edge Decision] Evaluating Grounding...")
        h_prompt = ChatPromptTemplate.from_messages([
            ("system", "Is this answer completely supported by the facts? Output 'yes' or 'no'."),
            ("human", "Facts:\n{documents}\n\nAnswer:\n{generation}")
        ])
        h_score = (h_prompt | self.critique_llm.with_structured_output(GradeHallucinations)).invoke({
            "documents": documents, "generation": generation
        })

        if h_score.binary_score != "yes":
            print("[*] [Edge Decision] Hallucination detected. Triggering grounded regeneration.")
            state["critique"] = "You included unsupported claims outside the provided facts."
            return "regenerate_grounded"

        print("[*] [Edge Decision] Answer grounded. Evaluating Utility against original query...")
        u_prompt = ChatPromptTemplate.from_messages([
            ("system", "Does this answer address the user's specific question? Output 'yes' or 'no'."),
            ("human", "Question:\n{question}\n\nAnswer:\n{generation}")
        ])
        u_score = (u_prompt | self.critique_llm.with_structured_output(GradeAnswer)).invoke({
            "question": question, "generation": generation
        })

        if u_score.binary_score == "yes":
            print("[*] [Edge Decision] Answer is grounded and useful. Done.")
            return "useful"

        if search_retries < 2:
            print("[*] [Edge Decision] Answer not useful. Re-routing to 'rewrite_query' for new context.")
            return "rewrite_and_search"
            
        return "useful"

    def run(self, query: str):
        initial_state = {
            "original_question": query,
            "current_query": query,
            "context": [],
            "sources": [],
            "answer": "",
            "retry_count": 0,
            "generation_retries": 0,
            "critique": ""
        }
        final_state = self.app.invoke(initial_state)
        return final_state["answer"]

    def run_with_sources(self, query: str) -> Dict[str, Any]:
        initial_state = {
            "original_question": query,
            "current_query": query,
            "context": [],
            "sources": [],
            "answer": "",
            "retry_count": 0,
            "generation_retries": 0,
            "critique": "",
        }
        final_state = self.app.invoke(initial_state)
        return {
            "answer": final_state["answer"],
            "sources": final_state.get("sources", []),
        }