# backend/app/agent/graph.py
import os
import re
import sys
from typing import Any, Dict, List, TypedDict
from pathlib import Path
from pydantic import BaseModel, Field
from fastembed.rerank.cross_encoder import TextCrossEncoder

# Ensure backend directory is in the Python search path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph

from app.retrieval.vector_store import HybridVectorStore

load_dotenv()

# Utility to clean reasoning traces (<think> tags) from models like Qwen
def clean_reasoning(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# 1. State Schema with loop counter
class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    retry_count: int
    generation_retries: int  

class GradeHallucinations(BaseModel):
    """Binary score for hallucination presence in generation answer."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")
    
class ResearchAgent:
    def check_hallucinations_and_utility(self, state: AgentState) -> str:
        """
        Determines whether the generation is grounded in the document and answers question.
        """
        question = state["question"]
        documents = state["context"]
        generation = state["answer"]
        retries = state.get("generation_retries", 0)

        if retries >= 3:
            print("[*] Max generation retries reached. Outputting best attempt.")
            return "useful"

        print("[*] [Edge Decision] Evaluating Hallucinations...")
        
        # 1. Hallucination Check
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
        Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by the set of facts."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
        ])
        hallucination_grader = prompt | self.llm.with_structured_output(GradeHallucinations)
        
        score = hallucination_grader.invoke({"documents": documents, "generation": generation})
        
        if score.binary_score == "yes":
            print("[*] [Edge Decision] Generation is grounded. Evaluating Utility...")
            # 2. Utility Check
            system = """You are a grader assessing whether an answer addresses / resolves a question.
            Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question."""
            prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
            ])
            answer_grader = prompt | self.llm.with_structured_output(GradeAnswer)
            
            score = answer_grader.invoke({"question": question, "generation": generation})
            if score.binary_score == "yes":
                print("[*] [Edge Decision] Answer is useful. Terminating execution.")
                return "useful"
            else:
                print("[*] [Edge Decision] Answer is NOT useful. Re-routing to 'generate'.")
                return "not useful"
        else:
            print("[*] [Edge Decision] Hallucination detected! Re-routing to 'generate'.")
            return "not supported"

    def __init__(self):
        print("[*] Initializing Self-Correcting Agent State Graph...")
        
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            model_name="openai/gpt-oss-20b",
            temperature=0
        )
        self.store = HybridVectorStore()
        
        # --- NEW: Initialize Cross-Encoder ---
        print("[*] Initializing Cross-Encoder Reranker...")
        self.reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
        
        # Build the Graph
        self.workflow = StateGraph(AgentState)
        
        # Add Nodes
        self.workflow.add_node("retrieve", self.retrieve_node)
        self.workflow.add_node("grade_documents", self.grade_documents_node)
        self.workflow.add_node("rewrite_query", self.rewrite_query_node)
        self.workflow.add_node("generate", self.generate_node)
        
        # Add Static Edges
        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_edge("rewrite_query", "retrieve")
        self.workflow.add_conditional_edges(
            "generate",
            self.check_hallucinations_and_utility,
            {
                "not supported": "generate",
                "not useful": "generate",
                "useful": END
            }
        )
        
        # Add Conditional Routing Edge
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "generate": "generate",
                "rewrite_query": "rewrite_query"
            }
        )
        
        self.app = self.workflow.compile()

    def retrieve_node(self, state: AgentState):
        """Node 1: High-Recall Hybrid search + High-Precision Cross-Encoder Reranking."""
        query = state["question"]
        print(f"\n[*] [Node: Retrieve] Fetching context for: '{query}'")
        
        # 1. High Recall Phase: Fetch 15 chunks from Qdrant
        results = self.store.hybrid_search(query, limit=15)
        raw_chunks = [res.payload.get("text", "") for res in results]
        
        if not raw_chunks:
            return {"context": []}
            
        print(f"[*] [Node: Retrieve] Reranking {len(raw_chunks)} chunks using Cross-Encoder...")
        
        # 2. High Precision Phase: Cross-Encoder Scoring
        # FastEmbed yields an iterable of float scores corresponding to the documents
        scores = list(self.reranker.rerank(query, raw_chunks))
        
        # Handle cases where the library might wrap the output in a single nested array
        if len(scores) == 1 and hasattr(scores[0], '__len__'):
            scores = scores[0]
            
        # 3. Sort chunks by highest semantic score
        scored_chunks = list(zip(raw_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Keep only the top 3 densest chunks for the LLM context window
        best_chunks = [chunk for chunk, score in scored_chunks[:3]]
        
        print(f"    -> Top chunk rerank score: {scored_chunks[0][1]:.4f}")
        
        return {"context": best_chunks}

    def grade_documents_node(self, state: AgentState):
        """Node 2: Evaluates relevance of retrieved chunks; discards noise."""
        print("[*] [Node: Grade Documents] Evaluating chunk relevance...")
        question = state["question"]
        raw_chunks = state.get("context", [])
        
        grader_template = """You are an evaluator assessing whether an excerpt from a research paper is relevant to a user question.
        If the excerpt contains ANY information that helps answer at least one part of the question, mark it as relevant.
        Respond with ONLY 'yes' or 'no'. No explanation, no punctuation.

        Question: {question}
        Excerpt: {excerpt}

        Relevant (yes/no):"""
        
        grader_prompt = PromptTemplate.from_template(grader_template)
        grader_chain = grader_prompt | self.llm | StrOutputParser()
        
        relevant_chunks = []
        for i, chunk in enumerate(raw_chunks, 1):
            raw_eval = grader_chain.invoke({"question": question, "excerpt": chunk})
            grade = clean_reasoning(raw_eval).strip().lower()
            
            if "yes" in grade:
                print(f"    -> Chunk {i}: [RELEVANT]")
                relevant_chunks.append(chunk)
            else:
                print(f"    -> Chunk {i}: [IRRELEVANT - DROPPED]")
                
        return {"context": relevant_chunks}

    def rewrite_query_node(self, state: AgentState):
        """Node 3: Reformulates query for better vector retrieval."""
        current_retries = state.get("retry_count", 0) + 1
        original_query = state["question"]
        print(f"[*] [Node: Rewrite Query] Attempt {current_retries}: Optimizing search query...")

        rewrite_template = """You are an AI research specialist. 
        The previous retrieval query failed to surface relevant passages from an arXiv paper.
        Rewrite the query into precise academic terminology optimized for semantic and keyword retrieval.
        Return ONLY the rewritten query text.

        Original Query: {question}

        Rewritten Academic Query:"""

        prompt = PromptTemplate.from_template(rewrite_template)
        chain = prompt | self.llm | StrOutputParser()
        
        rewritten = clean_reasoning(chain.invoke({"question": original_query})).strip('"\n ')
        print(f"    -> Optimized Query: '{rewritten}'")
        
        return {"question": rewritten, "retry_count": current_retries}

    def decide_to_generate(self, state: AgentState) -> str:
        """Conditional Router: Determines if we synthesize or loop back."""
        has_context = len(state.get("context", [])) > 0
        retries = state.get("retry_count", 0)
        max_retries = 2

        if has_context:
            print("[*] [Edge Decision] Relevant context confirmed -> Routing to 'generate'")
            return "generate"
        
        if retries >= max_retries:
            print(f"[*] [Edge Decision] Max retries ({max_retries}) reached -> Routing to 'generate' with fallback")
            return "generate"

        print("[*] [Edge Decision] Zero relevant chunks -> Routing to 'rewrite_query' (Loop)")
        return "rewrite_query"

    def generate_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 4: Grounded synthesis."""
        print("[*] [Node: Generate] Synthesizing final answer...")
        question = state["question"]
        context_chunks = state.get("context", [])
        retries = state.get("generation_retries", 0)
        
        if not context_chunks:
            return {
                "answer": "I do not have sufficient relevant context from the document to answer this query accurately.",
                "generation_retries": retries + 1
            }

        context_str = "\n\n---\n\n".join(context_chunks)
        
        template = """You are an expert AI research assistant.
        Use ONLY the following verified chunks to answer the user's question.
        If the context does not contain the answer, state that clearly.
        Do not use outside knowledge.

        Question: {question}

        Verified Context:
        {context}

        Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        raw_answer = chain.invoke({"question": question, "context": context_str})
        clean_answer = clean_reasoning(raw_answer)
        return {"answer": clean_answer, "generation_retries": retries + 1}

    def run(self, query: str):
        print(f"\n{'='*60}\n[>>>] Triggering Self-Correcting Workflow: '{query}'\n{'='*60}")
        initial_state = {
            "question": query,
            "context": [],
            "answer": "",
            "retry_count": 0,
            "generation_retries": 0
        }
        
        final_state = self.app.invoke(initial_state)
        return final_state["answer"]

if __name__ == "__main__":
    agent = ResearchAgent()
    
    # Test with a colloquial/vague query to observe the grader & routing in action
    test_query = "What technique was used for adapting the learning speed and what was the warmup limit?"
    final_output = agent.run(test_query)
    
    print("\n[=== FINAL VERIFIED ANSWER ===]")
    print(final_output)