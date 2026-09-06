import os
import sys
from pathlib import Path
from pydantic import BaseModel

# Point Python to the backend directory so we can import the Agent
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from langchain_openai import ChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval import assert_test

from app.agent.graph import ResearchAgent

# 1. Override DeepEval's default OpenAI Judge with our Free Groq Endpoint
class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model_name = "openai/gpt-oss-20b"
        self.chat_model = ChatOpenAI(
            openai_api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            model_name=self.model_name,
            temperature=0
        )

    def load_model(self):
        return self.chat_model

    def generate(self, prompt: str, schema: BaseModel | None = None):
        model = self.load_model()
        if schema:
            model = model.with_structured_output(schema)
            return model.invoke(prompt)
        return model.invoke(prompt).content

    async def a_generate(self, prompt: str, schema: BaseModel | None = None):
        model = self.load_model()
        if schema:
            model = model.with_structured_output(schema)
            return await model.ainvoke(prompt)
        res = await model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return self.model_name

# 2. Define the Test Suite
def test_optimizer_extraction():
    print("\n[*] Initializing Agent and Judge...")
    agent = ResearchAgent()
    judge = GroqJudge()
    
    query = "What specific optimizer was used, what was the exact learning rate, and did they use weight decay?"
    
    # Execute the LangGraph State Machine directly
    initial_state = {"question": query, "context": [], "answer": "", "retry_count": 0}
    final_state = agent.app.invoke(initial_state)
    
    # 3. Build the DeepEval Test Case
    test_case = LLMTestCase(
        input=query,
        actual_output=final_state["answer"],
        retrieval_context=final_state["context"]
    )
    
    # 4. Configure Metrics (Pass threshold = 0.7 out of 1.0)
    faithfulness = FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True)
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True)
    
    # 5. Execute Mathematical Evaluation
    assert_test(test_case, [faithfulness, relevancy])