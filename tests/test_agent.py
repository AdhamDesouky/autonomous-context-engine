import os
import sys
import json
from pathlib import Path
from pydantic import BaseModel, ValidationError

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Point Python to the backend directory so we can import the Agent
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval import assert_test

from app.agent.graph import ResearchAgent

# 1. Use the configured Gemini endpoint as the DeepEval judge.
class GeminiJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
        self.chat_model = ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            model=self.model_name,
            temperature=0
        )

    def load_model(self):
        return self.chat_model

    @staticmethod
    def _parse_structured_response(content: str, schema: type[BaseModel]):
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].lstrip()

        payload = json.loads(cleaned)
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            for error in exc.errors():
                if error["type"] != "missing":
                    raise
                location = error["loc"]
                target = payload
                for key in location[:-1]:
                    target = target[key]
                target[location[-1]] = ""
            return schema.model_validate(payload)

    def generate(self, prompt: str, schema: BaseModel | None = None):
        model = self.load_model()
        if schema:
            response = model.invoke(prompt)
            return self._parse_structured_response(response.content, schema)
        return model.invoke(prompt).content

    async def a_generate(self, prompt: str, schema: BaseModel | None = None):
        model = self.load_model()
        if schema:
            response = await model.ainvoke(prompt)
            return self._parse_structured_response(response.content, schema)
        res = await model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return self.model_name

# 2. Define the Test Suite
def test_optimizer_extraction():
    print("\n[*] Initializing Agent and Judge...")
    agent = ResearchAgent()
    judge = GeminiJudge()
    
    query = "What specific optimizer was used, what was the exact learning rate, and did they use weight decay?"
    
    # Execute the LangGraph State Machine directly
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