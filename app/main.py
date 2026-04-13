"""
FastAPI backend — production REST API with health check,
query endpoint, evaluation results, and full middleware stack.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from app.rag import rag_query, get_specialist_recommendation
from app.guardrails import check_content
from app.function_calling import route_query
import json, os

load_dotenv()

app = FastAPI(
    title="Medical Q&A RAG API",
    description=(
        "Azure OpenAI + Azure AI Search powered medical "
        "information assistant using NIH MedQuAD dataset"
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_function_calling: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list
    latency_ms: int
    retrieval_ms: int
    tokens_used: int
    routed_action: str = "search_knowledge_base"
    safety_checked: bool = True


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Medical Q&A RAG API",
        "dataset": "MedQuAD (NIH — 47,457 Q&A pairs)"
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    # 1. Input safety guardrail
    safety = check_content(request.question)
    if not safety["safe"]:
        raise HTTPException(
            status_code=400,
            detail=f"Query blocked: {safety['reason']}"
        )

    routed_action = "search_knowledge_base"
    topic_type = None

    # 2. Function calling router
    if request.use_function_calling:
        routing = route_query(request.question)
        routed_action = routing.get("action", "search_knowledge_base")
        topic_type = routing.get("topic_type")

        # Emergency redirect
        if routed_action == "emergency_redirect":
            return QueryResponse(
                answer=(
                    "🚨 **This sounds like a medical emergency.**\n\n"
                    "**Please call 000 (Australia) or your local emergency number immediately.**\n\n"
                    "Do not wait — go to your nearest emergency department or call an ambulance now."
                ),
                sources=[],
                latency_ms=0,
                retrieval_ms=0,
                tokens_used=0,
                routed_action="emergency_redirect"
            )

        # Specialist recommendation
        if routed_action == "recommend_specialist":
            answer = get_specialist_recommendation(
                routing.get("condition_area", "general"),
                routing.get("urgency", "routine")
            )
            return QueryResponse(
                answer=answer,
                sources=[],
                latency_ms=0,
                retrieval_ms=0,
                tokens_used=0,
                routed_action="recommend_specialist"
            )

    # 3. RAG pipeline
    result = rag_query(
        request.question,
        top_k=request.top_k,
        topic_type=topic_type
    )

    # 4. Output safety guardrail
    output_safety = check_content(result["answer"])
    if not output_safety["safe"]:
        result["answer"] = (
            "Response filtered by content safety policy. "
            "Please consult a healthcare professional."
        )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=result["latency_ms"],
        retrieval_ms=result["retrieval_ms"],
        tokens_used=result["tokens_used"],
        routed_action=routed_action,
        safety_checked=True
    )


@app.get("/eval-results")
def eval_results():
    try:
        with open("eval/eval_results.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"message": "Run python app/evaluate.py to generate scores"}