"""
Core RAG pipeline — Azure OpenAI + Azure AI Search hybrid retrieval.
Hybrid = vector + BM25 keyword + semantic reranking.
Includes: latency tracking, source attribution, confidence scoring.
"""
import os
import time
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery, QueryType, QueryCaptionType
)
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

SYSTEM_PROMPT = """You are a helpful medical information assistant.
Your knowledge comes from the NIH (National Institutes of Health) medical database.

Rules:
1. Answer using ONLY the provided context — never guess or hallucinate
2. Always add: "⚠️ This is for informational purposes only. Always consult a qualified healthcare provider."
3. If the context doesn't answer the question, say so clearly
4. Cite which part of the context you used
5. Keep answers clear and accessible — avoid unnecessary jargon

Format:
**Answer:** [your answer based on context]
**Based on:** [brief source note]
**Important:** ⚠️ This is for informational purposes only. Always consult a healthcare professional.
"""

SPECIALIST_MAP = {
    "heart": "cardiologist",
    "cardiac": "cardiologist",
    "skin": "dermatologist",
    "brain": "neurologist",
    "nerve": "neurologist",
    "bone": "orthopaedist",
    "joint": "rheumatologist",
    "lung": "pulmonologist",
    "breathing": "pulmonologist",
    "kidney": "nephrologist",
    "stomach": "gastroenterologist",
    "digestive": "gastroenterologist",
    "eye": "ophthalmologist",
    "mental": "psychiatrist",
    "anxiety": "psychiatrist",
    "depression": "psychiatrist",
    "child": "paediatrician",
    "cancer": "oncologist",
    "hormone": "endocrinologist",
    "diabetes": "endocrinologist",
}


def get_specialist_recommendation(condition_area: str, urgency: str) -> str:
    area_lower = condition_area.lower()
    specialist = next(
        (v for k, v in SPECIALIST_MAP.items() if k in area_lower),
        "general practitioner (GP)"
    )
    urgency_text = {
        "routine": "at your next available appointment",
        "soon": "within the next few days",
        "urgent": "as soon as possible — today if you can"
    }.get(urgency, "when convenient")

    return (
        f"Based on your description, I'd suggest seeing a **{specialist}** "
        f"{urgency_text}. Your GP can also provide a referral if needed.\n\n"
        f"⚠️ This is for informational purposes only. Always consult a healthcare professional."
    )


def embed(text: str) -> list:
    response = openai_client.embeddings.create(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        input=text
    )
    return response.data[0].embedding


def retrieve(query: str, top_k: int = 5, topic_type: str = None) -> list:
    """Hybrid retrieval: vector + keyword + semantic reranking."""
    query_embedding = embed(query)

    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=top_k,
        fields="embedding"
    )

    # Optional filter by topic type
    filter_expr = f"qtype eq '{topic_type}'" if topic_type else None

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name="semantic-config",
        query_caption=QueryCaptionType.EXTRACTIVE,
        filter=filter_expr,
        select=["id", "content", "question", "focus", "qtype"],
        top=top_k
    )

    return [
        {
            "id": r["id"],
            "content": r["content"],
            "question": r.get("question", ""),
            "focus": r.get("focus", ""),
            "score": r.get("@search.score", 0),
            "reranker_score": r.get("@search.reranker_score", 0)
        }
        for r in results
    ]


def generate_answer(query: str, docs: list) -> dict:
    context = "\n\n---\n\n".join([
        f"Source {i+1} (related to: {d['focus']}):\n{d['content']}"
        for i, d in enumerate(docs)
    ])

    t0 = time.time()
    response = openai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0,
        max_tokens=600
    )
    latency_ms = round((time.time() - t0) * 1000)

    return {
        "answer": response.choices[0].message.content,
        "latency_ms": latency_ms,
        "tokens_used": response.usage.total_tokens,
        "sources": [d["question"] for d in docs if d.get("question")]
    }


def rag_query(query: str, top_k: int = 5, topic_type: str = None) -> dict:
    t0 = time.time()
    docs = retrieve(query, top_k, topic_type)
    retrieval_ms = round((time.time() - t0) * 1000)

    if not docs:
        return {
            "answer": (
                "I couldn't find relevant information in the medical knowledge base. "
                "Please consult a healthcare professional for this question.\n\n"
                "⚠️ This assistant is for informational purposes only."
            ),
            "sources": [],
            "latency_ms": 0,
            "retrieval_ms": retrieval_ms,
            "tokens_used": 0,
            "retrieved_docs": []
        }

    result = generate_answer(query, docs)
    result["retrieval_ms"] = retrieval_ms
    result["retrieved_docs"] = docs
    return result