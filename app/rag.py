"""
rag.py — Core RAG pipeline with improved retrieval and dynamic responses
"""
import os
import time
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery, QueryType, QueryCaptionType
)
from azure.core.credentials import AzureKeyCredential

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

# ── Improved system prompt — varied, conversational, not formulaic ─────────
SYSTEM_PROMPT = """You are a knowledgeable and warm medical information assistant.
Your knowledge comes from the NIH (National Institutes of Health) medical database.

RULES:
- Answer ONLY from the provided context — never guess or make up facts
- If context is insufficient, say so honestly and suggest seeing a doctor
- Always end with the disclaimer below
- VARY your response style — don't always use the same format
- Be conversational and clear, not robotic
- Use bullet points, numbered lists, or prose depending on what fits best
- Never say "typically" or "commonly" unless the context says so
- Cite sources naturally (e.g. "According to NIH sources..." or "The research indicates...")
- Do NOT add information from your training data if it's not backed up by research or NIH sources

DISCLAIMER (always include at end):
⚠️ This is for informational purposes only. Always consult a qualified healthcare provider.

STYLE GUIDE:
- For symptom questions: list symptoms clearly, mention when to seek care
- For treatment questions: explain options, mention that treatment varies per person  
- For cause questions: explain mechanisms simply, avoid jargon
- For general questions: be conversational and thorough
- Never start two responses the same way
- Keep it clear and concise
"""

# ── Expanded specialist map — fuzzy keyword matching ──────────────────────
SPECIALIST_MAP = [
    # (keywords_list, specialist_name)
    (["heart", "cardiac", "chest pain", "palpitation", "arrhythmia",
      "coronary", "cardiovascular", "blood pressure", "hypertension"],
     "cardiologist"),
    (["skin", "rash", "acne", "eczema", "psoriasis", "dermatitis",
      "mole", "lesion", "itching"],
     "dermatologist"),
    (["brain", "nerve", "neurology", "seizure", "epilepsy", "migraine",
      "stroke", "multiple sclerosis", "parkinson", "alzheimer", "dementia",
      "memory", "tremor", "numbness", "tingling"],
     "neurologist"),
    (["bone", "fracture", "joint pain", "arthritis", "spine", "back pain",
      "knee", "hip", "shoulder", "sports injury", "ligament"],
     "orthopaedist or rheumatologist"),
    (["lung", "breathing", "breath", "asthma", "copd", "pneumonia",
      "cough", "respiratory", "inhaler", "wheeze"],
     "pulmonologist"),
    (["kidney", "renal", "dialysis", "urinary", "urine", "bladder",
      "nephritis"],
     "nephrologist or urologist"),
    (["stomach", "bowel", "gut", "digestive", "ibs", "crohn", "colitis",
      "acid reflux", "heartburn", "liver", "gallbladder", "pancreas",
      "nausea", "vomiting", "diarrhea", "constipation"],
     "gastroenterologist"),
    (["eye", "vision", "sight", "glaucoma", "cataract", "retina",
      "blurred vision"],
     "ophthalmologist"),
    (["mental health", "anxiety", "depression", "panic", "ocd",
      "bipolar", "schizophrenia", "ptsd", "mood", "stress", "therapy",
      "psychiatry", "psychological"],
     "psychiatrist or psychologist"),
    (["child", "infant", "baby", "toddler", "pediatric", "newborn",
      "kid", "adolescent", "teenager"],
     "paediatrician"),
    (["cancer", "tumor", "tumour", "oncology", "chemotherapy",
      "radiation", "lymphoma", "leukemia", "malignant"],
     "oncologist"),
    (["hormone", "thyroid", "diabetes", "insulin", "adrenal",
      "pituitary", "metabolic", "obesity", "weight"],
     "endocrinologist"),
    (["allergy", "allergic", "anaphylaxis", "hives", "hay fever",
      "immune", "autoimmune", "immunology"],
     "allergist or immunologist"),
    (["ear", "hearing", "nose", "throat", "sinus", "tonsil", "ent",
      "voice", "swallowing", "hoarse"],
     "ENT specialist (otolaryngologist)"),
    (["pregnant", "pregnancy", "fertility", "period", "menstrual",
      "ovary", "uterus", "gynecology", "obstetric", "reproductive"],
     "gynaecologist or obstetrician"),
    (["teeth", "tooth", "gum", "dental", "mouth", "jaw", "cavity"],
     "dentist"),
    (["blood", "anaemia", "anemia", "platelet", "clotting", "bleeding",
      "haematology"],
     "haematologist"),
]


def get_specialist_recommendation(condition_area: str, urgency: str) -> str:
    """
    Improved specialist matching — searches across all keywords,
    not just exact single-word matches.
    """
    area_lower = condition_area.lower()

    specialist = None
    for keywords, spec_name in SPECIALIST_MAP:
        if any(kw in area_lower for kw in keywords):
            specialist = spec_name
            break

    if not specialist:
        specialist = "general practitioner (GP)"

    urgency_phrases = {
        "routine": "at your next available appointment — no rush",
        "soon": "within the next few days if possible",
        "urgent": "as soon as possible, ideally today"
    }
    urgency_text = urgency_phrases.get(urgency, "when you can")

    # Varied response templates so it doesn't always read the same
    templates = [
        f"Based on what you've described, a **{specialist}** would be the right person to see — {urgency_text}. Your GP can also provide a referral if needed.",
        f"For what you're describing, I'd suggest booking an appointment with a **{specialist}** — {urgency_text}. If you're unsure, start with your GP who can refer you.",
        f"This sounds like something a **{specialist}** specialises in. I'd recommend seeing one {urgency_text}. Your GP is always a good first call if you're not sure where to start.",
    ]

    answer = random.choice(templates)
    return f"{answer}\n\n⚠️ This is for informational purposes only. Always consult a healthcare professional."


def embed(text: str) -> list:
    """Convert text to embedding vector."""
    response = openai_client.embeddings.create(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        input=text[:8000]
    )
    return response.data[0].embedding


def retrieve(query: str, top_k: int = 5, topic_type: str = None) -> list:
    """
    Hybrid retrieval with three fallback levels.
    topic_type filter is intentionally disabled — causes false negatives.
    """
    print(f"\n--- RETRIEVE: '{query[:50]}' ---")

    # Step 1: get embedding
    try:
        query_embedding = embed(query)
        print(f"✅ Embedding: {len(query_embedding)} dims")
    except Exception as e:
        print(f"❌ Embedding failed: {e} — falling back to keyword only")
        query_embedding = None

    vector_queries = []
    if query_embedding:
        vector_queries = [VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="embedding"
        )]

    # filter_expr intentionally None — topic_type filter causes false negatives
    # because GPT's topic labels don't match the actual qtype values in the index
    filter_expr = None

    def parse_results(results) -> list:
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

    # Level 1: semantic hybrid (vector + BM25 + semantic reranking)
    if vector_queries:
        try:
            results = list(search_client.search(
                search_text=query,
                vector_queries=vector_queries,
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name="semantic-config",
                query_caption=QueryCaptionType.EXTRACTIVE,
                filter=filter_expr,
                select=["id", "content", "question", "focus", "qtype"],
                top=top_k
            ))
            docs = parse_results(results)
            if docs:
                print(f"✅ Semantic hybrid: {len(docs)} docs")
                return docs
            print("⚠️ Semantic returned 0, trying plain hybrid...")
        except Exception as e:
            print(f"⚠️ Semantic failed: {e}")

    # Level 2: plain hybrid (vector + BM25, no semantic reranking)
    if vector_queries:
        try:
            results = list(search_client.search(
                search_text=query,
                vector_queries=vector_queries,
                filter=filter_expr,
                select=["id", "content", "question", "focus", "qtype"],
                top=top_k
            ))
            docs = parse_results(results)
            if docs:
                print(f"✅ Plain hybrid: {len(docs)} docs")
                return docs
            print("⚠️ Plain hybrid returned 0, trying keyword only...")
        except Exception as e:
            print(f"⚠️ Plain hybrid failed: {e}")

    # Level 3: pure BM25 keyword (last resort, no vectors needed)
    try:
        results = list(search_client.search(
            search_text=query,
            filter=filter_expr,
            select=["id", "content", "question", "focus", "qtype"],
            top=top_k
        ))
        docs = parse_results(results)
        print(f"{'✅' if docs else '❌'} Keyword only: {len(docs)} docs")
        return docs
    except Exception as e:
        print(f"❌ All search levels failed: {e}")
        return []


def generate_answer(query: str, docs: list) -> dict:
    """Generate a varied, grounded answer from retrieved chunks."""
    context = "\n\n---\n\n".join([
        f"Source {i+1} (topic: {d['focus'] or 'general'}):\n{d['content']}"
        for i, d in enumerate(docs)
    ])

    t0 = time.time()
    response = openai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Context from NIH medical database:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Please answer based on the context above. "
                f"Vary your format — use lists, prose, or a mix depending "
                f"on what best suits this specific question."
            )}
        ],
        temperature=0.1,
        # Slightly above 0 for more natural varied responses
        # Still low enough to stay factual and grounded
        max_tokens=700
    )
    latency_ms = round((time.time() - t0) * 1000)

    return {
        "answer": response.choices[0].message.content,
        "latency_ms": latency_ms,
        "tokens_used": response.usage.total_tokens,
        "sources": list(dict.fromkeys(
            d["question"] for d in docs if d.get("question")
        ))
        # dict.fromkeys removes duplicates while preserving order
        # This stops the same source appearing 4 times in the list
    }


def rag_query(query: str, top_k: int = 5, topic_type: str = None) -> dict:
    """Full RAG pipeline: retrieve → generate."""
    t0 = time.time()
    docs = retrieve(query, top_k, topic_type)
    retrieval_ms = round((time.time() - t0) * 1000)

    if not docs:
        return {
            "answer": (
                "I wasn't able to find relevant information in the "
                "NIH knowledge base for that question. It may be outside "
                "the scope of our current dataset. Please consult a "
                "healthcare professional for personalised advice.\n\n"
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