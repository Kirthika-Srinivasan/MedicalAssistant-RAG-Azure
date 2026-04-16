# debug_alzheimers.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
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

query = "What are the early signs of Alzheimer's disease?"

# Test 1: Is Alzheimer's even in the index?
print("Test 1: Keyword search for 'alzheimer'")
results = list(search_client.search(
    search_text="alzheimer",
    select=["id", "question", "content"],
    top=5
))
print(f"  Results: {len(results)}")
for r in results:
    print(f"  - {r['question'][:80]}")

print()

# Test 2: Can we embed successfully?
print("Test 2: Embedding")
try:
    emb = openai_client.embeddings.create(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        input=query
    ).data[0].embedding
    print(f"  ✅ Embedding OK: {len(emb)} dims")
except Exception as e:
    print(f"  ❌ Embedding FAILED: {e}")
    emb = None

print()

# Test 3: Vector search
if emb:
    print("Test 3: Vector search")
    vq = VectorizedQuery(vector=emb, k_nearest_neighbors=5, fields="embedding")
    results = list(search_client.search(
        search_text=query,
        vector_queries=[vq],
        select=["id", "question", "content"],
        top=5
    ))
    print(f"  Results: {len(results)}")
    for r in results:
        print(f"  - score:{r.get('@search.score',0):.3f} | {r['question'][:80]}")