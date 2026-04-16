# debug_search.py
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

load_dotenv()

print("=" * 50)
print("AZURE SEARCH DEBUG")
print("=" * 50)
print(f"Endpoint : {os.getenv('AZURE_SEARCH_ENDPOINT')}")
print(f"Index    : {os.getenv('AZURE_SEARCH_INDEX')}")
print(f"Key      : {os.getenv('AZURE_SEARCH_KEY')[:8]}...")
print()

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

# Test 1: Count all docs
print("Test 1: Count all documents in index")
try:
    results = list(search_client.search(
        search_text="*",
        select=["id"],
        top=1000
    ))
    print(f"  Total docs: {len(results)}")
    if len(results) == 0:
        print("  ❌ INDEX IS EMPTY — this is your problem")
        print("  Fix: re-run python ingest/ingest.py")
    else:
        print(f"  ✅ Index has {len(results)} documents")
except Exception as e:
    print(f"  ❌ Failed: {e}")

print()

# Test 2: Simple keyword search
print("Test 2: Keyword search for 'depression'")
try:
    results = list(search_client.search(
        search_text="depression",
        select=["id", "content"],
        top=3
    ))
    if results:
        print(f"  ✅ Found {len(results)} results")
        print(f"  First chunk: {results[0]['content'][:120]}...")
    else:
        print("  ❌ No results — index exists but search returns nothing")
except Exception as e:
    print(f"  ❌ Failed: {e}")

print()

# Test 3: Embedding + vector search
print("Test 3: Vector search for 'depression diagnosis treatment'")
try:
    openai_client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    embedding = openai_client.embeddings.create(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        input="depression diagnosis treatment"
    ).data[0].embedding
    print(f"  ✅ Embedding OK — {len(embedding)} dimensions")

    vector_query = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=3,
        fields="embedding"
    )
    results = list(search_client.search(
        search_text="depression",
        vector_queries=[vector_query],
        select=["id", "content"],
        top=3
    ))
    if results:
        print(f"  ✅ Vector search returned {len(results)} results")
        print(f"  First chunk: {results[0]['content'][:120]}...")
    else:
        print("  ❌ Vector search returned nothing")
except Exception as e:
    print(f"  ❌ Failed: {e}")

print()

# Test 4: Check if semantic search is failing
print("Test 4: Semantic search (may fail on free tier quota)")
try:
    from azure.search.documents.models import QueryType
    results = list(search_client.search(
        search_text="depression",
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name="semantic-config",
        select=["id", "content"],
        top=3
    ))
    if results:
        print(f"  ✅ Semantic search returned {len(results)} results")
    else:
        print("  ❌ Semantic search returned nothing — FREE TIER QUOTA LIKELY HIT")
except Exception as e:
    print(f"  ❌ Semantic search failed: {e}")
    print("  This is likely your problem — semantic quota exhausted on free tier")