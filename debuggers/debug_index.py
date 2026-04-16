# debug_index.py  — run from project root
import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
key      = os.getenv("AZURE_SEARCH_KEY")
index    = os.getenv("AZURE_SEARCH_INDEX")

print(f"Endpoint : {endpoint}")
print(f"Index    : {index}")
print(f"Key      : {key[:8] if key else 'NOT FOUND'}...")
print()

# --- Check what indexes exist on this resource ---
index_client = SearchIndexClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)
print("All indexes on this Azure AI Search resource:")
try:
    indexes = list(index_client.list_indexes())
    if not indexes:
        print("  ❌ NO INDEXES EXIST AT ALL")
        print("  Fix: re-run python ingest/ingest.py")
    for idx in indexes:
        print(f"  - {idx.name}")
except Exception as e:
    print(f"  ❌ Could not list indexes: {e}")

print()

# --- Check document count in the target index ---
search_client = SearchClient(
    endpoint=endpoint,
    index_name=index,
    credential=AzureKeyCredential(key)
)
print(f"Document count in '{index}':")
try:
    count = search_client.get_document_count()
    print(f"  Count: {count}")
    if count == 0:
        print("  ❌ INDEX EXISTS BUT IS EMPTY")
        print("  Fix: re-run python ingest/ingest.py")
    else:
        print(f"  ✅ {count} documents indexed")
except Exception as e:
    print(f"  ❌ Could not count: {e}")

print()

# --- Try fetching one document directly ---
print("Fetching first document by wildcard:")
try:
    results = list(search_client.search(
        search_text="*",
        select=["id", "content"],
        top=1
    ))
    if results:
        print(f"  ✅ Got a doc: id={results[0]['id']}")
        print(f"  Content: {results[0]['content'][:100]}...")
    else:
        print("  ❌ Wildcard search returned nothing — index is empty")
except Exception as e:
    print(f"  ❌ Failed: {e}")