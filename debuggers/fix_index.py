# fix_index.py
import os
from dotenv import load_dotenv
from pathlib import Path
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
key = os.getenv("AZURE_SEARCH_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX")

client = SearchIndexClient(endpoint, AzureKeyCredential(key))

# Delete the bloated index
try:
    client.delete_index(index_name)
    print(f"✅ Deleted index '{index_name}'")
except Exception as e:
    print(f"⚠️  Could not delete: {e}")