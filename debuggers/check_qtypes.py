# check_qtypes.py
import os
from dotenv import load_dotenv
from pathlib import Path
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

# Get a sample of docs and see what qtype values exist
results = list(client.search(search_text="*", select=["id", "qtype"], top=50))
qtypes = set(r.get("qtype", "None") for r in results)
print(f"qtype values in your index: {qtypes}")