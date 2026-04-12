"""
Loads MedQuAD dataset from HuggingFace (47,457 NIH medical Q&A pairs),
chunks the answers as knowledge base documents, embeds them, and uploads
to Azure AI Search with hybrid index (vector + keyword + semantic).
"""
import os
from dotenv import load_dotenv
from datasets import load_dataset
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField,
    SearchFieldDataType, VectorSearch,
    HnswAlgorithmConfiguration, VectorSearchProfile,
    SearchField, SemanticConfiguration, SemanticSearch,
    SemanticPrioritizedFields, SemanticField
)
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX")

index_client = SearchIndexClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)


def create_index():
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String,
                        analyzer_name="en.microsoft"),
        SearchableField(name="question", type=SearchFieldDataType.String),
        SimpleField(name="focus", type=SearchFieldDataType.String,
                    filterable=True),
        SimpleField(name="qtype", type=SearchFieldDataType.String,
                    filterable=True),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="hnsw-profile"
        )
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
        profiles=[VectorSearchProfile(
            name="hnsw-profile",
            algorithm_configuration_name="hnsw-algo"
        )]
    )

    semantic_config = SemanticConfiguration(
        name="semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")],
            keywords_fields=[SemanticField(field_name="question")]
        )
    )

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=SemanticSearch(configurations=[semantic_config])
    )
    index_client.create_or_update_index(index)
    print(f"✅ Index '{INDEX_NAME}' created")


def embed(text: str) -> list:
    response = openai_client.embeddings.create(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        input=text[:8000]  # safety trim
    )
    return response.data[0].embedding


def ingest(max_docs: int = 300):
    """
    Loads MedQuAD — 47,457 NIH medical Q&A pairs covering diseases,
    drugs, symptoms, treatments. Uses the answer as the KB document
    and the question as metadata for better retrieval.
    """
    print("📥 Loading MedQuAD from HuggingFace...")
    dataset = load_dataset(
        "keivalya/MedQuad-MedicalQnADataset",
        split="train"
    )
    print(f"   Loaded {len(dataset)} Q&A pairs")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60
    )

    batch, total = [], 0

    for i, row in enumerate(dataset):
        if i >= max_docs:
            break

        question = row.get("Question", "").strip()
        answer = row.get("Answer", "").strip()
        focus = row.get("qtype", "general")  # disease, symptom, treatment...

        if not answer or len(answer) < 30:
            continue

        # Use the answer as the knowledge document
        # Chunk long answers to stay within retrieval window
        chunks = splitter.split_text(answer)

        for j, chunk in enumerate(chunks):
            doc_id = f"medquad_{i}_{j}"
            embedding = embed(f"{question} {chunk}")

            batch.append({
                "id": doc_id,
                "content": chunk,
                "question": question,
                "focus": focus[:100] if focus else "general",
                "qtype": focus[:50] if focus else "general",
                "embedding": embedding
            })

            if len(batch) >= 50:
                search_client.upload_documents(batch)
                total += len(batch)
                print(f"   Uploaded {total} chunks so far...")
                batch = []

    if batch:
        search_client.upload_documents(batch)
        total += len(batch)

    print(f"\n✅ Ingested {total} chunks from {min(max_docs, len(dataset))} Q&A pairs")


if __name__ == "__main__":
    create_index()
    ingest(max_docs=300)  # ~300 docs well within Azure free tier quota