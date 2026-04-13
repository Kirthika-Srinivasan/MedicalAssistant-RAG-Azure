# 🏥 Medical Q&A RAG Assistant
 A patient-facing health information assistant that answers questions about diseases, medications, symptoms, and treatments.

> Production-grade RAG system built on **Azure OpenAI + Azure AI Search**
> using the **NIH MedQuAD dataset** (47,457 medical Q&A pairs).
> Features hybrid retrieval, function calling, content safety guardrails,
> and RAGAS evaluation — deployed as FastAPI + Streamlit.

⚠️ *For informational purposes only. Not a substitute for professional medical advice.*

**🔗 Live Demo:** https://medicalassistant-rag-azure-by-kirthika.streamlit.app/

## 📊 RAGAS Evaluation Results

| Metric | Score |
|--------|-------|
| Faithfulness | 0.88 |
| Answer Relevancy | 0.90 |
| Context Precision | 0.84 |
| Context Recall | 0.81 |

*Evaluated on 20 held-out MedQuAD samples.*

## 🏗️ Architecture

```
User Query
    │
    ▼
Azure Content Safety (input guardrail)
    │
    ▼
Function Calling Router
    ├── search_knowledge_base → Azure AI Search
    ├── recommend_specialist  → rule-based specialist map
    └── emergency_redirect    → call 000 / emergency services
    │
    ▼ (if search_knowledge_base)
Azure AI Search — Hybrid Retrieval
    ├── BM25 keyword search
    ├── Vector similarity (text-embedding-ada-002)
    └── Semantic reranking
    │
    ▼
Azure OpenAI GPT-4o-mini (grounded generation)
    │
    ▼
Azure Content Safety (output guardrail)
    │
    ▼
FastAPI → Streamlit UI
```

## 🛠️ Azure Services

| Service | Purpose |
|---------|---------|
| Azure OpenAI (GPT-4o-mini) | Chat completions + function calling |
| Azure OpenAI (text-embedding-ada-002) | Embeddings |
| Azure AI Search | Hybrid vector + keyword + semantic retrieval |
| Azure Content Safety | Input/output guardrails |
| Azure App Service | Hosting |

## 📦 Dataset

**[keivalya/MedQuad-MedicalQnADataset](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)**
- 47,457 medical Q&A pairs from the NIH
- Covers: diseases, symptoms, treatments, medications, genetics
- Sources: GARD, MedlinePlus, ClinicalTrials.gov

## 🚀 Quick Start

```bash
git clone https://github.com/Kirthika-Srinivasan/medical-rag-assistant
cd medical-rag-assistant
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.sample .env  # fill in your Azure credentials

# Ingest data (run once)
python ingest/ingest.py

# Start API
uvicorn app.main:app --reload

# Start UI (new terminal)
streamlit run frontend/streamlit_app.py

# Run evaluation
python app/evaluate.py
```