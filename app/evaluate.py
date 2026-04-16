"""
evaluate.py — RAGAS evaluation configured for Azure OpenAI
Run: python app/evaluate.py
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from datasets import load_dataset, Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Import rag_query from the same folder
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag import rag_query


def get_azure_llm():
    """Returns AzureChatOpenAI wrapped for RAGAS."""
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        temperature=0
    )
    return LangchainLLMWrapper(llm)


def get_azure_embeddings():
    """Returns AzureOpenAIEmbeddings wrapped for RAGAS."""
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    )
    return LangchainEmbeddingsWrapper(embeddings)


def run_evaluation(n_samples: int = 20):
    print(f"🧪 Running RAGAS evaluation on {n_samples} samples...")
    print(f"   Using Azure OpenAI: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"   Chat deployment:    {os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT')}")
    print(f"   Embed deployment:   {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')}")
    print()

    # Load test questions from MedQuAD
    dataset = load_dataset(
        "keivalya/MedQuad-MedicalQnADataset",
        split="train"
    )

    indexed_rows = list(dataset)[:300]
    test_samples = indexed_rows[-n_samples:]

    questions, answers, contexts, ground_truths = [], [], [], []
    skipped = 0

    for row in test_samples:
        q = row.get("Question", "").strip()
        gt = row.get("Answer", "").strip()

        if not q or not gt or len(gt) < 20:
            skipped += 1
            continue

        print(f"   Querying: {q[:60]}...")
        result = rag_query(q)

        # Only include if retrieval actually found something
        if not result.get("retrieved_docs"):
            skipped += 1
            print(f"   ⚠️ No docs retrieved — skipping")
            continue

        questions.append(q)
        answers.append(result["answer"])
        contexts.append([d["content"] for d in result["retrieved_docs"]])
        ground_truths.append(gt[:1000])  # trim very long ground truths

    print(f"\n   Evaluated: {len(questions)} samples (skipped {skipped})")

    if len(questions) < 3:
        print("❌ Not enough samples to evaluate — need at least 3")
        print("   Try increasing n_samples or checking your index")
        return

    # Build RAGAS dataset
    eval_dataset = Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts,
        "ground_truth": ground_truths
    })

    # Configure all metrics to use Azure OpenAI
    azure_llm = get_azure_llm()
    azure_embeddings = get_azure_embeddings()

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # Inject Azure into every metric
    for metric in metrics:
        metric.llm = azure_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = azure_embeddings

    print("\n📊 Running RAGAS scoring...")
    result = evaluate(
        eval_dataset,
        metrics=metrics,
        llm=azure_llm,
        embeddings=azure_embeddings,
        raise_exceptions=False   # don't crash on individual sample failures
    )

    scores = {
        "faithfulness":       round(float(result["faithfulness"]), 3),
        "answer_relevancy":   round(float(result["answer_relevancy"]), 3),
        "context_precision":  round(float(result["context_precision"]), 3),
        "context_recall":     round(float(result["context_recall"]), 3),
        "n_samples":          len(questions),
        "dataset":            "MedQuAD (NIH)"
    }

    # Save results
    os.makedirs(os.path.join(Path(__file__).parent.parent, "eval"), exist_ok=True)
    output_path = Path(__file__).resolve().parent.parent / "eval" / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    print("\n✅ RAGAS Evaluation Results:")
    print(f"   Faithfulness:      {scores['faithfulness']}")
    print(f"   Answer Relevancy:  {scores['answer_relevancy']}")
    print(f"   Context Precision: {scores['context_precision']}")
    print(f"   Context Recall:    {scores['context_recall']}")
    print(f"\n   Saved to: {output_path}")

    return scores


if __name__ == "__main__":
    run_evaluation(n_samples=20)