"""
RAGAS evaluation harness — measures RAG quality against MedQuAD test questions.
Run: python app/evaluate.py
Saves scores to eval/eval_results.json for the README.
"""
import os, json
from datasets import load_dataset, Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall
)
from dotenv import load_dotenv
from rag import rag_query

load_dotenv()


def run_evaluation(n_samples: int = 20):
    print(f"🧪 Evaluating on {n_samples} MedQuAD samples...")

    dataset = load_dataset(
        "keivalya/MedQuad-MedicalQnADataset",
        split="train"
    )

    # Use last N samples as a held-out test set
    test_samples = list(dataset)[-n_samples:]

    questions, answers, contexts, ground_truths = [], [], [], []

    for row in test_samples:
        q = row.get("Question", "").strip()
        gt = row.get("Answer", "").strip()
        if not q or not gt or len(gt) < 20:
            continue

        result = rag_query(q)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append([d["content"] for d in result["retrieved_docs"]])
        ground_truths.append(gt[:1000])  # trim very long ground truths

    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    result = evaluate(
        eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    )

    scores = {
        "faithfulness": round(result["faithfulness"], 3),
        "answer_relevancy": round(result["answer_relevancy"], 3),
        "context_precision": round(result["context_precision"], 3),
        "context_recall": round(result["context_recall"], 3),
        "n_samples": len(questions),
        "dataset": "MedQuAD (NIH)"
    }

    os.makedirs("eval", exist_ok=True)
    with open("eval/eval_results.json", "w") as f:
        json.dump(scores, f, indent=2)

    print("\n📊 RAGAS Results:")
    for k, v in scores.items():
        print(f"   {k}: {v}")

    return scores


if __name__ == "__main__":
    run_evaluation()