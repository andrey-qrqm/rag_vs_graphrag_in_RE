from pathlib import Path

import pandas as pd
from bert_score import BERTScorer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

GOLDEN_FILE = "gold_answers/multi_hop_blitdraft_answers.txt"
GENERATED_FILE = "model_answers/llama32/3b/blitdraft/local_graphrag_multihop.txt"
OUTPUT_FILE = "evaluation_results/blitdraft/3b/local_graphrag_multihop.csv"

LANGUAGE = "en"

# For English, the default BERTScore model is typically
# RoBERTa-large. You can explicitly specify one if desired.
MODEL_TYPE = "roberta-base"

# Set to True if you want baseline-rescaled scores.
RESCALE_WITH_BASELINE = True


# ---------------------------------------------------------
# Your existing parser
# ---------------------------------------------------------

def parse_txt_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [{"content": line.rstrip("\n"), "source": path} for line in f]


def parse_csv_file(filepath: str) -> list[str]:
    text = Path(filepath).read_text(encoding="utf-8")
    blocks = [
        block.strip()
        for block in text.split("<END>")
        if block.strip()
    ]
    return blocks

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_bertscore(
    generated_answers: list[str],
    golden_answers: list[str],
):
    golden_texts = [g["content"] for g in golden_answers]

    if len(generated_answers) != len(golden_texts):
        raise ValueError(
            f"Number of generated answers ({len(generated_answers)}) "
            f"does not match number of golden answers ({len(golden_texts)})."
        )

    scorer = BERTScorer(
        model_type=MODEL_TYPE,
        lang=LANGUAGE,
        rescale_with_baseline=RESCALE_WITH_BASELINE,
    )

    precision, recall, f1 = scorer.score(
        generated_answers,
        golden_texts,
    )

    # Convert PyTorch tensors -> Python floats
    precision = precision.tolist()
    recall = recall.tolist()
    f1 = f1.tolist()

    return precision, recall, f1


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    golden_answers = parse_txt_file(GOLDEN_FILE)
    print(len(golden_answers), "golden answers loaded from", GOLDEN_FILE)
    generated_answers = parse_csv_file(GENERATED_FILE)
    print(len(generated_answers), "generated answers loaded from", GENERATED_FILE)

    precision, recall, f1 = evaluate_bertscore(
        generated_answers,
        golden_answers,
    )

    # Per-example results
    golden_texts = [g["content"] for g in golden_answers]
    results = pd.DataFrame({
        "golden": golden_texts,
        "generated": generated_answers,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

    results.to_csv(OUTPUT_FILE, index=False)

    # Aggregate results
    print("\nBERTScore results")
    print("-----------------")
    print(f"Examples: {len(results)}")
    print(f"Precision: {results['precision'].mean():.4f}")
    print(f"Recall:    {results['recall'].mean():.4f}")
    print(f"F1:        {results['f1'].mean():.4f}")

    print(f"\nPer-example results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()