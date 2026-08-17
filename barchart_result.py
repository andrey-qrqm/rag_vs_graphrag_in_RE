import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Adjust these to your actual file paths / labels
INPUT_FILES = {
    "VSR": "evaluation_results/blitdraft/3b/human_eval/vsr_multihop_HE.csv",
    "NEO4J GraphRag": "evaluation_results/blitdraft/3b/human_eval/neo4j_multihop_HE.csv",
    "Local GraphRag": "evaluation_results/blitdraft/3b/human_eval/local_graphrag_multihop_HE.csv",
}

OUTPUT_CHART = "visuals/combined_multihop_3b_blitdraft_HE.png"
METRICS = ["precision", "recall", "f1", "truthfullness"]
COLORS = ["#4C72B0", "#55A868", "#C44E52"]  # one color per file"


def main():
    # label -> [mean_precision, mean_recall, mean_f1]
    means = {}
    for label, path in INPUT_FILES.items():
        df = pd.read_csv(path)
        means[label] = [df[m].mean() for m in METRICS]

    labels = list(INPUT_FILES.keys())
    n_files = len(labels)
    n_metrics = len(METRICS)

    x = np.arange(n_metrics)          # one group per metric
    width = 0.8 / n_files             # bar width so groups fit nicely

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, label in enumerate(labels):
        offset = (i - (n_files - 1) / 2) * width
        values = means[label]
        bars = ax.bar(
            x + offset, values, width,
            label=label, color=COLORS[i % len(COLORS)]
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in METRICS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 / Truthfullness")
    ax.legend(title="Retrieval")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=200)
    plt.close()

    print(f"Saved chart to {OUTPUT_CHART}")


if __name__ == "__main__":
    main()