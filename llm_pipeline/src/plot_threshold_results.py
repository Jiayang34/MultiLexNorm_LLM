import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(path):
    results = []
    with path.open(encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                results.append(json.loads(line))
    return results


def plot_entropy_metric(results, output_path, metric, ylabel, title):
    grouped = defaultdict(list)
    for result in results:
        grouped[result["detector_threshold"]].append(result)

    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    markers = ["o", "s", "^", "D", "X"]
    grouped_rows = sorted(grouped.items())

    plt.figure(figsize=(11, 7))
    for index, (detector_threshold, rows) in enumerate(grouped_rows):
        rows.sort(key=lambda row: row["entropy_threshold"])
        plt.plot(
            [row["entropy_threshold"] for row in rows],
            [row["total"][metric] for row in rows],
            linestyle=line_styles[index % len(line_styles)],
            marker=markers[index % len(markers)],
            markerfacecolor="none",
            markeredgewidth=1.8,
            linewidth=2,
            markersize=7 + index,
            zorder=len(grouped_rows) - index,
            label=f"Detector={detector_threshold:g}",
        )

    plt.xlabel("Entropy threshold")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    plt.tight_layout(rect=(0, 0, 0.82, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Wrote threshold plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot entropy threshold against final ERR and F1."
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--output",
        "--err-output",
        dest="err_output",
        type=Path,
        help="Output path for the ERR plot.",
    )
    parser.add_argument(
        "--f1-output",
        type=Path,
        help="Output path for the F1 plot.",
    )
    args = parser.parse_args()

    data_dir = Path("data") / f"{args.language}_thresholds"
    input_path = (
        args.input
        if args.input is not None
        else data_dir / f"threshold_results_{args.language}.jsonl"
    )
    err_output_path = (
        args.err_output
        if args.err_output is not None
        else data_dir / f"threshold_entropy_err_{args.language}.png"
    )
    f1_output_path = (
        args.f1_output
        if args.f1_output is not None
        else data_dir / f"threshold_entropy_f1_{args.language}.png"
    )

    results = load_results(input_path)
    plot_entropy_metric(
        results,
        err_output_path,
        metric="err",
        ylabel="Final ERR",
        title="Entropy Threshold vs Final ERR",
    )
    plot_entropy_metric(
        results,
        f1_output_path,
        metric="f1",
        ylabel="Final F1",
        title="Entropy Threshold vs Final F1",
    )


if __name__ == "__main__":
    main()
