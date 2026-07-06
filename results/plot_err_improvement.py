import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent
QWEN_RESULTS = (
    RESULTS_DIR / "qwen35_final_new_detector_threshold_comparison.csv"
)
DEEPSEEK_RESULTS = RESULTS_DIR / "deepseek_val_base_newdetector_threshold.csv"
OUTPUT_PATH = RESULTS_DIR / "err_improvement_by_language.png"

LANGUAGE_NAMES = {
    "de": "DE",
    "en": "EN",
    "hr": "HR",
    "id": "ID",
    "iden": "ID-EN",
    "ja": "JA",
    "ko": "KO",
    "nl": "NL",
    "sl": "SL",
    "sr": "SR",
    "th": "TH",
    "vi": "VI",
}


def load_qwen_deltas():
    deltas = {}
    with QWEN_RESULTS.open(encoding="utf-8", newline="") as reader:
        for row in csv.DictReader(reader):
            if row["system"] != "new_detector_with_threshold":
                continue
            deltas[row["language"]] = (
                float(row["overall_final_delta_err"]) * 100
            )
    return deltas


def load_deepseek_deltas():
    deltas = {}
    with DEEPSEEK_RESULTS.open(encoding="utf-8", newline="") as reader:
        for row in csv.DictReader(reader):
            deltas[row["language"]] = 100 * (
                float(row["threshold_search_err"]) - float(row["base_err"])
            )
    return deltas


def annotate_points(ax, values, y_positions, color):
    for value, y_position in zip(values, y_positions):
        horizontal_offset = 5 if value >= 0 else -5
        alignment = "left" if value >= 0 else "right"
        ax.annotate(
            f"{value:+.2f}",
            (value, y_position),
            xytext=(horizontal_offset, 0),
            textcoords="offset points",
            va="center",
            ha=alignment,
            fontsize=10,
            color=color,
            fontweight="semibold",
        )


def main():
    qwen = load_qwen_deltas()
    deepseek = load_deepseek_deltas()

    if qwen.keys() != deepseek.keys():
        raise ValueError(
            "Qwen and DeepSeek results contain different languages: "
            f"Qwen={sorted(qwen)}, DeepSeek={sorted(deepseek)}"
        )

    languages = sorted(
        qwen,
        key=lambda language: (qwen[language] + deepseek[language]) / 2,
    )
    qwen_values = [qwen[language] for language in languages]
    deepseek_values = [deepseek[language] for language in languages]

    y_positions = list(range(len(languages)))
    qwen_y = [position + 0.14 for position in y_positions]
    deepseek_y = [position - 0.14 for position in y_positions]

    fig, ax = plt.subplots(figsize=(11, 7.5))

    for position, qwen_value, deepseek_value in zip(
        y_positions,
        qwen_values,
        deepseek_values,
    ):
        ax.plot(
            [qwen_value, deepseek_value],
            [position + 0.14, position - 0.14],
            color="#C8CDD3",
            linewidth=1.4,
            zorder=1,
        )

    ax.scatter(
        qwen_values,
        qwen_y,
        s=80,
        color="#0072B2",
        label="Qwen3.5-9B",
        zorder=3,
    )
    ax.scatter(
        deepseek_values,
        deepseek_y,
        s=80,
        color="#D55E00",
        label="DeepSeek-V4-Pro",
        zorder=3,
    )

    annotate_points(ax, qwen_values, qwen_y, "#005A8D")
    annotate_points(ax, deepseek_values, deepseek_y, "#A84800")

    ax.axvline(0, color="#333333", linewidth=1.4, linestyle="--", zorder=0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [LANGUAGE_NAMES.get(language, language.upper()) for language in languages],
        fontsize=12,
        fontweight="semibold",
    )
    ax.set_xlabel(
        "ERR improvement over baseline (percentage points)",
        fontsize=13,
        fontweight="semibold",
        labelpad=10,
    )
    ax.set_title(
        "Effect of Length-Aware Detection and Threshold Optimization",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0.5,
        1.01,
        "Final system minus original detector with default thresholds",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#555555",
    )

    all_values = qwen_values + deepseek_values
    ax.set_xlim(min(min(all_values) - 2.2, -3.5), max(all_values) + 2.8)
    ax.xaxis.grid(True, color="#E1E4E8", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=11)
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=11,
        ncol=2,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
