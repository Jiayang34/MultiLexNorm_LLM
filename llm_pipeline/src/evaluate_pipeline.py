import argparse
import json
from pathlib import Path

from src.config import (
    DATA_DIR,
    LANGUAGE,
    NORM_LABEL,
)

NON_LLM_SOURCES = {"keep", "dictionary", "llm_pending"}


# Read token-level master table.
def load_jsonl(path):
    records = []
    with path.open(encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                records.append(json.loads(line))
    return records


# Write evaluation summary for experiment tracking.
def write_json(record, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        json.dump(record, writer, ensure_ascii=False, indent=2)
        writer.write("\n")


# Treat model-name sources as LLM outputs.
def is_llm_source(source):
    return source not in NON_LLM_SOURCES


# Avoid division by zero
def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


# Convert final normalization records into evaluation labels
def build_normalization_outcomes(records):
    outcomes = []
    for record in records:
        gold_positive = record["RAW"] != record["Gold_NORM"]
        prediction_correct = record["Replacement"] == record["Gold_NORM"]
        outcomes.append((gold_positive, prediction_correct))
    return outcomes


# Convert detector labels into evaluation labels
def build_detector_outcomes(records):
    outcomes = []
    for record in records:
        gold_positive = record["RAW"] != record["Gold_NORM"]
        predicted_positive = record["Detector_label"] == NORM_LABEL
        prediction_correct = predicted_positive == gold_positive
        outcomes.append((gold_positive, prediction_correct))
    return outcomes


# Compute TP/FP/TN/FN
def compute_confusion(outcomes):
    confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    for gold_positive, prediction_correct in outcomes:
        if gold_positive and prediction_correct:
            confusion["TP"] += 1
        elif gold_positive:
            confusion["FN"] += 1
        elif prediction_correct:
            confusion["TN"] += 1
        else:
            confusion["FP"] += 1

    return confusion


# Compute precision for the positive class.
def compute_precision(confusion):
    tp = confusion["TP"]
    fp = confusion["FP"]
    return safe_divide(tp, tp + fp)


# Compute recall for the positive class.
def compute_recall(confusion):
    tp = confusion["TP"]
    fn = confusion["FN"]
    return safe_divide(tp, tp + fn)


# Compute F1 from precision and recall.
def compute_f1(precision, recall):
    return safe_divide(2 * precision * recall, precision + recall)


# Compute ERR using the official MultiLexNorm formula.
def compute_err(confusion):
    total = sum(confusion.values())
    changed = confusion["TP"] + confusion["FN"]
    correct = confusion["TP"] + confusion["TN"]

    accuracy = safe_divide(correct, total)
    lai = safe_divide(total - changed, total)
    return safe_divide(accuracy - lai, 1 - lai)


# Package confusion counts with metric scores.
def evaluate_outcomes(outcomes):
    confusion = compute_confusion(outcomes)
    precision = compute_precision(confusion)
    recall = compute_recall(confusion)

    return {
        "count": len(outcomes),
        "confusion": confusion,
        "scores": {
            "precision": precision,
            "recall": recall,
            "f1": compute_f1(precision, recall),
            "err": compute_err(confusion),
        },
    }


# Build overall and per-source evaluation summary.
def summarize(final_records):
    dictionary_source_records = [
        record for record in final_records if record["Source"] == "dictionary"
    ]
    llm_source_records = [
        record for record in final_records if is_llm_source(record["Source"])
    ]

    summary = {
        "counts": {
            "total_tokens": len(final_records),
            "dictionary_replaced": len(dictionary_source_records),
            "llm_applied": len(llm_source_records),
            "llm_replaced": sum(
                record["Replacement"] != record["RAW"]
                for record in llm_source_records
            ),
        },
        "overall_final": evaluate_outcomes(build_normalization_outcomes(final_records)),
        "detector": evaluate_outcomes(build_detector_outcomes(final_records)),
        "dictionary_source": evaluate_outcomes(
            build_normalization_outcomes(dictionary_source_records)
        ),
        "llm_source": evaluate_outcomes(
            build_normalization_outcomes(llm_source_records)
        ),
    }

    return summary


# Print compact terminal summary.
def print_summary(summary):
    counts = summary["counts"]
    print(f"Total tokens: {counts['total_tokens']}")
    print(f"Dictionary replaced: {counts['dictionary_replaced']}")
    print(f"LLM applied: {counts['llm_applied']}")
    print(f"LLM replaced: {counts['llm_replaced']}")

    for name in [
        "overall_final",
        "detector",
        "dictionary_source",
        "llm_source",
    ]:
        if name not in summary:
            continue
        result = summary[name]
        scores = result["scores"]
        confusion = result["confusion"]
        print(
            f"{name}: "
            f"ERR={scores['err']:.4f}, "
            f"F1={scores['f1']:.4f}, "
            f"P={scores['precision']:.4f}, "
            f"R={scores['recall']:.4f}, "
            f"TP={confusion['TP']}, "
            f"FP={confusion['FP']}, "
            f"FN={confusion['FN']}, "
            f"TN={confusion['TN']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate token-level MultiLexNorm pipeline outputs."
    )
    parser.add_argument("--language", default=LANGUAGE)
    parser.add_argument(
        "--final-table",
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    args = parser.parse_args()

    data_dir = (
        DATA_DIR
        if args.language == LANGUAGE
        else Path("data") / args.language
    )
    final_table = (
        args.final_table
        if args.final_table is not None
        else data_dir / f"table_applied_llm_{args.language}.jsonl"
    )
    output = (
        args.output
        if args.output is not None
        else data_dir / f"evaluation_summary_{args.language}.json"
    )

    final_records = load_jsonl(final_table)

    summary = summarize(final_records)
    print_summary(summary)
    write_json(summary, output)

    print(f"Wrote evaluation summary to {output}")


if __name__ == "__main__":
    main()
