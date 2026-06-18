import argparse
import json
from pathlib import Path

from src.config import (
    EVALUATION_SUMMARY_PATH,
    NORM_LABEL,
    STAGE3_MASTER_TABLE_PATH,
)

NON_LLM_SOURCES = {"keep", "dictionary", "llm_pending"}


def load_jsonl(path):
    records = []
    with path.open(encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(record, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        json.dump(record, writer, ensure_ascii=False, indent=2)
        writer.write("\n")


def is_llm_source(source):
    return source not in NON_LLM_SOURCES


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def build_normalization_outcomes(records):
    outcomes = []

    for record in records:
        gold_positive = record["RAW"] != record["Gold_NORM"]
        prediction_correct = record["Replacement"] == record["Gold_NORM"]
        outcomes.append((gold_positive, prediction_correct))

    return outcomes


def build_detector_outcomes(records):
    outcomes = []

    for record in records:
        gold_positive = record["RAW"] != record["Gold_NORM"]
        predicted_positive = record["Detector_label"] == NORM_LABEL
        prediction_correct = predicted_positive == gold_positive
        outcomes.append((gold_positive, prediction_correct))

    return outcomes


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


def compute_precision(confusion):
    tp = confusion["TP"]
    fp = confusion["FP"]
    return safe_divide(tp, tp + fp)


def compute_recall(confusion):
    tp = confusion["TP"]
    fn = confusion["FN"]
    return safe_divide(tp, tp + fn)


def compute_f1(precision, recall):
    return safe_divide(2 * precision * recall, precision + recall)


def compute_lai(confusion):
    total = sum(confusion.values())
    changed = confusion["TP"] + confusion["FN"]
    return safe_divide(total - changed, total)


def compute_err(confusion):
    total = sum(confusion.values())
    correct = confusion["TP"] + confusion["TN"]

    accuracy = safe_divide(correct, total)
    lai = compute_lai(confusion)

    return safe_divide(accuracy - lai, 1 - lai)


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


def evaluate_sentence_outcomes(records):
    sentences = {}

    for record in records:
        sentence_id = record["Sentence_id"]
        gold_positive = record["RAW"] != record["Gold_NORM"]
        token_correct = record["Replacement"] == record["Gold_NORM"]

        if sentence_id not in sentences:
            sentences[sentence_id] = {
                "gold_positive": False,
                "prediction_correct": True,
            }

        # A sentence is positive if at least one token needs normalization.
        sentences[sentence_id]["gold_positive"] |= gold_positive

        # One incorrect token makes the entire sentence incorrect.
        sentences[sentence_id]["prediction_correct"] &= token_correct

    outcomes = [
        (
            sentence["gold_positive"],
            sentence["prediction_correct"],
        )
        for sentence in sentences.values()
    ]

    confusion = compute_confusion(outcomes)
    precision = compute_precision(confusion)
    recall = compute_recall(confusion)
    total = len(outcomes)
    correct = confusion["TP"] + confusion["TN"]

    return {
        "count": total,
        "correct": correct,
        "incorrect": total - correct,
        "confusion": confusion,
        "scores": {
            "accuracy": safe_divide(correct, total),
            "precision": precision,
            "recall": recall,
            "f1": compute_f1(precision, recall),
            "lai": compute_lai(confusion),
            "err": compute_err(confusion),
        },
    }


def summarize(final_records):
    dictionary_source_records = [
        record
        for record in final_records
        if record["Source"] == "dictionary"
    ]

    llm_source_records = [
        record
        for record in final_records
        if is_llm_source(record["Source"])
    ]

    return {
        "counts": {
            "total_tokens": len(final_records),
            "dictionary_replaced": len(dictionary_source_records),
            "llm_applied": len(llm_source_records),
            "llm_replaced": sum(
                record["Replacement"] != record["RAW"]
                for record in llm_source_records
            ),
        },
        "overall_final": evaluate_outcomes(
            build_normalization_outcomes(final_records)
        ),
        "sentence_level": evaluate_sentence_outcomes(final_records),
        "detector": evaluate_outcomes(
            build_detector_outcomes(final_records)
        ),
        "dictionary_source": evaluate_outcomes(
            build_normalization_outcomes(dictionary_source_records)
        ),
        "llm_source": evaluate_outcomes(
            build_normalization_outcomes(llm_source_records)
        ),
    }


def print_summary(summary):
    counts = summary["counts"]

    print(f"Total tokens: {counts['total_tokens']}")
    print(f"Dictionary replaced: {counts['dictionary_replaced']}")
    print(f"LLM applied: {counts['llm_applied']}")
    print(f"LLM replaced: {counts['llm_replaced']}")

    sentence_result = summary["sentence_level"]
    sentence_scores = sentence_result["scores"]
    sentence_confusion = sentence_result["confusion"]

    print(
        "sentence_level: "
        f"accuracy={sentence_scores['accuracy']:.4f}, "
        f"ERR={sentence_scores['err']:.4f}, "
        f"LAI={sentence_scores['lai']:.4f}, "
        f"F1={sentence_scores['f1']:.4f}, "
        f"P={sentence_scores['precision']:.4f}, "
        f"R={sentence_scores['recall']:.4f}, "
        f"correct={sentence_result['correct']}, "
        f"incorrect={sentence_result['incorrect']}, "
        f"total={sentence_result['count']}, "
        f"TP={sentence_confusion['TP']}, "
        f"FP={sentence_confusion['FP']}, "
        f"FN={sentence_confusion['FN']}, "
        f"TN={sentence_confusion['TN']}"
    )

    for name in [
        "overall_final",
        "detector",
        "dictionary_source",
        "llm_source",
    ]:
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
        description="Evaluate token-level and sentence-level pipeline outputs."
    )
    parser.add_argument(
        "--final-table",
        type=Path,
        default=STAGE3_MASTER_TABLE_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVALUATION_SUMMARY_PATH,
    )
    args = parser.parse_args()

    final_records = load_jsonl(args.final_table)
    summary = summarize(final_records)

    print_summary(summary)
    write_json(summary, args.output)

    print(f"Wrote evaluation summary to {args.output}")


if __name__ == "__main__":
    main()