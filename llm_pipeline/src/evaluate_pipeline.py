import argparse
import json
import os

from src.config import (
    DATA_DIR,
    DETECTOR_THRESHOLD,
    ENTROPY_THRESHOLD,
    LANGUAGE,
    MODEL,
    NORM_LABELS,
    build_run_data_dir,
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


# Compute normalization TP/FP/TN/FN using the convention from the paper.
#
# In particular, a token that needs normalization but is changed to the wrong
# new form is an FP. It is only an FN when the system leaves the raw token
# unchanged.
def compute_normalization_confusion(records):
    confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    for record in records:
        raw = record["RAW"]
        gold = record["Gold_NORM"]
        prediction = record["Replacement"]

        if raw != gold:
            if prediction == gold:
                confusion["TP"] += 1
            elif prediction == raw:
                confusion["FN"] += 1
            else:
                confusion["FP"] += 1
        elif prediction == gold:
            confusion["TN"] += 1
        else:
            confusion["FP"] += 1

    return confusion


# Convert detector labels into evaluation labels
def build_detector_outcomes(records):
    outcomes = []
    for record in records:
        gold_positive = record["RAW"] != record["Gold_NORM"]
        predicted_positive = record["Detector_label"] in NORM_LABELS
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


def compute_lai(confusion):
    total = sum(confusion.values())
    changed = confusion["TP"] + confusion["FN"]
    return safe_divide(total - changed, total)


# Compute ERR using the official MultiLexNorm formula.
def compute_err(confusion):
    total = sum(confusion.values())
    correct = confusion["TP"] + confusion["TN"]

    accuracy = safe_divide(correct, total)
    lai = compute_lai(confusion)
    return safe_divide(accuracy - lai, 1 - lai)


# Compute official normalization accuracy, LAI, and ERR directly from
# raw/gold/prediction values. 
def compute_normalization_base_scores(records):
    total = len(records)
    changed = sum(
        record["RAW"] != record["Gold_NORM"]
        for record in records
    )
    correct = sum(
        record["Replacement"] == record["Gold_NORM"]
        for record in records
    )

    accuracy = safe_divide(correct, total)
    lai = safe_divide(total - changed, total)
    err = safe_divide(accuracy - lai, 1 - lai)
    return accuracy, lai, err


# Package paper-style normalization counts with independently computed ERR.
def evaluate_normalization_records(records):
    confusion = compute_normalization_confusion(records)
    precision = compute_precision(confusion)
    recall = compute_recall(confusion)
    accuracy, lai, err = compute_normalization_base_scores(records)

    return {
        "count": len(records),
        "confusion": confusion,
        "scores": {
            "accuracy": accuracy,
            "lai": lai,
            "precision": precision,
            "recall": recall,
            "f1": compute_f1(precision, recall),
            "err": err,
        },
    }


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
            "lai": compute_lai(confusion),
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
        "overall_final": evaluate_normalization_records(final_records),
        "sentence_level": evaluate_sentence_outcomes(final_records),
        "detector": evaluate_outcomes(build_detector_outcomes(final_records)),
        "dictionary_source": evaluate_normalization_records(
            dictionary_source_records
        ),
        "llm_source": evaluate_normalization_records(llm_source_records),
    }

    return summary


# Print compact terminal summary.
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
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--detector-threshold",
        type=float,
        default=DETECTOR_THRESHOLD,
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=ENTROPY_THRESHOLD,
    )
    args = parser.parse_args()

    data_dir = (
        DATA_DIR
        if os.getenv("PIPELINE_DATA_DIR")
        else build_run_data_dir(
            args.language,
            args.model,
            args.detector_threshold,
            args.entropy_threshold,
        )
    )
    final_table = data_dir / f"table_applied_llm_{args.language}.jsonl"
    output = data_dir / f"evaluation_summary_{args.language}.json"

    final_records = load_jsonl(final_table)

    summary = summarize(final_records)
    print_summary(summary)
    write_json(summary, output)

    print(f"Wrote evaluation summary to {output}")


if __name__ == "__main__":
    main()
