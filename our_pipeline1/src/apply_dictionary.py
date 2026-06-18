import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from src.config import (
    DATASET_NAME,
    DETECTOR_CONFIDENCE_PATH,
    DETECTOR_THRESHOLD,
    DEV_RATIO,
    DICTIONARY_PATH,
    ENTROPY_THRESHOLD,
    GOLD_PATH,
    IS_VAL,
    KEEP_LABEL,
    LANGUAGE,
    NORM_LABELS,
    SEED,
    SPLIT_NAME,
    STAGE2_OUTPUT_PATH,
    STAGE3_LLM_CANDIDATES_PATH,
)


# Load dictionary
def load_dictionary(path):
    dictionary = {}
    with path.open(encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            raw = record["raw"]
            dictionary[raw] = {
                "replacement": record["replacement"],
                "entropy": record.get("entropy"),
            }
    return dictionary


# Load DEV golden data (RAW and NORM) directly from the HF dataset
def load_dev_gold():
    dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME)
    dataset = dataset.filter(lambda row: row.get("lang") == LANGUAGE)

    records = []
    for row in dataset:
        if len(row["raw"]) != len(row["norm"]):
            continue
        records.append(
            {
                "raw": [str(token) for token in row["raw"]],
                "norm": [str(token) for token in row["norm"]],
            }
        )

    random.Random(SEED).shuffle(records)
    dev_size = int(len(records) * DEV_RATIO)
    return [
        {
            "Sentence_id": sent_id,
            "raw": record["raw"],
            "norm": record["norm"],
        }
        for sent_id, record in enumerate(records[:dev_size])
    ]


def load_gold_jsonl(path):
    records = []
    with path.open(encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_gold_sentences():
    if IS_VAL:
        return load_gold_jsonl(GOLD_PATH)
    return load_dev_gold()


# Decide token label
# If confidence score > threshold then return NORM else 0
def parse_label_scores(score_field):
    scores = {}
    for part in score_field.split("|"):
        label, score = part.split("=", maxsplit=1)
        scores[label] = float(score)

    # select the norm label with maximal confidence
    norm_label, norm_conf = max(
        ((label, scores.get(label, 0.0)) for label in NORM_LABELS),
        key=lambda item: item[1],
    )
    if norm_conf >= DETECTOR_THRESHOLD:
        return norm_label, norm_conf
    keep_conf = scores.get(KEEP_LABEL, 0.0)
    return KEEP_LABEL, keep_conf


# Load and parse detector output
def read_detector_output(path):
    sentences = []
    current_sentence = []

    with path.open(encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            parts = stripped.split("\t")
            if len(parts) < 2:
                raise ValueError(f"Expected token and score at line {line_number}: {stripped}")

            raw = parts[0]
            label, confidence = parse_label_scores(parts[1])
            current_sentence.append(
                {
                    "raw": raw,
                    "label": label,
                    "confidence": confidence,
                }
            )

    if current_sentence:
        sentences.append(current_sentence)

    return sentences

# Validate detector output aligned with raw data
def validate_alignment(detector_sentences, gold_sentences):
    # Sentences number has to be same
    if len(detector_sentences) != len(gold_sentences):
        raise ValueError(
            "Sentence count mismatch: "
            f"detector={len(detector_sentences)}, gold={len(gold_sentences)}"
        )
    
    for sent_id, (detector_sentence, gold_sentence) in enumerate(
        zip(detector_sentences, gold_sentences)
    ):
        gold_raw = gold_sentence["raw"]
        # Tokens number has to be same
        if len(detector_sentence) != len(gold_raw):
            raise ValueError(
                f"Token count mismatch at sentence {sent_id}: "
                f"detector={len(detector_sentence)}, gold={len(gold_raw)}"
            )

        # Raw tokens have to be same
        for token_index, (detector_token, gold_token) in enumerate(
            zip(detector_sentence, gold_raw)
        ):
            if detector_token["raw"] != gold_token:
                raise ValueError(
                    f"Token mismatch at sentence {sent_id}, token {token_index}: "
                    f"detector={detector_token['raw']!r}, gold={gold_token!r}"
                )

# Apply dictionary replacement to detector output and build table
def build_master_table(detector_sentences, gold_sentences, dictionary):
    validate_alignment(detector_sentences, gold_sentences)

    records = []
    token_id = 0
    for sent_id, (detector_sentence, gold_sentence) in enumerate(
        zip(detector_sentences, gold_sentences)
    ):
        gold_norm = gold_sentence.get("norm")
        for token_index, detector_token in enumerate(detector_sentence):
            raw = detector_token["raw"]
            detector_label = detector_token["label"]
            detector_confidence = detector_token["confidence"]
            gold_token_norm = gold_norm[token_index] if gold_norm is not None else None
            dictionary_entropy = None

            # Replace NORM tokens only when the dictionary candidate is low entropy
            if detector_label in NORM_LABELS and raw in dictionary:
                dictionary_entry = dictionary[raw]
                dictionary_entropy = dictionary_entry["entropy"]
                if dictionary_entropy <= ENTROPY_THRESHOLD:
                    replacement = dictionary_entry["replacement"]
                    source = "dictionary"
                else:
                    replacement = raw
                    source = "llm_pending"
            # Rest tokens should be processed by LLM
            elif detector_label in NORM_LABELS:
                replacement = raw
                source = "llm_pending"
            # Keep
            elif detector_label == KEEP_LABEL:
                replacement = raw
                source = "keep"
            else:
                raise ValueError(
                    f"Unknown detector label {detector_label!r} "
                    f"at sentence {sent_id}, token {token_index}"
                )

            records.append(
                {
                    "Token_id": token_id,
                    "Sentence_id": sent_id,
                    "Token_index": token_index,
                    "RAW": raw,
                    "Gold_NORM": gold_token_norm,
                    "Detector_label": detector_label,
                    "Detector_confidence": detector_confidence,
                    "Replacement": replacement,
                    "Dictionary_entropy": dictionary_entropy,
                    "Source": source,
                }
            )
            token_id += 1

    return records


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")


# Print info
def summarize(records):
    source_counts = Counter(record["Source"] for record in records)
    detected_norm = sum(record["Detector_label"] in NORM_LABELS for record in records)
    dictionary_count = source_counts["dictionary"]
    coverage = dictionary_count / detected_norm if detected_norm else 0.0

    print(f"Total tokens: {len(records)}")
    print(f"Detected NORM: {detected_norm}")
    print(f"Keep: {source_counts['keep']}")
    print(f"Dictionary: {dictionary_count}")
    print(f"LLM pending: {source_counts['llm_pending']}")
    print(f"Dictionary coverage among detected NORM: {coverage:.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply dictionary lookup to detector confidence output."
    )
    parser.add_argument(
        "--detector-output",
        type=Path,
        default=DETECTOR_CONFIDENCE_PATH,
    )
    parser.add_argument("--dictionary", type=Path, default=DICTIONARY_PATH)
    parser.add_argument("--output", type=Path, default=STAGE2_OUTPUT_PATH)
    parser.add_argument(
        "--llm-candidates-output",
        type=Path,
        default=STAGE3_LLM_CANDIDATES_PATH,
    )
    args = parser.parse_args()

    # Inputs: dictionary, target raw/norm data, detector output
    dictionary = load_dictionary(args.dictionary)
    gold_sentences = load_gold_sentences()
    detector_sentences = read_detector_output(args.detector_output)

    # Apply dictionary replacement to detector output and build table
    records = build_master_table(detector_sentences, gold_sentences, dictionary)

    # Record llm candidates
    llm_candidates = [
        record for record in records if record["Source"] == "llm_pending"
    ]

    write_jsonl(records, args.output)
    write_jsonl(llm_candidates, args.llm_candidates_output)
    summarize(records)

    print(f"Wrote master table to {args.output}")
    print(f"Wrote LLM candidates to {args.llm_candidates_output}")


if __name__ == "__main__":
    main()
