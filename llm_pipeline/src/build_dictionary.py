import argparse
import json
import math
import random
from pathlib import Path

from datasets import load_dataset

from src.config import (
    DATASET_NAME,
    DEV_RATIO,
    DICTIONARY_PATH,
    LANGUAGE,
    SEED,
    SPLIT_NAME,
    ENTROPY_THRESHOLD
)


# Code from https://github.com/WeerayutBu/MultiLexNorm2026
# Count frequency, e.g. "vaek": {"væk": 3, "vaek": 1}
def counting(data):
    counts = {}
    for item in data:
        sentRaw = item["raw"]
        sentGold = item["norm"]
        for wordRaw, wordGold in zip(sentRaw, sentGold):
            if wordRaw not in counts:
                counts[wordRaw] = {}
            if wordGold not in counts[wordRaw]:
                counts[wordRaw][wordGold] = 0
            counts[wordRaw][wordGold] += 1
    return counts


# Code from https://github.com/WeerayutBu/MultiLexNorm2026
# Select most frequency
def mfr(input_sent, counts):
    predictions = []
    for word in input_sent:
        if word in counts:
            replacement = max(counts[word], key=counts[word].get)
        else:
            replacement = word
        predictions.append(replacement)
    return predictions


# Miller-Madow corrected entropy for one raw token's replacements
def miller_madow_entropy(replacement_counts):
    total = sum(replacement_counts.values())
    entropy = 0.0

    # H = - sum(p * log2(p))
    # lower entropy <-> more confident replacement
    for count in replacement_counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    # Miller-Madow correction: (K - 1) / (2N ln 2)
    correction = (len(replacement_counts) - 1) / (2 * total * math.log(2))

    return entropy + correction


# Load train data
def load_train_records():
    dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME)
    dataset = dataset.filter(lambda row: row.get("lang") == LANGUAGE)

    # Defensive check: MFR counts require aligned raw/norm lists e.g. "wanna"->"want to"
    records = []
    for row in dataset:
        if len(row["raw"]) != len(row["norm"]):
            continue
        records.append({"raw": row["raw"], "norm": row["norm"]})

    # Shuffle and split data
    random.Random(SEED).shuffle(records)
    dev_size = int(len(records) * DEV_RATIO)
    return records[dev_size:]


# Build look-up dictionary
def build_dictionary(records, entropy_threshold):
    dictionary = []

    counts = counting(records)

    for raw_token, replacement_counts in counts.items():
        # MFR
        replacement = mfr([raw_token], counts)[0]
        if replacement == raw_token:
            continue

        # Add only replacement with low entropy to dictionary
        entropy = miller_madow_entropy(replacement_counts)
        if entropy <= entropy_threshold:
            dictionary.append(
                {
                    "raw": raw_token,
                    "replacement": replacement,
                    "entropy": entropy,
                    "counts": replacement_counts,
                }
            )

    return dictionary


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build low-entropy MFR dictionary for Stage 2."
    )
    parser.add_argument("--output", type=Path, default=DICTIONARY_PATH)
    parser.add_argument("--entropy-threshold", type=float, default=ENTROPY_THRESHOLD)
    args = parser.parse_args()

    records = load_train_records()
    dictionary = build_dictionary(records, args.entropy_threshold)
    write_jsonl(dictionary, args.output)

    print(f"Wrote {len(dictionary)} entries to {args.output}")


if __name__ == "__main__":
    main()
