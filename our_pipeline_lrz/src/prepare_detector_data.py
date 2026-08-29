import json
import random

from datasets import load_dataset
from tqdm import tqdm

from src.config import (
    DATASET_NAME,
    DEV_RATIO,
    DETECTOR_TRAIN_DIR,
    DEV_PATH,
    KEEP_LABEL,
    LANGUAGE,
    MACHAMP_DATASET_CONFIG_PATH,
    MACHAMP_DATASET_NAME,
    MACHAMP_DEV_PATH,
    MACHAMP_TRAIN_PATH,
    NORM_1WORD_LABEL,
    NORM_2WORD_LABEL,
    NORM_3PLUS_LABEL,
    NORM_LABELS,
    SEED,
    SPLIT_NAME,
)

# Read raw and normalized token lists from one dataset row
def read_tokens(data):
    raw = data.get("raw")
    norm = data.get("norm")
    if raw is None or norm is None:
        raise KeyError("Each example must contain raw and norm")
    if not isinstance(raw, list):
        raise TypeError(f"Expected raw token list, got {type(raw).__name__}")
    if not isinstance(norm, list):
        raise TypeError(f"Expected norm token list, got {type(norm).__name__}")
    return [str(token) for token in raw], [str(token) for token in norm]


def build_norm_label(norm_token):
    word_count = len(str(norm_token).split())
    if word_count <= 1:
        # e.g. "u" -> "you"
        return NORM_1WORD_LABEL
    if word_count == 2:
        # e.g. "could've" -> "could have"
        return NORM_2WORD_LABEL
    # 1 word -> multiple words(>=3)
    return NORM_3PLUS_LABEL


# Label tokens by comparing raw with norm
def build_label_record(data):
    raw_tokens, norm_tokens = read_tokens(data)

    # Defensive check: MFR counts require aligned raw/norm lists e.g. "wanna"->"want to"
    if len(raw_tokens) != len(norm_tokens):
        return None

    # raw token == norm token -> O
    # raw token != norm token -> NORM_LABELS = {NORM_1WORD_LABEL, NORM_2WORD_LABEL, NORM_3PLUS_LABEL}
    labels = [
        KEEP_LABEL if raw_token == norm_token else build_norm_label(norm_token)
        for raw_token, norm_token in zip(raw_tokens, norm_tokens)
    ]
    return {
        "tokens": raw_tokens,
        "labels": labels,
        "raw": raw_tokens,
        "norm": norm_tokens,
    }



# Write detector records in MaChAmp TSV format
def write_machamp_tsv(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for record in records:
            for token, label in zip(record["tokens"], record["labels"]):
                writer.write(f"{token}\t{label}\n")
            writer.write("\n")


def write_raw_norm_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for sent_id, record in enumerate(records):
            writer.write(
                json.dumps(
                    {
                        "Sentence_id": sent_id,
                        "raw": record["raw"],
                        "norm": record["norm"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


# Write the language-specific MaChAmp dataset config.
def write_machamp_dataset_config(path):
    config = {
        MACHAMP_DATASET_NAME: {
            "train_data_path": str(MACHAMP_TRAIN_PATH),
            "dev_data_path": str(MACHAMP_DEV_PATH),
            "word_idx": 0,
            "tasks": {
                "norm_detect": {
                    "task_type": "seq",
                    "column_idx": 1,
                    "metric": "f1_macro",
                    "additional_metrics": ["accuracy", "f1_micro"],
                }
            },
        }
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        json.dump(config, writer, ensure_ascii=False, indent=2)
        writer.write("\n")


# Shuffle records with a fixed seed, split into train and dev subsets by a ratio
def split_records(records):
    shuffled = records.copy()
    random.Random(SEED).shuffle(shuffled)
    dev_size = int(len(shuffled) * DEV_RATIO)
    return shuffled[dev_size:], shuffled[:dev_size]


# Main function: iterate through dataset and label tokens
def convert_dataset(dataset_name):
    total = 0
    skipped = 0
    records = []

    dataset = load_dataset(dataset_name, split=SPLIT_NAME)
    dataset = dataset.filter(lambda row: row.get("lang") == LANGUAGE)

    for row in tqdm(dataset, desc="Preparing detector data"):
        total += 1
        record = build_label_record(row)
        if record is None:
            skipped += 1
            continue
        records.append(record)

    train_records, dev_records = split_records(records)
    write_machamp_tsv(train_records, MACHAMP_TRAIN_PATH)
    write_machamp_tsv(dev_records, MACHAMP_DEV_PATH)
    write_raw_norm_jsonl(dev_records, DEV_PATH)

    return {
        "total": total,
        "written": len(records),
        "skipped": skipped,
        "train": len(train_records),
        "dev": len(dev_records),
    }


def main():
    stats = convert_dataset(DATASET_NAME)
    write_machamp_dataset_config(MACHAMP_DATASET_CONFIG_PATH)
    print(
        "Done: "
        f"{stats['written']} written, "
        f"{stats['skipped']} skipped, "
        f"{stats['total']} total. "
        f"Train: {stats['train']}, "
        f"Dev: {stats['dev']}. "
        f"Output: {DETECTOR_TRAIN_DIR}. "
        f"Dev raw/norm: {DEV_PATH}. "
        f"Config: {MACHAMP_DATASET_CONFIG_PATH}"
    )


if __name__ == "__main__":
    main()
