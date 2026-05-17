import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "weerayut/multilexnorm2026-dev-pub"
SPLIT_NAME = "train"
LANGUAGE = "en"
OUTPUT_PATH = Path(f"data/detector_{SPLIT_NAME}_{LANGUAGE}.jsonl")
MACHAMP_DIR = Path("data/machamp")
MACHAMP_TRAIN_PATH = MACHAMP_DIR / f"detector_train_{LANGUAGE}.tsv"
MACHAMP_DEV_PATH = MACHAMP_DIR / f"detector_dev_{LANGUAGE}.tsv"
DEV_RATIO = 0.1
SEED = 42
KEEP_LABEL = "O"
NORM_LABEL = "NORM"


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


# Label tokens by comparing raw with norm
def build_label_record(data):
    raw_tokens, norm_tokens = read_tokens(data)

    # Skip if 1->n or n->1
    if len(raw_tokens) != len(norm_tokens):
        return None

    # raw token == norm token  -> O
    # raw token != norm token  -> NORM
    labels = [
        KEEP_LABEL if raw_token == norm_token else NORM_LABEL
        for raw_token, norm_token in zip(raw_tokens, norm_tokens)
    ]
    return {"tokens": raw_tokens, "labels": labels}


# Write detector records in JSONL format
def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")


# Write detector records in MaChAmp TSV format
def write_machamp_tsv(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for record in records:
            for token, label in zip(record["tokens"], record["labels"]):
                writer.write(f"{token}\t{label}\n")
            writer.write("\n")


# Split detector records into train and dev subsets with a fixed seed
def split_records(records):
    shuffled = records.copy()
    random.Random(SEED).shuffle(shuffled)
    dev_size = int(len(shuffled) * DEV_RATIO)
    return shuffled[dev_size:], shuffled[:dev_size]


# Main function: iterate through dataset and label tokens
def convert_dataset(dataset_name, output_path):
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
    write_jsonl(records, output_path)
    write_machamp_tsv(train_records, MACHAMP_TRAIN_PATH)
    write_machamp_tsv(dev_records, MACHAMP_DEV_PATH)

    return {
        "total": total,
        "written": len(records),
        "skipped": skipped,
        "train": len(train_records),
        "dev": len(dev_records),
    }


def main():
    stats = convert_dataset(DATASET_NAME, OUTPUT_PATH)
    print(
        "Done: "
        f"{stats['written']} written, "
        f"{stats['skipped']} skipped, "
        f"{stats['total']} total. "
        f"Train: {stats['train']}, "
        f"Dev: {stats['dev']}. "
        f"Output: {OUTPUT_PATH}, {MACHAMP_DIR}"
    )


if __name__ == "__main__":
    main()
