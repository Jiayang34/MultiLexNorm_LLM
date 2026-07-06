import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.config import (
    DATASET_NAME,
    DEV_RATIO,
    KEEP_LABEL,
    LANGUAGE,
    MACHAMP_DATASET_NAME,
    MACHAMP_DEV_PATH,
    MACHAMP_TRAIN_PATH,
    NORM_LABEL,
    SEED,
    SPLIT_NAME,
)

VALIDATION_SPLIT_NAME = "validation"


# Build all language-specific output paths without relying on import-time config.
def build_language_paths(language):
    train_dir = Path("models") / "machamp" / "train_dev" / language
    validation_dir = Path("models") / "machamp" / "validation" / language
    return {
        "dataset_name": f"detector_{language}",
        "train_dir": train_dir,
        "train": train_dir / f"detector_train_{language}.tsv",
        "dev": train_dir / f"detector_dev_{language}.tsv",
        "validation": (
            validation_dir / f"detector_validation_{language}.tsv"
        ),
        "config": (
            Path("models")
            / "machamp"
            / "configs"
            / f"machamp_detector_{language}.json"
        ),
    }


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

    # Defensive check: MFR counts require aligned raw/norm lists e.g. "wanna"->"want to"
    if len(raw_tokens) != len(norm_tokens):
        return None

    # raw token == norm token  -> O
    # raw token != norm token  -> NORM
    labels = [
        KEEP_LABEL if raw_token == norm_token else NORM_LABEL
        for raw_token, norm_token in zip(raw_tokens, norm_tokens)
    ]
    return {"tokens": raw_tokens, "labels": labels}



# Write detector records in MaChAmp TSV format
def write_machamp_tsv(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for record in records:
            for token, label in zip(record["tokens"], record["labels"]):
                writer.write(f"{token}\t{label}\n")
            writer.write("\n")


# Build a sibling path with a custom experiment suffix.
def build_suffixed_path(path, suffix):
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


# Write the language-specific MaChAmp dataset config.
def write_machamp_dataset_config(
    path,
    train_path=MACHAMP_TRAIN_PATH,
    dev_path=MACHAMP_DEV_PATH,
    dataset_name=MACHAMP_DATASET_NAME,
):
    config = {
        dataset_name: {
            "train_data_path": str(train_path),
            "dev_data_path": str(dev_path),
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


# Read synthetic raw/norm JSONL and convert it to detector records.
def read_synthetic_records(path):
    records = []
    skipped = 0

    with path.open(encoding="utf-8") as reader:
        for line in reader:
            if not line.strip():
                continue
            record = build_label_record(json.loads(line))
            if record is None:
                skipped += 1
                continue
            records.append(record)

    return records, skipped


# Load one HF split and label tokens for MaChAmp.
def load_labeled_records(dataset_name, split_name, language=LANGUAGE):
    total = 0
    skipped = 0
    records = []

    dataset = load_dataset(dataset_name, split=split_name)
    dataset = dataset.filter(lambda row: row.get("lang") == language)

    for row in tqdm(dataset, desc=f"Preparing detector data ({split_name})"):
        total += 1
        record = build_label_record(row)
        if record is None:
            skipped += 1
            continue
        records.append(record)

    return records, total, skipped


# Main function: iterate through training data and prepare train/dev TSVs.
def convert_dataset(
    dataset_name,
    synthetic_data=None,
    run_name="synthetic",
    language=LANGUAGE,
):
    paths = build_language_paths(language)
    records, total, skipped = load_labeled_records(
        dataset_name,
        SPLIT_NAME,
        language,
    )
    train_records, dev_records = split_records(records)

    train_path = paths["train"]
    synthetic_count = 0
    synthetic_skipped = 0
    if synthetic_data:
        synthetic_records, synthetic_skipped = read_synthetic_records(synthetic_data)
        synthetic_count = len(synthetic_records)
        train_path = build_suffixed_path(paths["train"], run_name)
        train_records = synthetic_records

    write_machamp_tsv(train_records, train_path)
    write_machamp_tsv(dev_records, paths["dev"])

    return {
        "total": total,
        "written": len(records),
        "skipped": skipped,
        "train": len(train_records),
        "dev": len(dev_records),
        "synthetic": synthetic_count,
        "synthetic_skipped": synthetic_skipped,
        "train_path": train_path,
        "dev_path": paths["dev"],
        "train_dir": paths["train_dir"],
        "dataset_name": paths["dataset_name"],
        "config_path": paths["config"],
    }


def export_validation_dataset(dataset_name, language=LANGUAGE):
    paths = build_language_paths(language)
    records, total, skipped = load_labeled_records(
        dataset_name,
        VALIDATION_SPLIT_NAME,
        language,
    )
    write_machamp_tsv(records, paths["validation"])
    return {
        "total": total,
        "written": len(records),
        "skipped": skipped,
        "path": paths["validation"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MaChAmp detector data from train split and optional synthetic data."
    )
    parser.add_argument(
        "--language",
        default=LANGUAGE,
        help="Language code to prepare, e.g. en, de, or ja.",
    )
    parser.add_argument(
        "--synthetic-data",
        type=Path,
        default=None,
        help="Optional raw/norm JSONL used as synthetic-only pretraining data.",
    )
    parser.add_argument(
        "--run-name",
        default="synthetic",
        help="Suffix for synthetic train/config files.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Export labeled official validation data only; does not split train/dev "
            "or write a MaChAmp training config."
        ),
    )
    args = parser.parse_args()

    if args.eval_only:
        if args.synthetic_data:
            raise ValueError("--eval-only cannot be combined with --synthetic-data")
        stats = export_validation_dataset(DATASET_NAME, args.language)
        print(
            "Done: "
            f"{stats['written']} written, "
            f"{stats['skipped']} skipped, "
            f"{stats['total']} total. "
            f"Validation path: {stats['path']}"
        )
        return

    stats = convert_dataset(
        DATASET_NAME,
        synthetic_data=args.synthetic_data,
        run_name=args.run_name,
        language=args.language,
    )

    config_path = stats["config_path"]
    if args.synthetic_data:
        config_path = build_suffixed_path(config_path, args.run_name)
    write_machamp_dataset_config(
        config_path,
        train_path=stats["train_path"],
        dev_path=stats["dev_path"],
        dataset_name=stats["dataset_name"],
    )
    print(
        "Done: "
        f"{stats['written']} written, "
        f"{stats['skipped']} skipped, "
        f"{stats['total']} total. "
        f"Train: {stats['train']}, "
        f"Dev: {stats['dev']}. "
        f"Synthetic: {stats['synthetic']}, "
        f"Synthetic skipped: {stats['synthetic_skipped']}. "
        f"Train path: {stats['train_path']}. "
        f"Dev path: {stats['dev_path']}. "
        f"Output: {stats['train_dir']}. "
        f"Config: {config_path}"
    )


if __name__ == "__main__":
    main()
