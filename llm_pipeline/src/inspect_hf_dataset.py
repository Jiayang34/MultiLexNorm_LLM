import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from src.config import (
    DATASET_NAME,
    DEV_PATH,
    DEV_RATIO,
    LANGUAGE,
    SEED,
    SPLIT_NAME,
)

# Print data examples for each dataset split:
# python -m src.inspect_hf_dataset

# Export and save dataset:
# python -m src.inspect_hf_dataset --export-dev-gold

PREVIEW_SIZE = 3


def inspect_split(name, split_dataset):
    print(f"\n[{name}]")
    print(f"features: {list(split_dataset.features.keys())}")
    print(f"num_rows: {split_dataset.num_rows}")

    # Iterate through examples
    for index, example in enumerate(split_dataset.select(range(PREVIEW_SIZE)), start=1):
        print(f"\nexample {index}:")
        print(json.dumps(example, ensure_ascii=False, indent=2))


def export_dev_gold(output_path):
    dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME)
    dataset = dataset.filter(lambda row: row.get("lang") == LANGUAGE)

    records = []
    skipped = 0
    for row in dataset:
        # Keep the same aligned-only examples used by detector data preparation.
        if len(row["raw"]) != len(row["norm"]):
            skipped += 1
            continue
        records.append(
            {
                "raw": [str(token) for token in row["raw"]],
                "norm": [str(token) for token in row["norm"]],
            }
        )

    # Match prepare_detector_data.py exactly, so sentence ids align with detector_dev_en.tsv.
    random.Random(SEED).shuffle(records)
    dev_size = int(len(records) * DEV_RATIO)
    dev_records = records[:dev_size]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as writer:
        for sent_id, record in enumerate(dev_records):
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

    print(f"Wrote {len(dev_records)} sentences to {output_path}")
    print(f"Skipped {skipped} unaligned examples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-dev-gold",
        type=Path,
        nargs="?",
        const=DEV_PATH,
        help="Write detector dev raw/norm gold JSONL and exit.",
    )
    args = parser.parse_args()

    if args.export_dev_gold:
        export_dev_gold(args.export_dev_gold)
        return

    dataset = load_dataset(DATASET_NAME)
    print(f"dataset: {DATASET_NAME}")
    print(f"splits: {list(dataset.keys())}")

    # Iterate through each split
    for split_name, split_dataset in dataset.items():
        inspect_split(split_name, split_dataset)


if __name__ == "__main__":
    main()
