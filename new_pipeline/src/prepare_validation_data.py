import json

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from src.config import (
    DATASET_NAME,
    GOLD_PATH,
    LANGUAGE,
    MACHAMP_VAL_PATH,
    VAL_SPLIT_NAME,
)
from src.prepare_detector_data import build_label_record, read_tokens, write_machamp_tsv


def write_gold_jsonl(records, path):
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


def convert_validation_dataset(dataset_name):
    total = 0
    skipped = 0
    detector_records = []
    gold_records = []
    local_path = "/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM/datasets/weerayut"
    dataset = load_from_disk(local_path)["validation"]
    dataset = dataset.filter(lambda x: x["lang"] == LANGUAGE)

    for row in tqdm(dataset, desc="Preparing validation data"):
        total += 1
        detector_record = build_label_record(row)
        if detector_record is None:
            skipped += 1
            continue

        raw_tokens, norm_tokens = read_tokens(row)
        detector_records.append(detector_record)
        gold_records.append({"raw": raw_tokens, "norm": norm_tokens})

    write_machamp_tsv(detector_records, MACHAMP_VAL_PATH)
    write_gold_jsonl(gold_records, GOLD_PATH)

    return {
        "total": total,
        "written": len(detector_records),
        "skipped": skipped,
    }


def main():
    stats = convert_validation_dataset(DATASET_NAME)
    print(
        "Done: "
        f"{stats['written']} written, "
        f"{stats['skipped']} skipped, "
        f"{stats['total']} total. "
        f"MaChAmp validation input: {MACHAMP_VAL_PATH}. "
        f"Gold data: {GOLD_PATH}"
    )


if __name__ == "__main__":
    main()
