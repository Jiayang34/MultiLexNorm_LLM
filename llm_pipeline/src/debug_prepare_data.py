from datasets import load_dataset

# For debugging : print some examples of produced data from prepare_data.py
# idx     raw     norm    label

from llm_pipeline.src.prepare_detector_data import (
    DATASET_NAME,
    LANGUAGE,
    NORM_LABEL,
    SPLIT_NAME,
    build_label_record,
    read_tokens,
)


PREVIEW_SIZE = 5


def print_alignment(index, raw_tokens, norm_tokens, labels):
    print(f"\nExample {index}")
    print(f"raw : {' '.join(raw_tokens)}")
    print(f"norm: {' '.join(norm_tokens)}")
    print("idx\traw\tnorm\tlabel")

    for token_index, (raw_token, norm_token, label) in enumerate(
        zip(raw_tokens, norm_tokens, labels), start=1
    ):
        print(f"{token_index}\t{raw_token}\t{norm_token}\t{label}")


def main():
    dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME)
    dataset = dataset.filter(lambda row: row.get("lang") == LANGUAGE)

    shown = 0
    for index, row in enumerate(dataset, start=1):
        record = build_label_record(row)
        if record is None or NORM_LABEL not in record["labels"]:
            continue

        raw_tokens, norm_tokens = read_tokens(row)
        print_alignment(index, raw_tokens, norm_tokens, record["labels"])
        shown += 1

        if shown == PREVIEW_SIZE:
            break

    print(f"\nShown: {shown}")


if __name__ == "__main__":
    main()
