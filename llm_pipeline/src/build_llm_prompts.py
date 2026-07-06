import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from src.config import (
    DATASET_NAME,
    DEV_RATIO,
    LANGUAGE,
    NORM_1WORD_LABEL,
    NORM_2WORD_LABEL,
    NORM_3PLUS_LABEL,
    NUM_LLM_SHOTS,
    SEED,
    SPLIT_NAME,
    STAGE2_OUTPUT_PATH,
    STAGE3_LLM_CANDIDATES_PATH,
    STAGE3_LLM_PROMPTS_PATH,
)


def load_jsonl(path):
    records = []
    with path.open(encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")


# Group token records by sentence
def group_by_sentence(records):
    sentences = defaultdict(list)
    for record in records:
        sentences[record["Sentence_id"]].append(record)

    return {
        sent_id: sorted(sentence, key=lambda item: item["Token_index"])
        for sent_id, sentence in sentences.items()
    }


# Load train records
def load_train_records():

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
    return records[dev_size:]


# Sample few-shot examples from TRAIN records.
def sample_few_shot_examples(train_records, num_shots, seed):
    candidates = []
    for record in train_records:
        for token_index, (raw_token, norm_token) in enumerate(
            zip(record["raw"], record["norm"])
        ):
            if raw_token == norm_token:
                continue

            masked_tokens = record["raw"].copy()
            masked_tokens[token_index] = f"<MASK>{raw_token}</MASK>"
            candidates.append(
                {
                    "input": " ".join(masked_tokens),
                    "output": [norm_token],
                }
            )

    if num_shots > len(candidates):
        raise ValueError(
            f"Requested {num_shots} shots, but only found {len(candidates)} examples."
        )

    return random.Random(seed).sample(candidates, num_shots)


# Build the current sentence context and mark the target token with <MASK>.
def mask_sentence(sentence_records, candidate):
    token_index = candidate["Token_index"]
    masked_tokens = []
    for record in sentence_records:
        token = record["Replacement"]
        if record["Token_index"] == token_index:
            token = f"<MASK>{record['RAW']}</MASK>"
        masked_tokens.append(token)
    return " ".join(masked_tokens)


# Prompt: instruction, few-shot examples, and masked input sentence.
def format_prompt(masked_sentence, examples, norm_label=None):
    word_prompt = ""
    if norm_label == NORM_1WORD_LABEL:
        word_prompt = (
            "The normalized output should contain exactly 1 "
            "whitespace-separated word."
        )
    elif norm_label == NORM_2WORD_LABEL:
        word_prompt = (
            "The normalized output should contain exactly 2 "
            "whitespace-separated words and keep them in the same string, "
            'e.g. ["want to"].'
        )
    elif norm_label == NORM_3PLUS_LABEL:
        word_prompt = (
            "The normalized output should contain 3 or more "
            "whitespace-separated words and keep them in the same string, "
            'e.g. ["want to do"].'
        )

    lines = [
        "Standardize the word or phrase enclosed in <MASK> by converting it to "
        "its formal and grammatically correct form. Consider the sentence "
        "context, including punctuation, capitalization, and reduplication. "
        f"Preserve the original meaning with minimal edits. {word_prompt} If no normalization "
        "is needed, return the original word or phrase. Return only a JSON list "
        "with one string.",
        "",
        "In-context examples:",
    ]

    for index, example in enumerate(examples, start=1):
        lines.append(f"Input{index}: {example['input']}")
        lines.append(f"Output{index}: {json.dumps(example['output'], ensure_ascii=False)}")
        lines.append("")

    lines.append("Input:")
    lines.append(masked_sentence)
    lines.append("Output:")
    return "\n".join(lines)

# Build prompt record for each LLM candidate.
def build_prompts(master_records, candidates, examples):
    sentences = group_by_sentence(master_records)
    prompts = []

    for candidate in candidates:
        sentence = sentences[candidate["Sentence_id"]]
        masked_sentence = mask_sentence(sentence, candidate)
        prompts.append(
            {
                "Token_id": candidate["Token_id"],
                "Sentence_id": candidate["Sentence_id"],
                "Token_index": candidate["Token_index"],
                "RAW": candidate["RAW"],
                "Gold_NORM": candidate["Gold_NORM"],
                "Prompt": format_prompt(
                    masked_sentence,
                    examples,
                    candidate.get("Detector_label"),
                ),
            }
        )

    return prompts


def main():
    # Build prompt JSONL from the Stage 2 master table and LLM candidates.
    parser = argparse.ArgumentParser(
        description="Build LLM prompts for remaining normalization candidates."
    )
    parser.add_argument("--master-table", type=Path, default=STAGE2_OUTPUT_PATH)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=STAGE3_LLM_CANDIDATES_PATH,
    )
    parser.add_argument("--output", type=Path, default=STAGE3_LLM_PROMPTS_PATH)
    args = parser.parse_args()

    # Load data and sample few shot example
    train_records = load_train_records()
    examples = sample_few_shot_examples(train_records, NUM_LLM_SHOTS, SEED)

    master_records = load_jsonl(args.master_table)
    candidates = load_jsonl(args.candidates)
    # Build prompts
    prompts = build_prompts(master_records, candidates, examples)

    write_jsonl(prompts, args.output)

    print(f"Wrote {len(prompts)} prompts to {args.output}")
    print(f"Few-shot examples: {len(examples)}")


if __name__ == "__main__":
    main()
