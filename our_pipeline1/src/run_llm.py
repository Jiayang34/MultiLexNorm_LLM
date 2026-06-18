import argparse
import json
import re
from pathlib import Path

import requests

from src.config import (
    OLLAMA_MODEL,
    OLLAMA_URL,
    STAGE2_OUTPUT_PATH,
    STAGE3_LLM_APPLIED_PATH,
    STAGE3_LLM_CANDIDATES_PATH,
    STAGE3_MASTER_TABLE_PATH,
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



# Call Ollama chat API
def call_ollama(prompt, model, url):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 96},
    }
    response = requests.post(url, json=payload, timeout=600)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"].strip()


# Parse JSON output to string, check output format.
def load_first_json_value(raw_output):
    text = raw_output.strip()
    decoder = json.JSONDecoder()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        parsed, _ = decoder.raw_decode(text)
        return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*?\]", text)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not parse JSON prediction from: {raw_output!r}")


def parse_prediction(raw_output):
    parsed = load_first_json_value(raw_output)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected one-item JSON list, got: {raw_output!r}")
    prediction = None
    if len(parsed) >= 1:
        prediction = parsed[0]
    else:
        raise ValueError(f"Expected one-item JSON list, got an empty json: {raw_output!r}")
    if not isinstance(prediction, str):
        raise ValueError(f"Expected string prediction, got: {raw_output!r}")
    return prediction


# Index records by Token_id for prompt/candidate alignment.
def index_by_token_id(records):
    return {record["Token_id"]: record for record in records}


# Merge LLM-applied candidate records back into the master record table.
def merge_llm_outputs(master_records, llm_records):
    llm_by_token = index_by_token_id(llm_records)
    merged_records = []

    for record in master_records:
        token_id = record["Token_id"]
        if token_id in llm_by_token:
            merged_records.append(llm_by_token[token_id])
        else:
            merged_records.append(record)

    return merged_records


# Run LLM and return LLM candidates records
def run_inference(prompt_records, candidate_records, model, url):
    candidates_by_token = index_by_token_id(candidate_records)
    outputs = []
    invalid_outputs = 0

    for index, prompt_record in enumerate(prompt_records, start=1):
        token_id = prompt_record["Token_id"]
        if token_id not in candidates_by_token:
            raise ValueError(f"Missing candidate for Token_id={token_id}")

        # Call LLM and generate output
        raw_output = call_ollama(prompt_record["Prompt"], model, url)
        try:
            prediction = parse_prediction(raw_output)
        except Exception:
            prediction = prompt_record["RAW"]
            invalid_outputs += 1

        output_record = dict(candidates_by_token[token_id])
        output_record["Replacement"] = prediction
        output_record["Source"] = model
        outputs.append(output_record)

        print(
            f"[{index}/{len(prompt_records)}] "
            f"Token_id={token_id} RAW={output_record['RAW']!r} -> {prediction!r}"
        )

    print(f"Invalid outputs: {invalid_outputs}")
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM inference for normalization candidates."
    )
    parser.add_argument("--prompts", type=Path, default=STAGE3_LLM_PROMPTS_PATH)
    parser.add_argument("--master-table", type=Path, default=STAGE2_OUTPUT_PATH)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=STAGE3_LLM_CANDIDATES_PATH,
    )
    parser.add_argument("--output", type=Path, default=STAGE3_LLM_APPLIED_PATH)
    parser.add_argument("--master-output", type=Path, default=STAGE3_MASTER_TABLE_PATH)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--url", default=OLLAMA_URL)
    args = parser.parse_args()

    prompt_records = load_jsonl(args.prompts)
    master_records = load_jsonl(args.master_table)
    candidate_records = load_jsonl(args.candidates)
    llm_records = run_inference(prompt_records, candidate_records, args.model, args.url)
    merged_records = merge_llm_outputs(master_records, llm_records)

    write_jsonl(llm_records, args.output)
    write_jsonl(merged_records, args.master_output)

    print(f"Wrote {len(llm_records)} LLM-applied records to {args.output}")
    print(f"Wrote {len(merged_records)} Stage 3 master records to {args.master_output}")


if __name__ == "__main__":
    main()
