import argparse
import json
import os
from pathlib import Path

import requests
from openai import OpenAI

from src.config import (
    DATA_DIR,
    DETECTOR_THRESHOLD,
    ENTROPY_THRESHOLD,
    LANGUAGE,
    MODEL,
    build_run_data_dir,
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


def build_llm_paths(data_dir, language):
    return {
        "prompts": data_dir / f"llm_prompts_{language}.jsonl",
        "master_table": (
            data_dir / f"table_applied_dictionary_{language}.jsonl"
        ),
        "candidates": data_dir / f"llm_candidates_{language}.jsonl",
        "output": (
            data_dir / f"llm_candidates_applied_llm_{language}.jsonl"
        ),
        "master_output": data_dir / f"table_applied_llm_{language}.jsonl",
    }


# Call Ollama API
def call_ollama(prompt, model):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }
    response = requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"].strip()


# Create one reusable DeepSeek client
def create_deepseek_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. Run: "
            "export DEEPSEEK_API_KEY='your-api-key'"
        )

    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# Call DeepSeek
def call_deepseek(client, prompt, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You perform lexical normalization. "
                    "Follow the requested output format exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=0,
        max_tokens=64,
        extra_body={"thinking": {"type": "disabled"}},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("DeepSeek returned empty content")

    return content.strip()


# Parse JSON output to string, check output format
def parse_prediction(raw_output):
    cleaned_output = raw_output.strip()
    if cleaned_output.startswith("```") and cleaned_output.endswith("```"):
        lines = cleaned_output.splitlines()
        if len(lines) >= 3:
            cleaned_output = "\n".join(lines[1:-1]).strip()

    parsed = json.loads(cleaned_output)
    if not isinstance(parsed, list) or len(parsed) != 1:
        raise ValueError(f"Expected one-item JSON list, got: {raw_output!r}")
    prediction = parsed[0]
    if not isinstance(prediction, str):
        raise ValueError(f"Expected string prediction, got: {raw_output!r}")
    return prediction


# Index records by Token_id for prompt/candidate alignment.
def index_by_token_id(records):
    return {record["Token_id"]: record for record in records}


def is_deepseek_model(model):
    return model.lower().startswith("deepseek-")


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
def run_inference(prompt_records, candidate_records, model):
    # load LLM candidate list
    candidates_by_token = index_by_token_id(candidate_records)
    outputs = []
    invalid_outputs = 0
    deepseek_client = None

    if is_deepseek_model(model):
        deepseek_client = create_deepseek_client()

    for index, prompt_record in enumerate(prompt_records, start=1):
        token_id = prompt_record["Token_id"]
        if token_id not in candidates_by_token:
            raise ValueError(f"Missing candidate for Token_id={token_id}")

        # Call LLM
        if is_deepseek_model(model):
            raw_output = call_deepseek(
                deepseek_client,
                prompt_record["Prompt"],
                model,
            )
        else:
            raw_output = call_ollama(prompt_record["Prompt"], model)

        try:
            prediction = parse_prediction(raw_output)
        except (TypeError, ValueError) as error:
            prediction = prompt_record["RAW"]
            invalid_outputs += 1
            print(
                f"[invalid output] Token_id={token_id} "
                f"fallback={prediction!r} error={error}"
            )

        output_record = dict(candidates_by_token[token_id])
        output_record["Replacement"] = prediction
        output_record["Source"] = model
        outputs.append(output_record)

        print(
            f"[{index}/{len(prompt_records)}] "
            f"Token_id={token_id} "
            f"RAW={output_record['RAW']!r} -> {prediction!r}"
        )

    print(f"Invalid outputs replaced with RAW: {invalid_outputs}")
    return outputs


# Reuse cached LLM replacements for the current candidate subset
def load_cached_predictions(candidate_records, cache_records):
    cache_by_token = index_by_token_id(cache_records)
    outputs = []

    for candidate in candidate_records:
        token_id = candidate["Token_id"]
        if token_id not in cache_by_token:
            raise ValueError(f"Missing cached LLM output for Token_id={token_id}")

        cached_record = cache_by_token[token_id]
        output_record = dict(candidate)
        output_record["Replacement"] = cached_record["Replacement"]
        output_record["Source"] = cached_record["Source"]
        outputs.append(output_record)

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM inference for normalization candidates."
    )
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    if os.getenv("PIPELINE_DATA_DIR") or args.model == MODEL:
        data_dir = DATA_DIR
    else:
        data_dir = build_run_data_dir(
            LANGUAGE,
            args.model,
            DETECTOR_THRESHOLD,
            ENTROPY_THRESHOLD,
        )
    paths = build_llm_paths(data_dir, LANGUAGE)

    master_records = load_jsonl(paths["master_table"])
    candidate_records = load_jsonl(paths["candidates"])
    cache_path_value = os.getenv("PIPELINE_LLM_CACHE_PATH")
    if cache_path_value:
        cache_path = Path(cache_path_value)
        cache_records = load_jsonl(cache_path)
        llm_records = load_cached_predictions(candidate_records, cache_records)
        print(f"Reused {len(llm_records)} LLM outputs from {cache_path}")
    else:
        prompt_records = load_jsonl(paths["prompts"])
        llm_records = run_inference(
            prompt_records,
            candidate_records,
            args.model,
        )

        cache_output_value = os.getenv("PIPELINE_LLM_CACHE_OUTPUT_PATH")
        if cache_output_value:
            cache_output_path = Path(cache_output_value)
            write_jsonl(llm_records, cache_output_path)
            print(f"Wrote LLM cache to {cache_output_path}")

    merged_records = merge_llm_outputs(master_records, llm_records)

    write_jsonl(llm_records, paths["output"])
    write_jsonl(merged_records, paths["master_output"])

    print(f"Wrote {len(llm_records)} LLM-applied records to {paths['output']}")
    print(
        f"Wrote {len(merged_records)} Stage 3 master records "
        f"to {paths['master_output']}"
    )


if __name__ == "__main__":
    main()
