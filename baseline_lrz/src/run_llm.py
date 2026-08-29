import argparse
import json
import re
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.config import (
    HF_MODEL_PATH,
    HF_MODEL_NAME,
    LLM_DTYPE,
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



def load_local_model(model_path, dtype_name):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    dtype = dtype_map[dtype_name]

    print(f"Loading model from: {model_path}")
    print(f"Model dtype: {dtype_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    model.eval()

    print(f"Model loaded on: {model.device}")
    return tokenizer, model
    
    
def call_transformers(prompt, tokenizer, model):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    prompt_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            use_cache=True,
        )

    new_tokens = generated_ids[0, prompt_length:]

    output = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )
    output = re.sub(
        r"^\s*<think>[\s\S]*?</think>\s*",
        "",
        output,
    )

    return output.strip()


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
def run_inference(prompt_records, candidate_records, processor, model, model_name):
    candidates_by_token = index_by_token_id(candidate_records)
    outputs = []
    invalid_outputs = 0

    for index, prompt_record in enumerate(prompt_records, start=1):
        token_id = prompt_record["Token_id"]
        if token_id not in candidates_by_token:
            raise ValueError(f"Missing candidate for Token_id={token_id}")

        # Call LLM and generate output
        raw_output = call_transformers(prompt_record["Prompt"], processor, model)
        try:
            prediction = parse_prediction(raw_output)
        except Exception:
            prediction = prompt_record["RAW"]
            invalid_outputs += 1

        output_record = dict(candidates_by_token[token_id])
        output_record["Replacement"] = prediction
        output_record["Source"] = model_name
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
    parser.add_argument("--model-path", type=Path, default=HF_MODEL_PATH)
    parser.add_argument("--model-name", default=HF_MODEL_NAME)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default=LLM_DTYPE)
    args = parser.parse_args()

    prompt_records = load_jsonl(args.prompts)
    master_records = load_jsonl(args.master_table)
    candidate_records = load_jsonl(args.candidates)
    processor, model = load_local_model(args.model_path, args.dtype)
    llm_records = run_inference(prompt_records, candidate_records, processor, model, args.model_name)
    merged_records = merge_llm_outputs(master_records, llm_records)

    write_jsonl(llm_records, args.output)
    write_jsonl(merged_records, args.master_output)

    print(f"Wrote {len(llm_records)} LLM-applied records to {args.output}")
    print(f"Wrote {len(merged_records)} Stage 3 master records to {args.master_output}")


if __name__ == "__main__":
    main()
