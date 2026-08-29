import argparse
import json
from pathlib import Path

import src.apply_dictionary as dictionary_stage
import src.build_llm_prompts as prompt_stage
import src.evaluate_pipeline as evaluation_stage
import src.run_llm as llm_stage
from src.config import sanitize_model_name


LANGUAGES = ("de", "en", "hr", "id", "iden", "ja", "ko", "nl", "sl", "sr", "th", "vi")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOTS = {
    "original": PROJECT_ROOT / "baseline_lrz" / "data",
    "new": PROJECT_ROOT / "our_pipeline_lrz" / "data",
}
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / "data" / "val_threshold_runs"


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


def write_json(record, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        json.dump(record, writer, ensure_ascii=False, indent=2)
        writer.write("\n")


def build_input_paths(data_root, language):
    return {
        "detector": (
            data_root
            / "detector_output"
            / f"detector_{language}_val.confidence.out"
        ),
        "dictionary": data_root / f"dictionary_{language}.jsonl",
        "gold": data_root / f"gold_{language}_val.jsonl",
        "existing_prompts": data_root / f"llm_prompts_{language}_val.jsonl",
    }


def build_output_dir(
    output_root,
    detector,
    model,
    language,
    detector_threshold,
    entropy_threshold,
):
    run_name = (
        f"{language}_val_{detector_threshold:g}_{entropy_threshold:g}"
    )
    return (
        output_root
        / sanitize_model_name(model)
        / f"{detector}_detector"
        / run_name
    )


def build_output_paths(output_dir, language):
    return {
        "stage2": output_dir / f"table_applied_dictionary_{language}.jsonl",
        "candidates": output_dir / f"llm_candidates_{language}.jsonl",
        "prompts": output_dir / f"llm_prompts_{language}.jsonl",
        "llm_applied": (
            output_dir / f"llm_candidates_applied_llm_{language}.jsonl"
        ),
        "master": output_dir / f"table_applied_llm_{language}.jsonl",
        "summary": output_dir / f"evaluation_summary_{language}.json",
    }


def validate_inputs(paths):
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing validation input files:\n" + "\n".join(missing)
        )


def extract_few_shot_examples(prompt_text):
    lines = prompt_text.splitlines()
    examples = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if line == "Input:":
            break
        if line.startswith("Input") and ":" in line and index + 1 < len(lines):
            output_line = lines[index + 1]
            if output_line.startswith("Output") and ":" in output_line:
                examples.append(
                    {
                        "input": line.split(":", maxsplit=1)[1].strip(),
                        "output": json.loads(
                            output_line.split(":", maxsplit=1)[1].strip()
                        ),
                    }
                )
                index += 2
                continue
        index += 1

    if not examples:
        raise ValueError("Could not extract few-shot examples from existing prompt")
    return examples


def load_few_shot_examples(path):
    prompt_records = load_jsonl(path)
    if not prompt_records:
        raise ValueError(f"No prompt records found in {path}")
    return extract_few_shot_examples(prompt_records[0]["Prompt"])


def prepare_records(
    language,
    detector_threshold,
    entropy_threshold,
    input_paths,
    output_paths,
):
    dictionary_stage.LANGUAGE = language
    dictionary_stage.DETECTOR_THRESHOLD = detector_threshold
    dictionary_stage.ENTROPY_THRESHOLD = entropy_threshold
    prompt_stage.LANGUAGE = language

    dictionary = dictionary_stage.load_dictionary(input_paths["dictionary"])
    gold_sentences = load_jsonl(input_paths["gold"])
    detector_sentences = dictionary_stage.read_detector_output(
        input_paths["detector"]
    )
    master_records = dictionary_stage.build_master_table(
        detector_sentences,
        gold_sentences,
        dictionary,
    )
    candidates = [
        record
        for record in master_records
        if record["Source"] == "llm_pending"
    ]
    examples = load_few_shot_examples(input_paths["existing_prompts"])
    prompts = prompt_stage.build_prompts(
        master_records,
        candidates,
        examples,
    )

    write_jsonl(master_records, output_paths["stage2"])
    write_jsonl(candidates, output_paths["candidates"])
    write_jsonl(prompts, output_paths["prompts"])
    return master_records, candidates, prompts


def call_model(prompt, model, deepseek_client):
    if llm_stage.is_deepseek_model(model):
        return llm_stage.call_deepseek(deepseek_client, prompt, model)
    return llm_stage.call_ollama(prompt, model)


def run_model(
    prompts,
    candidates,
    master_records,
    model,
    output_paths,
    resume,
):
    candidates_by_token = llm_stage.index_by_token_id(candidates)
    outputs_by_token = {}

    if resume and output_paths["llm_applied"].is_file():
        existing = load_jsonl(output_paths["llm_applied"])
        outputs_by_token = llm_stage.index_by_token_id(existing)
        print(
            f"Resume: loaded {len(outputs_by_token)} predictions from "
            f"{output_paths['llm_applied']}"
        )

    deepseek_client = None
    if prompts and llm_stage.is_deepseek_model(model):
        deepseek_client = llm_stage.create_deepseek_client()

    for index, prompt_record in enumerate(prompts, start=1):
        token_id = prompt_record["Token_id"]
        if token_id in outputs_by_token:
            continue
        if token_id not in candidates_by_token:
            raise ValueError(f"Missing candidate for Token_id={token_id}")

        raw_output = call_model(
            prompt_record["Prompt"],
            model,
            deepseek_client,
        )
        try:
            prediction = llm_stage.parse_prediction(raw_output)
        except (TypeError, ValueError) as error:
            prediction = prompt_record["RAW"]
            print(
                f"[invalid output] Token_id={token_id} "
                f"fallback={prediction!r} error={error}"
            )

        output_record = dict(candidates_by_token[token_id])
        output_record["Replacement"] = prediction
        output_record["Source"] = model
        outputs_by_token[token_id] = output_record

        ordered_outputs = [
            outputs_by_token[prompt["Token_id"]]
            for prompt in prompts
            if prompt["Token_id"] in outputs_by_token
        ]
        write_jsonl(ordered_outputs, output_paths["llm_applied"])
        print(
            f"[{index}/{len(prompts)}] Token_id={token_id} "
            f"RAW={output_record['RAW']!r} -> {prediction!r}"
        )

    missing = [
        prompt["Token_id"]
        for prompt in prompts
        if prompt["Token_id"] not in outputs_by_token
    ]
    if missing:
        raise ValueError(f"Missing LLM predictions for Token_id values: {missing}")

    llm_records = [
        outputs_by_token[prompt["Token_id"]]
        for prompt in prompts
    ]
    merged_records = llm_stage.merge_llm_outputs(
        master_records,
        llm_records,
    )
    write_jsonl(llm_records, output_paths["llm_applied"])
    write_jsonl(merged_records, output_paths["master"])
    return merged_records


def run_language(args, data_root, language):
    detector_threshold, entropy_threshold = args.threshold
    input_paths = build_input_paths(data_root, language)
    validate_inputs(input_paths)
    output_dir = build_output_dir(
        args.output_root,
        args.detector,
        args.model,
        language,
        detector_threshold,
        entropy_threshold,
    )
    output_paths = build_output_paths(output_dir, language)

    print(
        f"\n=== detector={args.detector} language={language} "
        f"threshold={detector_threshold:g},{entropy_threshold:g} "
        f"model={args.model} ==="
    )
    print(f"Inputs: {data_root}")
    print(f"Outputs: {output_dir}")

    master_records, candidates, prompts = prepare_records(
        language,
        detector_threshold,
        entropy_threshold,
        input_paths,
        output_paths,
    )
    print(
        f"Prepared {len(master_records)} tokens and "
        f"{len(candidates)} LLM candidates"
    )

    if args.prepare_only:
        return None

    merged_records = run_model(
        prompts,
        candidates,
        master_records,
        args.model,
        output_paths,
        resume=not args.no_resume,
    )
    summary = evaluation_stage.summarize(merged_records)
    evaluation_stage.print_summary(summary)
    write_json(summary, output_paths["summary"])
    print(f"Wrote evaluation summary to {output_paths['summary']}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run validation data from the original or new detector with a "
            "configurable LLM and detector/dictionary thresholds."
        )
    )
    parser.add_argument(
        "--detector",
        choices=sorted(DEFAULT_DATA_ROOTS),
        required=True,
        help=(
            "original reads baseline_lrz/data; "
            "new reads our_pipeline_lrz/data."
        ),
    )
    parser.add_argument(
        "--model",
        choices=("deepseek-v4-pro", "qwen3.5:9b"),
        default="deepseek-v4-pro",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=LANGUAGES,
        default=list(LANGUAGES),
    )
    parser.add_argument(
        "--threshold",
        nargs=2,
        type=float,
        metavar=("DETECTOR", "ENTROPY"),
        default=(0.5, 0.5),
    )
    parser.add_argument(
        "--original-data",
        type=Path,
        default=DEFAULT_DATA_ROOTS["original"],
    )
    parser.add_argument(
        "--new-data",
        type=Path,
        default=DEFAULT_DATA_ROOTS["new"],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare table, candidates, and prompts without running the LLM.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore partial LLM output from an earlier run.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    data_root = (
        args.original_data
        if args.detector == "original"
        else args.new_data
    )

    for language in args.languages:
        run_language(args, data_root, language)


if __name__ == "__main__":
    main()
