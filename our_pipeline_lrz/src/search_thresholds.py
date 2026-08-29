import argparse
import json
from pathlib import Path

import src.apply_dictionary as dictionary_stage
import src.build_llm_prompts as prompt_stage
import src.evaluate_pipeline as evaluation_stage
import src.run_llm as llm_stage
from src.config import HF_MODEL_NAME, HF_MODEL_PATH, LLM_DTYPE, NUM_LLM_SHOTS, SEED


DEFAULT_LANGUAGES = [
    "de",
    "en",
    "hr",
    "id",
    "iden",
    "ja",
    "ko",
    "nl",
    "sl",
    "sr",
    "th",
    "vi",
]
DEFAULT_DETECTOR_THRESHOLDS = [0.1, 0.3, 0.5, 0.8, 1.0]
DEFAULT_ENTROPY_THRESHOLDS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4]
DEFAULT_THRESHOLD_PAIR = (0.5, 0.5)
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PIPELINE_ROOT / "data"
DEFAULT_TMP_ROOT = DEFAULT_DATA_ROOT / "thresold_search_tmp"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run in-memory dev threshold search for the length-aware pipeline."
        )
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=DEFAULT_LANGUAGES,
        help="Language codes to search.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tmp-root", type=Path, default=DEFAULT_TMP_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_ROOT / "threshold_search_results_qwen3.5_9b.jsonl",
    )
    parser.add_argument("--model-path", type=Path, default=HF_MODEL_PATH)
    parser.add_argument("--model-name", default=HF_MODEL_NAME)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default=LLM_DTYPE,
    )
    parser.add_argument(
        "--detector-thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_DETECTOR_THRESHOLDS,
    )
    parser.add_argument(
        "--entropy-thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_ENTROPY_THRESHOLDS,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse LLM cache and keep existing selected results.",
    )
    return parser.parse_args()


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


def append_jsonl(record, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as writer:
        writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_input_paths(data_root, language):
    return {
        "detector": (
            data_root
            / "detector_output"
            / f"detector_{language}.confidence.out"
        ),
        "dictionary": data_root / language / f"dictionary_{language}.jsonl",
        "gold": data_root / language / f"dev_raw_norm_{language}.jsonl",
    }


def build_tmp_paths(tmp_root, model_name, language):
    language_root = tmp_root / model_name / language
    return {
        "root": language_root,
        "widest_master": language_root / "widest_master.jsonl",
        "widest_candidates": language_root / "widest_candidates.jsonl",
        "widest_prompts": language_root / "widest_prompts.jsonl",
        "llm_cache": language_root / "llm_cache.jsonl",
        "grid_results": language_root / "grid_results.jsonl",
    }


def validate_inputs(paths):
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing threshold-search inputs:\n" + "\n".join(missing))


def build_combinations(detector_thresholds, entropy_thresholds):
    return [
        (detector_threshold, entropy_threshold)
        for detector_threshold in detector_thresholds
        for entropy_threshold in entropy_thresholds
    ]


def configure_thresholds(language, detector_threshold, entropy_threshold):
    dictionary_stage.LANGUAGE = language
    dictionary_stage.DETECTOR_THRESHOLD = detector_threshold
    dictionary_stage.ENTROPY_THRESHOLD = entropy_threshold
    prompt_stage.LANGUAGE = language


def build_records(language, input_paths, detector_threshold, entropy_threshold):
    configure_thresholds(language, detector_threshold, entropy_threshold)
    dictionary = dictionary_stage.load_dictionary(input_paths["dictionary"])
    gold_sentences = load_jsonl(input_paths["gold"])
    detector_sentences = dictionary_stage.read_detector_output(input_paths["detector"])
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
    return master_records, candidates


def load_prompt_examples(language):
    prompt_stage.LANGUAGE = language
    train_records = prompt_stage.load_train_records()
    return prompt_stage.sample_few_shot_examples(
        train_records,
        NUM_LLM_SHOTS,
        SEED,
    )


def load_llm_cache(path):
    if not path.is_file():
        return {}
    return llm_stage.index_by_token_id(load_jsonl(path))


def ensure_llm_cache(
    language,
    input_paths,
    tmp_paths,
    detector_threshold,
    entropy_threshold,
    model_name,
    model_state,
    model_path,
    dtype,
    resume,
):
    cached_outputs = load_llm_cache(tmp_paths["llm_cache"]) if resume else {}
    if cached_outputs:
        print(
            f"[cache] {language}: loaded {len(cached_outputs)} LLM outputs "
            f"from {tmp_paths['llm_cache']}"
        )
        return cached_outputs, model_state

    master_records, candidates = build_records(
        language,
        input_paths,
        detector_threshold,
        entropy_threshold,
    )
    examples = load_prompt_examples(language)
    prompts = prompt_stage.build_prompts(master_records, candidates, examples)

    write_jsonl(master_records, tmp_paths["widest_master"])
    write_jsonl(candidates, tmp_paths["widest_candidates"])
    write_jsonl(prompts, tmp_paths["widest_prompts"])

    if prompts:
        if model_state is None:
            processor, model = llm_stage.load_local_model(model_path, dtype)
            model_state = (processor, model)
        processor, model = model_state
        llm_records = llm_stage.run_inference(
            prompts,
            candidates,
            processor,
            model,
            model_name,
        )
    else:
        llm_records = []

    write_jsonl(llm_records, tmp_paths["llm_cache"])
    print(
        f"[cache] {language}: wrote {len(llm_records)} LLM outputs "
        f"to {tmp_paths['llm_cache']}"
    )
    return llm_stage.index_by_token_id(llm_records), model_state


def apply_cached_llm(master_records, candidates, cached_outputs):
    missing = [
        record["Token_id"]
        for record in candidates
        if record["Token_id"] not in cached_outputs
    ]
    if missing:
        preview = ", ".join(str(token_id) for token_id in missing[:10])
        raise ValueError(
            "Missing cached LLM outputs for Token_id values: "
            f"{preview}"
            + ("..." if len(missing) > 10 else "")
        )

    llm_records = [
        cached_outputs[record["Token_id"]]
        for record in candidates
    ]
    return llm_stage.merge_llm_outputs(master_records, llm_records)


def build_result(
    language,
    detector_threshold,
    entropy_threshold,
    summary,
    llm_mode,
):
    counts = summary["counts"]
    total = summary["overall_final"]
    detector = summary["detector"]
    dictionary = summary["dictionary_source"]
    llm = summary["llm_source"]

    return {
        "language": language,
        "detector_threshold": detector_threshold,
        "entropy_threshold": entropy_threshold,
        "llm_mode": llm_mode,
        "total": {
            "count": total["count"],
            "f1": total["scores"]["f1"],
            "err": total["scores"]["err"],
        },
        "detector": {
            "count": (
                detector["confusion"]["TP"]
                + detector["confusion"]["FP"]
            ),
            "precision": detector["scores"]["precision"],
            "recall": detector["scores"]["recall"],
            "f1": detector["scores"]["f1"],
        },
        "dictionary": {
            "replaced_count": counts["dictionary_replaced"],
            "f1": dictionary["scores"]["f1"],
            "err": dictionary["scores"]["err"],
        },
        "llm": {
            "replaced_count": counts["llm_replaced"],
            "f1": llm["scores"]["f1"],
            "err": llm["scores"]["err"],
        },
    }


def select_best_result(results):
    if not results:
        raise ValueError("Cannot select thresholds from empty results")

    default_detector, default_entropy = DEFAULT_THRESHOLD_PAIR
    return max(
        results,
        key=lambda result: (
            result["total"]["err"],
            result["total"]["f1"],
            -(
                abs(
                    result["detector_threshold"]
                    - default_detector
                )
                + abs(
                    result["entropy_threshold"]
                    - default_entropy
                )
            ),
            -result["detector_threshold"],
            -result["entropy_threshold"],
        ),
    )


def build_selected_summary(best_result, model_name, detector_grid, entropy_grid):
    return {
        "language": best_result["language"],
        "model": model_name,
        "split": "dev",
        "detector_threshold": best_result["detector_threshold"],
        "entropy_threshold": best_result["entropy_threshold"],
        "err": best_result["total"]["err"],
        "f1": best_result["total"]["f1"],
        "detector_grid": detector_grid,
        "entropy_grid": entropy_grid,
    }


def load_existing_languages(path):
    if not path.is_file():
        return set()
    languages = set()
    with path.open(encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                languages.add(json.loads(line)["language"])
    return languages


def run_language(args, language, model_state):
    input_paths = build_input_paths(args.data_root, language)
    tmp_paths = build_tmp_paths(args.tmp_root, args.model_name, language)
    validate_inputs(input_paths)

    min_detector = min(args.detector_thresholds)
    min_entropy = min(args.entropy_thresholds)
    cached_outputs, model_state = ensure_llm_cache(
        language,
        input_paths,
        tmp_paths,
        min_detector,
        min_entropy,
        args.model_name,
        model_state,
        args.model_path,
        args.dtype,
        args.resume,
    )

    tmp_paths["grid_results"].parent.mkdir(parents=True, exist_ok=True)
    tmp_paths["grid_results"].write_text("", encoding="utf-8")

    combinations = build_combinations(
        args.detector_thresholds,
        args.entropy_thresholds,
    )
    results = []
    for index, (detector_threshold, entropy_threshold) in enumerate(
        combinations,
        start=1,
    ):
        print(
            f"[{language} {index}/{len(combinations)}] "
            f"detector={detector_threshold:g}, entropy={entropy_threshold:g}"
        )
        master_records, candidates = build_records(
            language,
            input_paths,
            detector_threshold,
            entropy_threshold,
        )
        final_records = apply_cached_llm(master_records, candidates, cached_outputs)
        summary = evaluation_stage.summarize(final_records)
        result = build_result(
            language,
            detector_threshold,
            entropy_threshold,
            summary,
            "cached",
        )
        append_jsonl(result, tmp_paths["grid_results"])
        results.append(result)

    best_result = select_best_result(results)
    selected = build_selected_summary(
        best_result,
        args.model_name,
        args.detector_thresholds,
        args.entropy_thresholds,
    )
    print(
        f"[selected] {language}: "
        f"detector={selected['detector_threshold']:g}, "
        f"entropy={selected['entropy_threshold']:g}, "
        f"ERR={selected['err']:.4f}, F1={selected['f1']:.4f}"
    )
    return selected, model_state


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.tmp_root = args.tmp_root.resolve()
    args.output = args.output.resolve()
    args.model_path = args.model_path.resolve()

    if not args.resume:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")

    completed = load_existing_languages(args.output) if args.resume else set()
    model_state = None
    for language in args.languages:
        if language in completed:
            print(f"[skip] {language}: already present in {args.output}")
            continue
        selected, model_state = run_language(args, language, model_state)
        append_jsonl(selected, args.output)
        print(f"Wrote selected threshold to {args.output}")


if __name__ == "__main__":
    main()
