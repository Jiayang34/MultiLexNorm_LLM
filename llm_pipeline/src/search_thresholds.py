import argparse
import json
import os
import subprocess
import sys

from src.config import MODEL, build_threshold_data_dir


DEFAULT_DETECTOR_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_ENTROPY_THRESHOLDS = [0.2, 0.5, 0.8, 1.1, 1.4]
'''
python -m src.search_thresholds   --language en   --check-cache

Run at first round and reuse caches afterwards:
run detector, build dictionary, build prompt & run LLM (run for widest LLM candidates (least thresholds), LLM results then frozen and reused, maybe different in real case)

--check-cache
Check and reuse caches
for adjusting grid search range
'''


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a real detector and entropy threshold grid search."
    )
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--check-cache",
        action="store_true",
        help="Reuse existing detector, dictionary, and LLM cache files.",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help="LLM model used when building the threshold-search cache.",
    )
    return parser.parse_args()


def run_command(command, env=None):
    print("Running:", " ".join(str(part) for part in command))
    subprocess.run(command, check=True, env=env)


def load_json(path):
    with path.open(encoding="utf-8") as reader:
        return json.load(reader)


def build_result(
    language,
    detector_threshold,
    entropy_threshold,
    summary,
    llm_mode,
):
    """Select the main metrics from one pipeline evaluation."""
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


def write_jsonl_record(record, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as writer:
        writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_environment(
    language,
    data_dir,
    model,
    detector_threshold=None,
    entropy_threshold=None,
):
    """Build one shared environment for threshold pipeline stages."""
    env = os.environ.copy()
    env["PIPELINE_LANGUAGE"] = language
    env["PIPELINE_DATA_DIR"] = str(data_dir)
    env["MODEL"] = model
    if detector_threshold is not None:
        env["PIPELINE_DETECTOR_THRESHOLD"] = str(detector_threshold)
    if entropy_threshold is not None:
        env["PIPELINE_ENTROPY_THRESHOLD"] = str(entropy_threshold)
    return env


def build_paths(language, model):
    data_dir = build_threshold_data_dir(language, model)
    return {
        "data_dir": data_dir,
        "summary": data_dir / f"evaluation_summary_{language}.json",
        "detector": (
            data_dir
            / "detector_output"
            / f"detector_{language}.confidence.out"
        ),
        "dictionary": data_dir / f"dictionary_{language}.jsonl",
        "llm_cache": data_dir / f"threshold_llm_cache_{language}.jsonl",
        "results": data_dir / f"threshold_results_{language}.jsonl",
    }


# First round: Prepare caches: detector, dictionary and LLM outputs (applied on widest candidates)
def prepare_caches(language, paths, check_cache, model):
    detector_output_path = paths["detector"]
    dictionary_path = paths["dictionary"]
    cache_path = paths["llm_cache"]
    data_dir = paths["data_dir"]

    detector_output_path.parent.mkdir(parents=True, exist_ok=True)
    base_env = build_environment(language, data_dir, model)

    # Check cache
    reuse_detector = check_cache and detector_output_path.exists()
    reuse_dictionary = check_cache and dictionary_path.exists()
    reuse_llm = (
        reuse_detector
        and reuse_dictionary
        and cache_path.exists()
    )
    print(
        f"[cache] detector: {'found' if reuse_detector else 'rebuild'}\n"
        f"[cache] dictionary: {'found' if reuse_dictionary else 'rebuild'}\n"
        f"[cache] llm: {'found' if reuse_llm else 'rebuild'}\n"
        f"[cache] requested model: {model}"
    )

    # Build caches
    if not reuse_detector:
        run_command(
            [
                sys.executable,
                "external/machamp/predict.py",
                f"models/machamp/detector_{language}_xlmr/model.pt",
                (
                    f"models/machamp/train_dev/{language}/"
                    f"detector_dev_{language}.tsv"
                ),
                str(detector_output_path),
                "--dataset",
                f"detector_{language}",
                "--device",
                "0",
                "--topn",
                "2",
            ],
            env=base_env,
        )

    if not reuse_dictionary:
        run_command(
            [sys.executable, "-m", "src.build_dictionary"],
            env=base_env,
        )

    if not reuse_llm:
        cache_env = build_environment(
            language,
            data_dir,
            model,
            min(DEFAULT_DETECTOR_THRESHOLDS),
            min(DEFAULT_ENTROPY_THRESHOLDS),
        )
        run_command(
            [sys.executable, "-m", "src.apply_dictionary"],
            env=cache_env,
        )
        run_command(
            [sys.executable, "-m", "src.build_llm_prompts"],
            env=cache_env,
        )
        cache_env["PIPELINE_LLM_CACHE_OUTPUT_PATH"] = str(cache_path)
        run_command(
            [
                sys.executable,
                "-m",
                "src.run_llm",
                "--model",
                model,
            ],
            env=cache_env,
        )
        print(f"Saved LLM cache to {cache_path}")


def build_combinations():
    return [
        (detector_threshold, entropy_threshold)
        for detector_threshold in DEFAULT_DETECTOR_THRESHOLDS
        for entropy_threshold in DEFAULT_ENTROPY_THRESHOLDS
    ]


# Second rounds and afterwards: reuse caches and apply
def run_combination(
    language,
    paths,
    model,
    detector_threshold,
    entropy_threshold,
):
    env = build_environment(
        language,
        paths["data_dir"],
        model,
        detector_threshold,
        entropy_threshold,
    )
    env["PIPELINE_LLM_CACHE_PATH"] = str(paths["llm_cache"])
    run_command(
        [
            sys.executable,
            "-m",
            "src.apply_dictionary",
        ],
        env=env,
    )
    run_command(
        [
            sys.executable,
            "-m",
            "src.run_llm",
            "--model",
            model,
        ],
        env=env,
    )

    run_command(
        [sys.executable, "-m", "src.evaluate_pipeline"],
        env=env,
    )

    summary = load_json(paths["summary"])
    return build_result(
        language,
        detector_threshold,
        entropy_threshold,
        summary,
        "cached",
    )


def main():
    args = parse_args()
    paths = build_paths(args.language, args.model)
    paths["results"].parent.mkdir(parents=True, exist_ok=True)
    paths["results"].write_text("", encoding="utf-8")

    prepare_caches(args.language, paths, args.check_cache, args.model)
    combinations = build_combinations()

    for run_index, (detector_threshold, entropy_threshold) in enumerate(
        combinations,
        start=1,
    ):
        print(
            f"\n[{run_index}/{len(combinations)}] "
            f"detector={detector_threshold}, entropy={entropy_threshold}"
        )

        result = run_combination(
            args.language,
            paths,
            args.model,
            detector_threshold,
            entropy_threshold,
        )
        write_jsonl_record(result, paths["results"])
        print(f"Saved result to {paths['results']}")


if __name__ == "__main__":
    main()
