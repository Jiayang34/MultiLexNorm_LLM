import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.config import MODEL, build_threshold_data_dir


DEFAULT_DETECTOR_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_ENTROPY_THRESHOLDS = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7]
#DEFAULT_ENTROPY_THRESHOLDS = [0.5, 0.8, 1.4, 1.6, 1.8, 2.0]
DEFAULT_THRESHOLD_PAIR = (0.5, 0.5)
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
    parser.add_argument(
        "--detector-output",
        type=Path,
        default=None,
        help=(
            "Existing detector confidence output for the fixed dev split. "
            "When provided, skip MaChAmp prediction."
        ),
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=None,
        help=(
            "Existing dictionary JSONL. When provided, skip dictionary "
            "construction."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional parent output directory. Results are written to its "
            "<language>_thresholds subdirectory."
        ),
    )
    return parser.parse_args()


def run_command(command, env=None):
    print("Running:", " ".join(str(part) for part in command))
    subprocess.run(command, check=True, env=env)


def load_json(path):
    with path.open(encoding="utf-8") as reader:
        return json.load(reader)


def write_json(record, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        json.dump(record, writer, ensure_ascii=False, indent=2)
        writer.write("\n")


def validate_dev_detector_path(path):
    if "_val" in path.name:
        raise ValueError(
            "Threshold search must use the fixed 10% dev split, not "
            f"validation detector output: {path}"
        )


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


def build_paths(language, model, output_root=None):
    data_dir = (
        output_root / f"{language}_thresholds"
        if output_root is not None
        else build_threshold_data_dir(language, model)
    )
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
        "err_plot": data_dir / f"threshold_entropy_err_{language}.png",
        "f1_plot": data_dir / f"threshold_entropy_f1_{language}.png",
    }


# First round: Prepare caches: detector, dictionary and LLM outputs (applied on widest candidates)
def prepare_caches(
    language,
    paths,
    check_cache,
    model,
    external_detector=False,
    external_dictionary=False,
):
    detector_output_path = paths["detector"]
    dictionary_path = paths["dictionary"]
    cache_path = paths["llm_cache"]
    data_dir = paths["data_dir"]

    if not external_detector:
        detector_output_path.parent.mkdir(parents=True, exist_ok=True)
    base_env = build_environment(language, data_dir, model)

    # Check cache
    reuse_detector = (
        external_detector
        or (check_cache and detector_output_path.exists())
    )
    reuse_dictionary = (
        external_dictionary
        or (check_cache and dictionary_path.exists())
    )
    reuse_llm = (
        check_cache
        and reuse_detector
        and reuse_dictionary
        and cache_path.exists()
    )
    print(
        f"[cache] detector: "
        f"{'external' if external_detector else ('found' if reuse_detector else 'rebuild')}\n"
        f"[cache] dictionary: "
        f"{'external' if external_dictionary else ('found' if reuse_dictionary else 'rebuild')}\n"
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
            [
                sys.executable,
                "-m",
                "src.apply_dictionary",
                "--detector-output",
                str(detector_output_path),
                "--dictionary",
                str(dictionary_path),
            ],
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


# Best threshold selection order:
# higher ERR -> higher F1 -> closed to 0.5, 0.5 -> lower threshold
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


def build_selected_summary(best_result):
    return {
        "language": best_result["language"],
        "detector_threshold": best_result["detector_threshold"],
        "entropy_threshold": best_result["entropy_threshold"],
        "err": best_result["total"]["err"],
        "f1": best_result["total"]["f1"],
    }


def plot_results(language, model, paths):
    run_command(
        [
            sys.executable,
            "-m",
            "src.plot_threshold_results",
            "--language",
            language,
            "--model",
            model,
            "--input",
            str(paths["results"]),
            "--err-output",
            str(paths["err_plot"]),
            "--f1-output",
            str(paths["f1_plot"]),
        ]
    )


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
            "--detector-output",
            str(paths["detector"]),
            "--dictionary",
            str(paths["dictionary"]),
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
    paths = build_paths(args.language, args.model, args.output_root)

    external_detector = args.detector_output is not None
    if external_detector:
        paths["detector"] = args.detector_output.resolve()
        validate_dev_detector_path(paths["detector"])
        if not paths["detector"].is_file():
            raise FileNotFoundError(
                f"Detector output not found: {paths['detector']}"
            )

    external_dictionary = args.dictionary is not None
    if external_dictionary:
        paths["dictionary"] = args.dictionary.resolve()
        if not paths["dictionary"].is_file():
            raise FileNotFoundError(
                f"Dictionary not found: {paths['dictionary']}"
            )

    paths["results"].parent.mkdir(parents=True, exist_ok=True)
    paths["results"].write_text("", encoding="utf-8")

    prepare_caches(
        args.language,
        paths,
        args.check_cache,
        args.model,
        external_detector=external_detector,
        external_dictionary=external_dictionary,
    )
    combinations = build_combinations()
    results = []

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
        results.append(result)
        print(f"Saved result to {paths['results']}")

    best_result = select_best_result(results)
    best_key = (
        best_result["detector_threshold"],
        best_result["entropy_threshold"],
    )
    selected_summary = build_selected_summary(best_result)
    write_json(selected_summary, paths["summary"])
    print(
        "Selected best thresholds: "
        f"detector={best_key[0]:g}, entropy={best_key[1]:g}, "
        f"ERR={best_result['total']['err']:.4f}, "
        f"F1={best_result['total']['f1']:.4f}"
    )
    print(f"Wrote selected evaluation summary to {paths['summary']}")

    plot_results(args.language, args.model, paths)


if __name__ == "__main__":
    main()
