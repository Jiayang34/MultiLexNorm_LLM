import argparse
import subprocess
import sys
from pathlib import Path


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
SUPPORTED_MODELS = ["qwen3.5:9b", "deepseek-v4-pro"]
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PIPELINE_ROOT.parent / "our_pipeline1" / "data"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run length-aware detector threshold search for all languages."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=SUPPORTED_MODELS,
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=DEFAULT_LANGUAGES,
        help="Language codes to run in order.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="our_pipeline1 data directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Parent output directory. Defaults to "
            "data/newdetector_<model>."
        ),
    )
    parser.add_argument(
        "--check-cache",
        action="store_true",
        help="Forward --check-cache to each threshold search.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Attempt remaining languages after one language fails.",
    )
    return parser.parse_args()


def build_input_paths(input_root, language):
    return {
        "detector": (
            input_root
            / "detector_output"
            / f"detector_{language}.confidence.out"
        ),
        "dictionary": input_root / f"dictionary_{language}.jsonl",
    }


def build_search_command(
    language,
    model,
    input_root,
    output_root,
    check_cache=False,
):
    inputs = build_input_paths(input_root, language)
    command = [
        sys.executable,
        "-m",
        "src.search_thresholds",
        "--language",
        language,
        "--model",
        model,
        "--detector-output",
        str(inputs["detector"]),
        "--dictionary",
        str(inputs["dictionary"]),
        "--output-root",
        str(output_root),
    ]
    if check_cache:
        command.append("--check-cache")
    return command


def find_missing_inputs(input_root, language):
    return [
        path
        for path in build_input_paths(input_root, language).values()
        if not path.is_file()
    ]


def main():
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = (
        args.output_root
        if args.output_root is not None
        else Path("data") / f"newdetector_{args.model}"
    )
    if not output_root.is_absolute():
        output_root = PIPELINE_ROOT / output_root
    output_root = output_root.resolve()

    failures = []
    total = len(args.languages)
    for index, language in enumerate(args.languages, start=1):
        print(
            f"\n[{index}/{total}] Starting threshold search: "
            f"language={language}, model={args.model}",
            flush=True,
        )

        missing = find_missing_inputs(input_root, language)
        if missing:
            message = (
                f"Missing inputs for {language}:\n"
                + "\n".join(str(path) for path in missing)
            )
            if not args.continue_on_error:
                raise FileNotFoundError(message)
            print(message, file=sys.stderr, flush=True)
            failures.append(language)
            continue

        command = build_search_command(
            language,
            args.model,
            input_root,
            output_root,
            check_cache=args.check_cache,
        )
        print("Running:", " ".join(command), flush=True)
        try:
            subprocess.run(
                command,
                cwd=PIPELINE_ROOT,
                check=True,
            )
        except subprocess.CalledProcessError:
            if not args.continue_on_error:
                raise
            failures.append(language)
            print(
                f"Threshold search failed for {language}",
                file=sys.stderr,
                flush=True,
            )

    if failures:
        failed = ", ".join(failures)
        raise SystemExit(f"Threshold search failed for: {failed}")

    print(
        f"\nCompleted {total} language threshold searches. "
        f"Output root: {output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
