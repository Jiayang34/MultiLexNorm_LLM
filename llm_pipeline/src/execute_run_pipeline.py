import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.config import (
    DETECTOR_DEVICE,
    DETECTOR_THRESHOLD,
    ENTROPY_THRESHOLD,
    LANGUAGE,
    MODEL,
    build_run_data_dir,
)


def run_command(command, env):
    # Run one pipeline command and stop if it fails.
    print("Running:", " ".join(str(part) for part in command))
    subprocess.run(command, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Run the MultiLexNorm pipeline with configurable thresholds."
    )
    parser.add_argument("--language", default=LANGUAGE)
    parser.add_argument(
        "--detector-threshold",
        type=float,
        default=DETECTOR_THRESHOLD,
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=ENTROPY_THRESHOLD,
    )
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    data_dir = Path(
        os.getenv(
            "PIPELINE_DATA_DIR",
            str(
                build_run_data_dir(
                    args.language,
                    args.model,
                    args.detector_threshold,
                    args.entropy_threshold,
                )
            ),
        )
    )
    detector_model_path = (
        f"models/machamp/detector_{args.language}_xlmr/model.pt"
    )
    detector_dev_path = (
        f"models/machamp/train_dev/{args.language}/"
        f"detector_dev_{args.language}.tsv"
    )
    detector_confidence_path = (
        data_dir
        / "detector_output"
        / f"detector_{args.language}.confidence.out"
    )
    dataset_name = f"detector_{args.language}"

    # Child modules read these values through src.config.
    env = os.environ.copy()
    env["PIPELINE_LANGUAGE"] = args.language
    env["PIPELINE_DATA_DIR"] = str(data_dir)
    env["PIPELINE_DETECTOR_THRESHOLD"] = str(args.detector_threshold)
    env["PIPELINE_ENTROPY_THRESHOLD"] = str(args.entropy_threshold)
    env["MODEL"] = args.model

    print(
        "Pipeline settings: "
        f"language={args.language}, "
        f"detector_threshold={args.detector_threshold}, "
        f"entropy_threshold={args.entropy_threshold}, "
        f"model={args.model}, "
        f"output_dir={data_dir}"
    )

    # Run the 2026 LLM pipeline end to end after detector preparation.
    detector_confidence_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "external/machamp/predict.py",
            detector_model_path,
            detector_dev_path,
            str(detector_confidence_path),
            "--dataset",
            dataset_name,
            "--device",
            DETECTOR_DEVICE,
            "--topn",
            "2",
        ],
        env,
    )
    run_command([sys.executable, "-m", "src.build_dictionary"], env)
    run_command([sys.executable, "-m", "src.apply_dictionary"], env)
    run_command([sys.executable, "-m", "src.build_llm_prompts"], env)
    run_llm_command = [
        sys.executable,
        "-m",
        "src.run_llm",
        "--model",
        args.model,
    ]
    run_command(run_llm_command, env)
    run_command(
        [sys.executable, "-m", "src.evaluate_pipeline"],
        env,
    )


if __name__ == "__main__":
    main()
