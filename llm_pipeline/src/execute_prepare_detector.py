import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.config import (
    DETECTOR_DEVICE,
    MACHAMP_PARAMS_CONFIG_PATH,
)


# Run one pipeline command with the selected language.
def run_command(command, language):
    # Run one pipeline command and stop if it fails.
    print("Running:", " ".join(str(part) for part in command))
    env = os.environ.copy()
    env["PIPELINE_LANGUAGE"] = language
    subprocess.run(command, check=True, env=env)


# Parse the detector language.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare detector data and train a language-specific detector."
    )
    parser.add_argument("--language", default="en")
    return parser.parse_args()


# Prepare detector data and train the MaChAmp detector.
def main():
    args = parse_args()
    dataset_config = Path(
        f"models/machamp/configs/machamp_detector_{args.language}.json"
    )
    model_dir = Path(f"models/machamp/detector_{args.language}_xlmr")

    run_command(
        [sys.executable, "-m", "src.prepare_detector_data"],
        args.language,
    )
    run_command(
        [
            sys.executable,
            "external/machamp/train.py",
            "--dataset_configs",
            str(dataset_config),
            "--parameters_config",
            str(MACHAMP_PARAMS_CONFIG_PATH),
            "--model_dir",
            str(model_dir),
            "--device",
            DETECTOR_DEVICE,
        ],
        args.language,
    )


if __name__ == "__main__":
    main()
