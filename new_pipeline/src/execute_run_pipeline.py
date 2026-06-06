import subprocess
import sys

from src.config import (
    DETECTOR_CONFIDENCE_PATH,
    DETECTOR_DEVICE,
    DETECTOR_MODEL_PATH,
    MACHAMP_DEV_PATH,
)


def run_command(command):
    # Run one pipeline command and stop if it fails.
    print("Running:", " ".join(str(part) for part in command))
    subprocess.run(command, check=True)


def main():
    # Run the 2026 LLM pipeline end to end after detector preparation.
    DETECTOR_CONFIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "external/machamp/predict.py",
            str(DETECTOR_MODEL_PATH),
            str(MACHAMP_DEV_PATH),
            str(DETECTOR_CONFIDENCE_PATH),
            "--dataset",
            "detector_en",
            "--device",
            DETECTOR_DEVICE,
            "--topn",
            "2",
        ]
    )
    run_command([sys.executable, "-m", "src.build_dictionary"])
    run_command([sys.executable, "-m", "src.apply_dictionary"])
    run_command([sys.executable, "-m", "src.build_llm_prompts"])
    run_command([sys.executable, "-m", "src.run_llm"])


if __name__ == "__main__":
    main()
