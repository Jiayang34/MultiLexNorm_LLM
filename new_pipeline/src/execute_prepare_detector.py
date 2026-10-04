import subprocess
import sys

from src.config import (
    DETECTOR_DEVICE,
    DETECTOR_MODEL_DIR,
    MACHAMP_DATASET_CONFIG_PATH,
    MACHAMP_PARAMS_CONFIG_PATH,
)


def run_command(command):
    # Run one pipeline command and stop if it fails.
    print("Running:", " ".join(str(part) for part in command))
    subprocess.run(command, check=True)


def main():
    # Prepare detector data and train the MaChAmp detector.
    run_command([sys.executable, "-m", "src.prepare_detector_data"])
    run_command(
        [
            sys.executable,
            "external/machamp/train.py",
            "--dataset_configs",
            str(MACHAMP_DATASET_CONFIG_PATH),
            "--parameters_config",
            str(MACHAMP_PARAMS_CONFIG_PATH),
            "--model_dir",
            str(DETECTOR_MODEL_DIR),
            "--device",
            DETECTOR_DEVICE,
        ]
    )


if __name__ == "__main__":
    main()
