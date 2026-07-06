import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.config import (
    DETECTOR_DEVICE,
    MACHAMP_PARAMS_CONFIG_PATH,
)


DEFAULT_DATA_ROOT = Path("data/wiki_pretrain_finetune")
DEFAULT_MODEL_ROOT = Path("models/machamp")


# Run one command with the selected pipeline language.
def execute_command(command, language):
    print("Running:", " ".join(str(part) for part in command), flush=True)
    env = os.environ.copy()
    env["PIPELINE_LANGUAGE"] = language
    subprocess.run(command, check=True, env=env)


# Refuse to reuse existing outputs unless explicitly requested.
def check_outputs(paths, overwrite):
    if overwrite:
        return

    existing = [
        path
        for path in paths.values()
        if isinstance(path, Path) and path.exists()
    ]
    if existing:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Experiment outputs already exist. Use --overwrite to replace:\n"
            f"{joined}"
        )


# Build all run-specific output paths.
def build_paths(language, run_name, data_root, model_root):
    experiment_dir = data_root / language / run_name
    return {
        "experiment_dir": experiment_dir,
        "clean_tokens": experiment_dir / "clean_tokens.jsonl",
        "noisy_data": experiment_dir / "noisy_raw_norm.jsonl",
        "detector_output_dir": experiment_dir / "detector_output",
        "detector_output": (
            experiment_dir
            / "detector_output"
            / f"detector_{language}.confidence.out"
        ),
        "pretrain_config": (
            model_root
            / "configs"
            / f"machamp_detector_{language}_{run_name}_pretrain.json"
        ),
        "pretrain_model_dir": (
            model_root / f"detector_{language}_xlmr_{run_name}_pretrain"
        ),
        "pretrain_model": (
            model_root / f"detector_{language}_xlmr_{run_name}_pretrain" / "model.pt"
        ),
        "finetune_config": (
            model_root / "configs" / f"machamp_detector_{language}.json"
        ),
        "finetune_dev": (
            model_root
            / "train_dev"
            / language
            / f"detector_dev_{language}.tsv"
        ),
        "finetune_model_dir": (
            model_root / f"detector_{language}_xlmr_{run_name}_finetune"
        ),
        "finetune_model": (
            model_root / f"detector_{language}_xlmr_{run_name}_finetune" / "model.pt"
        ),
    }


# Stream clean wiki sentences into token JSONL.
def execute_import_clean_wiki(args, paths):
    execute_command(
        [
            sys.executable,
            "-m",
            "src.import_clean_wiki",
            "--language",
            args.language,
            "--max-sentences",
            str(args.max_sentences),
            "--stanza-model-dir",
            str(args.stanza_model_dir),
            "--output",
            str(paths["clean_tokens"]),
        ],
        args.language,
    )


# Add train-frequency and rule-based synthetic noise.
def execute_add_noise(args, paths):
    execute_command(
        [
            sys.executable,
            "-m",
            "src.add_noise",
            "--language",
            args.language,
            "--input",
            str(paths["clean_tokens"]),
            "--output",
            str(paths["noisy_data"]),
            "--noise-scale",
            str(args.noise_scale),
            "--apostrophe-prob",
            str(args.apostrophe_prob),
            "--delete-char-prob",
            str(args.delete_char_prob),
            "--repeat-char-prob",
            str(args.repeat_char_prob),
            "--swap-char-prob",
            str(args.swap_char_prob),
            "--case-prob",
            str(args.case_prob),
            "--seed",
            str(args.seed),
        ],
        args.language,
    )


# Prepare a synthetic-only MaChAmp detector config for pretraining.
def execute_prepare_pretrain_data(args, paths):
    execute_command(
        [
            sys.executable,
            "-m",
            "src.prepare_detector_data",
            "--synthetic-data",
            str(paths["noisy_data"]),
            "--run-name",
            f"{args.run_name}_pretrain",
        ],
        args.language,
    )


# Train the detector on synthetic-only data first.
def execute_pretrain_detector(args, paths):
    execute_command(
        [
            sys.executable,
            "external/machamp/train.py",
            "--dataset_configs",
            str(paths["pretrain_config"]),
            "--parameters_config",
            str(args.pretrain_parameters_config),
            "--model_dir",
            str(paths["pretrain_model_dir"]),
            "--device",
            args.device,
        ],
        args.language,
    )


# Finetune on the original train split, initialized from the synthetic encoder.
def execute_finetune_detector(args, paths):
    execute_command(
        [
            sys.executable,
            "external/machamp/train.py",
            "--dataset_configs",
            str(paths["finetune_config"]),
            "--parameters_config",
            str(args.finetune_parameters_config),
            "--model_dir",
            str(paths["finetune_model_dir"]),
            "--retrain",
            str(paths["pretrain_model"]),
            "--device",
            args.device,
        ],
        args.language,
    )


# Predict on the fixed detector dev split with the finetuned detector.
def execute_predict_detector(args, paths):
    paths["detector_output_dir"].mkdir(parents=True, exist_ok=True)
    execute_command(
        [
            sys.executable,
            "external/machamp/predict.py",
            str(paths["finetune_model"]),
            str(paths["finetune_dev"]),
            str(paths["detector_output"]),
            "--dataset",
            f"detector_{args.language}",
            "--device",
            args.device,
            "--topn",
            "2",
        ],
        args.language,
    )


# Parse experiment-level options.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute synthetic pretrain followed by original-train finetune."
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-sentences", type=int, default=5000)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--stanza-model-dir", type=Path, default=Path("models/stanza"))
    parser.add_argument(
        "--pretrain-parameters-config",
        type=Path,
        default=MACHAMP_PARAMS_CONFIG_PATH,
    )
    parser.add_argument(
        "--finetune-parameters-config",
        type=Path,
        default=MACHAMP_PARAMS_CONFIG_PATH,
    )
    parser.add_argument("--device", default=DETECTOR_DEVICE)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--apostrophe-prob", type=float, default=0.468)
    parser.add_argument("--delete-char-prob", type=float, default=0.003)
    parser.add_argument("--repeat-char-prob", type=float, default=0.003)
    parser.add_argument("--swap-char-prob", type=float, default=0.003)
    parser.add_argument("--case-prob", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--skip-noise", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# Execute the selected two-stage experiment steps.
def main():
    args = parse_args()
    paths = build_paths(
        args.language,
        args.run_name,
        args.data_root,
        args.model_root,
    )

    check_outputs(
        {
            "clean_tokens": paths["clean_tokens"],
            "noisy_data": paths["noisy_data"],
            "detector_output": paths["detector_output"],
            "pretrain_config": paths["pretrain_config"],
            "pretrain_model_dir": paths["pretrain_model_dir"],
            "finetune_model_dir": paths["finetune_model_dir"],
        },
        args.overwrite,
    )

    if not args.skip_import:
        execute_import_clean_wiki(args, paths)
    if not args.skip_noise:
        execute_add_noise(args, paths)
    if not args.skip_prepare:
        execute_prepare_pretrain_data(args, paths)
    if not args.skip_pretrain:
        execute_pretrain_detector(args, paths)
    if not args.skip_finetune:
        execute_finetune_detector(args, paths)
    if not args.skip_predict:
        execute_predict_detector(args, paths)

    print("Experiment paths:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
