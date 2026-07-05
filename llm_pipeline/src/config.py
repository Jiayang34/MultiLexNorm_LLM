import os
from pathlib import Path


def sanitize_model_name(model):
    return model.replace("/", "__")


# Run pipeline output dir
def build_run_data_dir(
    language,
    model,
    detector_threshold,
    entropy_threshold,
    data_root=Path("data"),
):
    run_name = (
        f"{language}_{detector_threshold:g}_{entropy_threshold:g}"
    )
    return Path(data_root) / sanitize_model_name(model) / run_name


# Search threshold output dir
def build_threshold_data_dir(
    language,
    model,
    data_root=Path("data"),
):
    return (
        Path(data_root)
        / sanitize_model_name(model)
        / f"{language}_thresholds"
    )

# Change language
LANGUAGE = os.getenv("PIPELINE_LANGUAGE", "en")
MODEL = os.getenv("MODEL", "qwen3.5:9b")
DETECTOR_THRESHOLD = float(os.getenv("PIPELINE_DETECTOR_THRESHOLD", "0.5"))
ENTROPY_THRESHOLD = float(os.getenv("PIPELINE_ENTROPY_THRESHOLD", "0.5"))
DATA_DIR = Path(
    os.getenv(
        "PIPELINE_DATA_DIR",
        str(
            build_run_data_dir(
                LANGUAGE,
                MODEL,
                DETECTOR_THRESHOLD,
                ENTROPY_THRESHOLD,
            )
        ),
    )
)


# All Data -> EN Training Data -> DEV_RATIO -> TRAIN Data (90%) & DEV Data (10%)
DATASET_NAME = "weerayut/multilexnorm2026-dev-pub"
SPLIT_NAME = "train"
DEV_RATIO = 0.1
SEED = 42


### Detector ### Train/dev data
MACHAMP_DATASET_NAME = f"detector_{LANGUAGE}"
DETECTOR_TRAIN_DIR = Path("models") / "machamp" / "train_dev" / LANGUAGE
MACHAMP_TRAIN_PATH = DETECTOR_TRAIN_DIR / f"detector_train_{LANGUAGE}.tsv"
MACHAMP_DEV_PATH = DETECTOR_TRAIN_DIR / f"detector_dev_{LANGUAGE}.tsv"

### Detector ### MaChAmp
MACHAMP_DATASET_CONFIG_PATH = Path(
    f"models/machamp/configs/machamp_detector_{LANGUAGE}.json"
)
MACHAMP_PARAMS_CONFIG_PATH = Path("models/machamp/configs/machamp_params_detector.json")
DETECTOR_MODEL_DIR = Path(f"models/machamp/detector_{LANGUAGE}_xlmr")
DETECTOR_MODEL_PATH = DETECTOR_MODEL_DIR / "model.pt"
DETECTOR_DEVICE = "0"

### Detector ### Setup
KEEP_LABEL = "O"
# Binary detector label.
NORM_LABEL = "NORM"
# Length-aware detector labels.
NORM_1WORD_LABEL = "NORM_1WORD"
NORM_2WORD_LABEL = "NORM_2WORD"
NORM_3PLUS_LABEL = "NORM_3PLUS"
NORM_LENGTH_LABELS = (
    NORM_1WORD_LABEL,
    NORM_2WORD_LABEL,
    NORM_3PLUS_LABEL,
)
NORM_LABELS = frozenset((NORM_LABEL, *NORM_LENGTH_LABELS))
DETECTOR_CONFIDENCE_PATH = (
    DATA_DIR / "detector_output" / f"detector_{LANGUAGE}.confidence.out"
)

# Selected language data -> DEV_RATIO -> DEV data
DEV_PATH = DATA_DIR / f"dev_raw_norm_{LANGUAGE}.jsonl"


### Dictionary ###
DICTIONARY_PATH = DATA_DIR / f"dictionary_{LANGUAGE}.jsonl"
STAGE2_OUTPUT_PATH = DATA_DIR / f"table_applied_dictionary_{LANGUAGE}.jsonl"
STAGE3_LLM_CANDIDATES_PATH = DATA_DIR / f"llm_candidates_{LANGUAGE}.jsonl"


### LLM inference ###
STAGE3_LLM_PROMPTS_PATH = DATA_DIR / f"llm_prompts_{LANGUAGE}.jsonl"
NUM_LLM_SHOTS = 8
STAGE3_LLM_APPLIED_PATH = (
    DATA_DIR / f"llm_candidates_applied_llm_{LANGUAGE}.jsonl"
)
STAGE3_MASTER_TABLE_PATH = DATA_DIR / f"table_applied_llm_{LANGUAGE}.jsonl"


### Evaluation ###
EVALUATION_SUMMARY_PATH = DATA_DIR / f"evaluation_summary_{LANGUAGE}.json"
