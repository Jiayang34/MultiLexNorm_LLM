from pathlib import Path
# Config file

# Change language
import os
LANGUAGE = os.environ.get("LANGUAGE", "en")
DEFAULT_DETECTOR_THRESHOLD = 0.5
DEFAULT_ENTROPY_THRESHOLD = 0.5

OPTIMAL_DETECTOR_THRESHOLDS = {
    "de": 0.3,
    "en": 0.3,
    "hr": 0.7,
    "id": 0.1,
    "iden": 0.7,
    "ja": 0.9,
    "ko": 0.9,
    "nl": 0.3,
    "sl": 0.3,
    "sr": 0.3,
    "th": 0.5,
    "vi": 0.5,
}

OPTIMAL_ENTROPY_THRESHOLDS = {
    "de": 1.4,
    "en": 1.4,
    "hr": 1.4,
    "id": 1.4,
    "iden": 1.4,
    "ja": 1.7,
    "ko": 1.7,
    "nl": 1.4,
    "sl": 1.4,
    "sr": 1.7,
    "th": 1.7,
    "vi": 1.4,
}


def get_threshold(env_name, optimal_thresholds, default_threshold, language):
    if env_name in os.environ:
        return float(os.environ[env_name])
    return optimal_thresholds.get(language, default_threshold)


# change datasets(train/ validation)
IS_VAL = os.environ.get("IS_VAL", "false").lower() in {"1", "true", "yes"}
RUN_SUFFIX = "_val" if IS_VAL else ""

# All Data -> EN Training Data -> DEV_RATIO -> TRAIN Data (90%) & DEV Data (10%)
DATASET_NAME = "weerayut/multilexnorm2026-dev-pub"
SPLIT_NAME = "train"
VAL_SPLIT_NAME = os.environ.get("VAL_SPLIT_NAME", "validation")
DEV_RATIO = 0.1
SEED = 42

# Detector train/dev data
MACHAMP_DATASET_NAME = f"detector_{LANGUAGE}"
DETECTOR_TRAIN_DIR = Path("data/detector_train")
MACHAMP_TRAIN_PATH = DETECTOR_TRAIN_DIR / f"detector_train_{LANGUAGE}.tsv"
MACHAMP_DEV_PATH = DETECTOR_TRAIN_DIR / f"detector_dev_{LANGUAGE}.tsv"
# Detector validation data
MACHAMP_VAL_PATH = DETECTOR_TRAIN_DIR / f"detector_val_{LANGUAGE}.tsv"
MACHAMP_PREDICT_INPUT_PATH = MACHAMP_VAL_PATH if IS_VAL else MACHAMP_DEV_PATH
GOLD_PATH = Path(f"data/{LANGUAGE}/gold_{LANGUAGE}{RUN_SUFFIX}.jsonl")

# MaChAmp detector model
MACHAMP_DATASET_CONFIG_PATH = Path(
    f"models/machamp/configs/machamp_detector_{LANGUAGE}{RUN_SUFFIX}.json"
)
MACHAMP_PARAMS_CONFIG_PATH = Path("models/machamp/configs/machamp_params_detector.json")
DETECTOR_MODEL_DIR = Path(f"models/machamp/detector_{LANGUAGE}_xlmr")
DETECTOR_MODEL_PATH = DETECTOR_MODEL_DIR / "model.pt"
DETECTOR_DEVICE = "0"

# Dictionary
DICTIONARY_PATH = Path(f"data/{LANGUAGE}/dictionary_{LANGUAGE}.jsonl")
ENTROPY_THRESHOLD = get_threshold(
    "ENTROPY_THRESHOLD",
    OPTIMAL_ENTROPY_THRESHOLDS,
    DEFAULT_ENTROPY_THRESHOLD,
    LANGUAGE,
)

# Selected language data -> DEV_RATIO -> DEV data
DEV_PATH = Path(f"data/{LANGUAGE}/dev_raw_norm_{LANGUAGE}.jsonl")

# Stage 2: Detector -> Dictionary lookup
KEEP_LABEL = "O"
NORM_LABEL = "NORM"
DETECTOR_THRESHOLD = get_threshold(
    "DETECTOR_THRESHOLD",
    OPTIMAL_DETECTOR_THRESHOLDS,
    DEFAULT_DETECTOR_THRESHOLD,
    LANGUAGE,
)
DETECTOR_CONFIDENCE_PATH = Path(
    f"data/detector_output/detector_{LANGUAGE}{RUN_SUFFIX}.confidence.out"
)
STAGE2_OUTPUT_PATH = Path(f"data/{LANGUAGE}/table_applied_dictionary_{LANGUAGE}{RUN_SUFFIX}.jsonl")
STAGE3_LLM_CANDIDATES_PATH = Path(f"data/{LANGUAGE}/llm_candidates_{LANGUAGE}{RUN_SUFFIX}.jsonl")
STAGE3_LLM_PROMPTS_PATH = Path(f"data/{LANGUAGE}/llm_prompts_{LANGUAGE}{RUN_SUFFIX}.jsonl")
NUM_LLM_SHOTS = 8

# Stage 3: LLM inference
OLLAMA_MODEL = "qwen3.5:9b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
# transfer to safer save name
safe_name = OLLAMA_MODEL.replace(":", "_")
STAGE3_LLM_APPLIED_PATH = Path(
    f"data/{LANGUAGE}/llm_candidates_applied_llm_{LANGUAGE}{RUN_SUFFIX}.jsonl"
)
STAGE3_MASTER_TABLE_PATH = Path(f"data/{LANGUAGE}/table_applied_{safe_name}_{LANGUAGE}{RUN_SUFFIX}.jsonl")

# Evaluation
EVALUATION_SUMMARY_PATH = Path(f"data/{LANGUAGE}/evaluation_summary_{safe_name}_{LANGUAGE}{RUN_SUFFIX}.json")
