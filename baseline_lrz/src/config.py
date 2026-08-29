from pathlib import Path
# Config file

# Change language
import json
import os
LANGUAGE = os.environ.get("LANGUAGE", "en")
DEFAULT_DETECTOR_THRESHOLD = 0.5
DEFAULT_ENTROPY_THRESHOLD = 0.5

# Stage 3: LLM inference
HF_MODEL_PATH = Path(
    os.environ.get(
        "HF_MODEL_PATH",
        "/dss/dsshome1/01/ge65nus2/projects/"
        "MultiLexNorm_LLM/models/huggingface/Qwen3.5-9B",
    )
)
HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME", "qwen3.5_9b")
LLM_DTYPE = os.environ.get("LLM_DTYPE", "float16")
safe_name = HF_MODEL_NAME


def read_bool_env(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


# change datasets(train/ validation)
IS_VAL = read_bool_env("IS_VAL")
IS_OPTIMIZED = read_bool_env("IS_OPTIMIZED")
INPUT_SUFFIX = "_val" if IS_VAL else ""
RUN_SUFFIX = INPUT_SUFFIX + ("_OP" if IS_OPTIMIZED else "")

THRESHOLD_SEARCH_RESULTS_PATH = Path(
    os.environ.get(
        "THRESHOLD_SEARCH_RESULTS_PATH",
        f"data/threshold_search_results_{HF_MODEL_NAME}.jsonl",
    )
)


def load_optimized_thresholds(path, language):
    if not path.is_file():
        raise FileNotFoundError(
            "IS_OPTIMIZED=true but threshold search results were not found: "
            f"{path}"
        )

    selected = {}
    with path.open(encoding="utf-8") as reader:
        for line in reader:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("language") == language:
                selected = record
    if not selected:
        raise KeyError(
            "IS_OPTIMIZED=true but no threshold result was found for "
            f"language={language!r} in {path}"
        )
    return selected


def get_threshold(env_name, threshold_key, default_threshold, language):
    if env_name in os.environ:
        return float(os.environ[env_name])
    if not IS_OPTIMIZED:
        return default_threshold

    optimized = load_optimized_thresholds(THRESHOLD_SEARCH_RESULTS_PATH, language)
    if threshold_key in optimized:
        return float(optimized[threshold_key])
    raise KeyError(
        "IS_OPTIMIZED=true but threshold key "
        f"{threshold_key!r} was not found for language={language!r} "
        f"in {THRESHOLD_SEARCH_RESULTS_PATH}"
    )

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
GOLD_PATH = Path(f"data/{LANGUAGE}/gold_{LANGUAGE}{INPUT_SUFFIX}.jsonl")

# MaChAmp detector model
MACHAMP_DATASET_CONFIG_PATH = Path(
    f"models/machamp/configs/machamp_detector_{LANGUAGE}{INPUT_SUFFIX}.json"
)
MACHAMP_PARAMS_CONFIG_PATH = Path("models/machamp/configs/machamp_params_detector.json")
DETECTOR_MODEL_DIR = Path(f"models/machamp/detector_{LANGUAGE}_xlmr")
DETECTOR_MODEL_PATH = DETECTOR_MODEL_DIR / "model.pt"
DETECTOR_DEVICE = "0"

# Dictionary
DICTIONARY_PATH = Path(f"data/{LANGUAGE}/dictionary_{LANGUAGE}.jsonl")
ENTROPY_THRESHOLD = get_threshold(
    "ENTROPY_THRESHOLD",
    "entropy_threshold",
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
    "detector_threshold",
    DEFAULT_DETECTOR_THRESHOLD,
    LANGUAGE,
)
DETECTOR_CONFIDENCE_PATH = Path(
    f"data/detector_output/detector_{LANGUAGE}{INPUT_SUFFIX}.confidence.out"
)
STAGE2_OUTPUT_PATH = Path(f"data/{LANGUAGE}/table_applied_dictionary_{LANGUAGE}{RUN_SUFFIX}.jsonl")
STAGE3_LLM_CANDIDATES_PATH = Path(f"data/{LANGUAGE}/llm_candidates_{LANGUAGE}{RUN_SUFFIX}.jsonl")
STAGE3_LLM_PROMPTS_PATH = Path(f"data/{LANGUAGE}/llm_prompts_{LANGUAGE}{RUN_SUFFIX}.jsonl")
NUM_LLM_SHOTS = 8

# transfer to safer save name
STAGE3_LLM_APPLIED_PATH = Path(
    f"data/{LANGUAGE}/llm_candidates_applied_llm_{LANGUAGE}{RUN_SUFFIX}.jsonl"
)
STAGE3_MASTER_TABLE_PATH = Path(f"data/{LANGUAGE}/table_applied_{safe_name}_{LANGUAGE}{RUN_SUFFIX}.jsonl")

# Evaluation
EVALUATION_SUMMARY_PATH = Path(f"data/{LANGUAGE}/evaluation_summary_{safe_name}_{LANGUAGE}{RUN_SUFFIX}.json")
