from pathlib import Path

# Change language
LANGUAGE = "de"
DATA_DIR = Path("data") / LANGUAGE


# All Data -> EN Training Data -> DEV_RATIO -> TRAIN Data (90%) & DEV Data (10%)
DATASET_NAME = "weerayut/multilexnorm2026-dev-pub"
SPLIT_NAME = "train"
DEV_RATIO = 0.1
SEED = 42


### Detector ### Train/dev data
MACHAMP_DATASET_NAME = f"detector_{LANGUAGE}"
DETECTOR_TRAIN_DIR = DATA_DIR / "detector_train"
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
NORM_LABEL = "NORM"
DETECTOR_THRESHOLD = 0.5
DETECTOR_CONFIDENCE_PATH = (
    DATA_DIR / "detector_output" / f"detector_{LANGUAGE}.confidence.out"
)

# Selected language data -> DEV_RATIO -> DEV data
DEV_PATH = DATA_DIR / f"dev_raw_norm_{LANGUAGE}.jsonl"


### Dictionary ###
DICTIONARY_PATH = DATA_DIR / f"dictionary_{LANGUAGE}.jsonl"
ENTROPY_THRESHOLD = 0.5
STAGE2_OUTPUT_PATH = DATA_DIR / f"table_applied_dictionary_{LANGUAGE}.jsonl"
STAGE3_LLM_CANDIDATES_PATH = DATA_DIR / f"llm_candidates_{LANGUAGE}.jsonl"


### LLM inference ###
STAGE3_LLM_PROMPTS_PATH = DATA_DIR / f"llm_prompts_{LANGUAGE}.jsonl"
NUM_LLM_SHOTS = 8
OLLAMA_MODEL = "qwen3.5:9b"
OLLAMA_URL = "http://localhost:11434/api/chat"
STAGE3_LLM_APPLIED_PATH = (
    DATA_DIR / f"llm_candidates_applied_llm_{LANGUAGE}.jsonl"
)
STAGE3_MASTER_TABLE_PATH = DATA_DIR / f"table_applied_llm_{LANGUAGE}.jsonl"


### Evaluation ###
EVALUATION_SUMMARY_PATH = DATA_DIR / f"evaluation_summary_{LANGUAGE}.json"
