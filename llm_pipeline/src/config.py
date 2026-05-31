from pathlib import Path
# Config file

# All Data -> EN Training Data -> DEV_RATIO -> TRAIN Data (90%) & DEV Data (10%)
DATASET_NAME = "weerayut/multilexnorm2026-dev-pub"
SPLIT_NAME = "train"
LANGUAGE = "en"
DEV_RATIO = 0.1
SEED = 42

# Detector train/dev data
DETECTOR_TRAIN_DIR = Path("data/detector_train")
MACHAMP_TRAIN_PATH = DETECTOR_TRAIN_DIR / f"detector_train_{LANGUAGE}.tsv"
MACHAMP_DEV_PATH = DETECTOR_TRAIN_DIR / f"detector_dev_{LANGUAGE}.tsv"

# Dictionary
DICTIONARY_PATH = Path("data/dictionary_en.jsonl")
ENTROPY_THRESHOLD = 0.5

# EN Training Data -> DEV_RATIO -> DEV Data
DEV_PATH = Path("data") / f"dev_raw_norm_{LANGUAGE}.jsonl"

# Stage 2: Detector -> Dictionary lookup
KEEP_LABEL = "O"
NORM_LABEL = "NORM"
DETECTOR_CONFIDENCE_PATH = Path(
    "data/detector_output/detector_en.confidence.out"
)
STAGE2_OUTPUT_PATH = Path("data/table_applied_dictionary_en.jsonl")
STAGE3_LLM_CANDIDATES_PATH = Path("data/llm_candidates_en.jsonl")
STAGE3_LLM_PROMPTS_PATH = Path("data/llm_prompts_en.jsonl")
NUM_LLM_SHOTS = 8

# Stage 3: LLM inference
OLLAMA_MODEL = "qwen3.5:9b"
OLLAMA_URL = "http://localhost:11434/api/chat"
STAGE3_LLM_APPLIED_PATH = Path("data/llm_candidates_applied_llm_en.jsonl")
STAGE3_MASTER_TABLE_PATH = Path("data/table_applied_llm_en.jsonl")
