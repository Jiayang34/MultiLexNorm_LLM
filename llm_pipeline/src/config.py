from pathlib import Path
# Config file

# All Data -> EN Training Data -> DEV_RATIO -> TRAIN Data (90%) & DEV Data (10%)
DATASET_NAME = "weerayut/multilexnorm2026-dev-pub"
SPLIT_NAME = "train"
LANGUAGE = "en"
DEV_RATIO = 0.1
SEED = 42

# Data for Machamp
MACHAMP_DIR = Path("data/machamp")
MACHAMP_TRAIN_PATH = MACHAMP_DIR / f"detector_train_{LANGUAGE}.tsv"
MACHAMP_DEV_PATH = MACHAMP_DIR / f"detector_dev_{LANGUAGE}.tsv"

# Dictionary
DICTIONARY_PATH = Path("data/dictionary_en.jsonl")

# EN Training Data -> DEV_RATIO -> DEV Data
DEV_PATH = Path("data") / f"dev_raw_norm_{LANGUAGE}.jsonl"

# Stage 2: Detector -> Dictionary lookup
KEEP_LABEL = "O"
NORM_LABEL = "NORM"
DETECTOR_CONFIDENCE_PATH = Path(
    "models/machamp/detector_en_xlmr_0/detector_en.confidence.out"
)
STAGE2_OUTPUT_PATH = Path("data/stage2_dictionary_applied_en.jsonl")
STAGE3_LLM_CANDIDATES_PATH = Path("data/stage3_llm_candidates_en.jsonl")
