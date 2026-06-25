from pathlib import Path
# Config file

# Change language
import os
LANGUAGE = os.environ.get("LANGUAGE", "en")
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
GOLD_PATH = Path("data") / f"gold_{LANGUAGE}{RUN_SUFFIX}.jsonl"

# MaChAmp detector model
MACHAMP_DATASET_CONFIG_PATH = Path(
    f"models/machamp/configs/machamp_detector_{LANGUAGE}{RUN_SUFFIX}.json"
)
MACHAMP_PARAMS_CONFIG_PATH = Path("models/machamp/configs/machamp_params_detector.json")
DETECTOR_MODEL_DIR = Path(f"models/machamp/detector_{LANGUAGE}_xlmr")
DETECTOR_MODEL_PATH = DETECTOR_MODEL_DIR / "model.pt"
DETECTOR_DEVICE = "0"

# Dictionary
DICTIONARY_PATH = Path(f"data/dictionary_{LANGUAGE}.jsonl")
ENTROPY_THRESHOLD = 0.5

# Selected language data -> DEV_RATIO -> DEV data
DEV_PATH = Path("data") / f"dev_raw_norm_{LANGUAGE}.jsonl"

# Stage 2: Detector -> Dictionary lookup
KEEP_LABEL = "O"
NORM_1WORD_LABEL = "NORM_1WORD"
NORM_2WORD_LABEL = "NORM_2WORD"
NORM_3PLUS_LABEL = "NORM_3PLUS"
NORM_LABELS = {
    NORM_1WORD_LABEL,
    NORM_2WORD_LABEL,
    NORM_3PLUS_LABEL,
}
DETECTOR_THRESHOLD = 0.5
DETECTOR_CONFIDENCE_PATH = Path(
    f"data/detector_output/detector_{LANGUAGE}{RUN_SUFFIX}.confidence.out"
)
STAGE2_OUTPUT_PATH = Path(f"data/table_applied_dictionary_{LANGUAGE}{RUN_SUFFIX}.jsonl")
STAGE3_LLM_CANDIDATES_PATH = Path(f"data/llm_candidates_{LANGUAGE}{RUN_SUFFIX}.jsonl")
STAGE3_LLM_PROMPTS_PATH = Path(f"data/llm_prompts_{LANGUAGE}{RUN_SUFFIX}.jsonl")
NUM_LLM_SHOTS = 8

# Stage 3: LLM inference
HF_MODEL_PATH = Path(
    os.environ.get(
        "HF_MODEL_PATH",
        "/dss/dsshome1/01/ge65nus2/projects/"
        "MultiLexNorm_LLM/models/huggingface/Qwen3.5-9B",
    )
)
HF_MODEL_NAME = "qwen3.5_9b"
LLM_DTYPE = os.environ.get("LLM_DTYPE", "float16")
safe_name = HF_MODEL_NAME
# transfer to safer save name
STAGE3_LLM_APPLIED_PATH = Path(
    f"data/llm_candidates_applied_llm_{LANGUAGE}{RUN_SUFFIX}.jsonl"
)
STAGE3_MASTER_TABLE_PATH = Path(f"data/table_applied_{safe_name}_{LANGUAGE}{RUN_SUFFIX}.jsonl")

# Evaluation
EVALUATION_SUMMARY_PATH = Path(f"data/evaluation_summary_{safe_name}_{LANGUAGE}{RUN_SUFFIX}.json")
