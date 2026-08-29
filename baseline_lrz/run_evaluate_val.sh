#!/bin/bash
#SBATCH --job-name=evaluate_val
#SBATCH --partition=lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=gemma3_all_val_%j.out
#SBATCH --error=gemma3_all_val_%j.err
#SBATCH --container-image=/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM/nvidia+pytorch+23.10-py3.sqsh
#SBATCH --container-mounts=/dss/dsshome1/01/ge65nus2:/dss/dsshome1/01/ge65nus2

PROJECT_ROOT="/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM"
PIPELINE_ROOT="$PROJECT_ROOT/baseline_lrz"

MODEL_NAME="gemma3_27b_it"
HF_MODEL_PATH="$PROJECT_ROOT/models/huggingface/gemma-3-27b-it"
DTYPE="float16"

source ~/miniconda3/bin/activate new_pipeline

cd "$PIPELINE_ROOT"

export PYTHONPATH="$PIPELINE_ROOT:${PYTHONPATH:-}"
export IS_VAL=true
export IS_OPTIMIZED=false
export HF_MODEL_NAME="$MODEL_NAME"
export LLM_DTYPE="$DTYPE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset DETECTOR_THRESHOLD
unset ENTROPY_THRESHOLD
unset THRESHOLD_SEARCH_RESULTS_PATH


LANGUAGES=("de" "en" "hr" "id" "iden" "ja" "ko" "nl" "sl" "sr" "th" "vi")

for LANG in "${LANGUAGES[@]}"; do
    export LANGUAGE="$LANG"

    echo
    echo "============================================================"
    echo "Running language: $LANG"
    echo "Started: $(date)"
    echo "============================================================"

    python -m src.apply_dictionary
    python -m src.build_llm_prompts

    PROMPTS="data/${LANG}/llm_prompts_${LANG}_val.jsonl"
    MASTER_TABLE="data/${LANG}/table_applied_dictionary_${LANG}_val.jsonl"
    CANDIDATES="data/${LANG}/llm_candidates_${LANG}_val.jsonl"
    GOLD="data/${LANG}/gold_${LANG}_val.jsonl"

    test -s "$PROMPTS"
    test -s "$MASTER_TABLE"
    test -s "$CANDIDATES"
    test -s "$GOLD"

    echo "Prompt count for $LANG: $(wc -l < "$PROMPTS")"

    python -m src.run_llm \
        --model-path "$HF_MODEL_PATH" \
        --model-name "$MODEL_NAME" \
        --dtype "$DTYPE"

    python -m src.evaluate_pipeline

    echo "Finished language: $LANG at $(date)"
    ls -lh \
        "data/${LANG}/llm_candidates_applied_llm_${LANG}_val.jsonl" \
        "data/${LANG}/table_applied_${MODEL_NAME}_${LANG}_val.jsonl" \
        "data/${LANG}/evaluation_summary_${MODEL_NAME}_${LANG}_val.json"
done

echo
echo "All languages finished at: $(date)"
