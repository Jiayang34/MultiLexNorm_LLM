#!/bin/bash
#SBATCH --job-name=threshold_search
#SBATCH --partition=lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=threshold_search_%j.out
#SBATCH --error=threshold_search_%j.err
#SBATCH --container-image=/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM/nvidia+pytorch+23.10-py3.sqsh
#SBATCH --container-mounts=/dss/dsshome1/01/ge65nus2:/dss/dsshome1/01/ge65nus2

PROJECT_ROOT="/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM"
PIPELINE_ROOT="$PROJECT_ROOT/our_pipeline_lrz"

MODEL_NAME="gemma3_27b_it"
HF_MODEL_PATH="$PROJECT_ROOT/models/huggingface/gemma-3-27b-it"
DTYPE="float16"

LANGUAGES=("de" "en" "hr" "id" "iden" "ja" "ko" "nl" "sl" "sr" "th" "vi")

source ~/miniconda3/bin/activate new_pipeline

cd "$PIPELINE_ROOT"

export PYTHONPATH="$PIPELINE_ROOT:${PYTHONPATH:-}"
export HF_MODEL_NAME="$MODEL_NAME"
export HF_MODEL_PATH="$HF_MODEL_PATH"
export LLM_DTYPE="$DTYPE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset IS_VAL
unset IS_OPTIMIZED
unset DETECTOR_THRESHOLD
unset ENTROPY_THRESHOLD
unset THRESHOLD_SEARCH_RESULTS_PATH

echo "============================================================"
echo "Threshold search"
echo "Pipeline: $PIPELINE_ROOT"
echo "Model name: $MODEL_NAME"
echo "Model path: $HF_MODEL_PATH"
echo "Dtype: $DTYPE"
echo "Languages: ${LANGUAGES[*]}"
echo "Started: $(date)"
echo "============================================================"

test -s "$HF_MODEL_PATH/config.json"

python -m src.search_thresholds \
    --languages "${LANGUAGES[@]}" \
    --model-path "$HF_MODEL_PATH" \
    --model-name "$MODEL_NAME" \
    --dtype "$DTYPE" \
    --output "data/threshold_search_results_${MODEL_NAME}.jsonl" \
    --tmp-root "data/thresold_search_tmp" \
    --resume

echo "Finished threshold search at: $(date)"
ls -lh "data/threshold_search_results_${MODEL_NAME}.jsonl"
