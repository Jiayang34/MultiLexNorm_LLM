#!/bin/bash
#SBATCH --job-name=test_qwen
#SBATCH --partition=lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=qwen_en_val_%j.out
#SBATCH --error=qwen_en_val_%j.err
#SBATCH --container-image=/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM/nvidia+pytorch+23.10-py3.sqsh
#SBATCH --container-mounts=/dss/dsshome1/01/ge65nus2:/dss/dsshome1/01/ge65nus2

PROJECT_ROOT="/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM"
PIPELINE_ROOT="$PROJECT_ROOT/our_pipeline1"
HF_MODEL_PATH="$PROJECT_ROOT/models/huggingface/Qwen3.5-9B"

source ~/miniconda3/bin/activate new_pipeline

cd "$PIPELINE_ROOT"

export PYTHONPATH="$PIPELINE_ROOT:${PYTHONPATH:-}"
export LANGUAGE="en"
export IS_VAL=true
export HF_MODEL_PATH="$HF_MODEL_PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LLM_DTYPE="float16"

echo "========================================"
echo "Qwen3.5 HF English validation test"
echo "Node: $(hostname)"
echo "Pipeline: $PIPELINE_ROOT"
echo "Model: $HF_MODEL_PATH"
echo "Time: $(date)"
echo "========================================"

nvidia-smi

echo "----- Environment check -----"
python - <<'PY'
import torch
import transformers
import huggingface_hub
import tokenizers

print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("transformers:", transformers.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("tokenizers:", tokenizers.__version__)

from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
print("qwen3_5:", CONFIG_MAPPING_NAMES.get("qwen3_5"))
PY

echo "----- Required files -----"
test -s "data/table_applied_dictionary_en_val.jsonl"
test -s "data/llm_candidates_en_val.jsonl"
test -s "data/llm_prompts_en_val.jsonl"
test -s "data/gold_en_val.jsonl"
test -s "$HF_MODEL_PATH/config.json"

ls -lh \
  data/table_applied_dictionary_en_val.jsonl \
  data/llm_candidates_en_val.jsonl \
  data/llm_prompts_en_val.jsonl \
  data/gold_en_val.jsonl

echo "----- Run LLM -----"
python -m src.run_llm \
  --model-path "$HF_MODEL_PATH" \
  --model-name "qwen3.5_9b_hf" \
  --dtype "$LLM_DTYPE"

echo "----- Evaluate -----"
python -m src.evaluate_pipeline

echo "----- Outputs -----"
ls -lh \
  data/llm_candidates_applied_llm_en_val.jsonl \
  data/table_applied_qwen3.5_9b_hf_en_val.jsonl \
  data/evaluation_summary_qwen3.5_9b_hf_en_val.json

echo "Done at $(date)"

