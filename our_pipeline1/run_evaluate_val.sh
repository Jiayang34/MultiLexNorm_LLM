#!/bin/bash
#SBATCH --job-name=val_pipeline
#SBATCH --partition=lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=res_%j.log
#SBATCH -e log_%j.err
#SBATCH --container-image=/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM/nvidia+pytorch+23.10-py3.sqsh
#SBATCH --container-mounts=/dss/dsshome1/01/ge65nus2:/dss/dsshome1/01/ge65nus2

PROJECT_ROOT="/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM"
PIPELINE_ROOT="$PROJECT_ROOT/our_pipeline1"
OLLAMA_ROOT="$PROJECT_ROOT/new_pipeline/tools/ollama"
export OLLAMA_LLM_LIBRARY=cuda_v13
export OLLAMA_LIBRARY_PATH="$OLLAMA_ROOT/ollama_package/lib:$OLLAMA_ROOT/ollama_package/lib/cuda_v13:$OLLAMA_ROOT"
export LD_LIBRARY_PATH="$OLLAMA_ROOT/ollama_package/lib:$OLLAMA_ROOT/ollama_package/lib/cuda_v13:${LD_LIBRARY_PATH:-}"

cd "$PIPELINE_ROOT"

source ~/miniconda3/bin/activate new_pipeline
export PYTHONPATH="$PIPELINE_ROOT:$PYTHONPATH"
export IS_VAL=true
export VAL_SPLIT_NAME="validation"

export PATH="$OLLAMA_ROOT:$PATH"
export OLLAMA_MODELS="$OLLAMA_ROOT/ollama_models"
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_URL="http://127.0.0.1:11434/api/chat"
export OLLAMA_MODEL="qwen3.5:9b"

which ollama
ollama --version

ollama serve > "$OLLAMA_ROOT/log/ollama_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!

cleanup() {
  kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

sleep 15

curl -s http://127.0.0.1:11434/api/tags
ollama list

LANGUAGES=("de" "en" "hr" "id" "iden" "ja" "ko" "nl" "sl" "sr" "th" "vi")

echo "---------------------Validation pipeline---------------------"
echo "Inference with: $OLLAMA_MODEL"
for LANG in "${LANGUAGES[@]}"; do
  echo "=============================="
  echo "Running validation language: $LANG"
  echo "=============================="

  export LANGUAGE="$LANG"

  test -f "models/machamp/detector_${LANG}_xlmr/model.pt"
  test -f "data/dictionary_${LANG}.jsonl"

  python -m src.execute_run_pipeline
  python -m src.evaluate_pipeline

  echo "Finished validation language: $LANG"
done
