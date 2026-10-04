#!/bin/bash
#SBATCH --job-name=train_detector
#SBATCH --partition=lrz-hgx-a100-80x4 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=res_%j.log
#SBATCH -e log_%j.err 
#SBATCH --container-image=/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM/nvidia+pytorch+23.10-py3.sqsh
#SBATCH --container-mounts=/dss/dsshome1/01/ge65nus2:/dss/dsshome1/01/ge65nus2

PROJECT_ROOT="/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM"
PIPELINE_ROOT="$PROJECT_ROOT/new_pipeline"
OLLAMA_ROOT="$PROJECT_ROOT/new_pipeline/tools/ollama"

cd "$PIPELINE_ROOT"

source ~/miniconda3/bin/activate new_pipeline
export PYTHONPATH="$PIPELINE_ROOT:$PYTHONPATH"
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


echo "Inference with: $OLLAMA_MODEL"
for LANG in "${LANGUAGES[@]}"; do
  echo "=============================="
  echo "Running language: $LANG"
  echo "=============================="

  export LANGUAGE="$LANG"
  
  test -f "data/${LANG}/llm_prompts_${LANG}.jsonl"
  test -f "data/${LANG}/llm_candidates_${LANG}.jsonl"
  test -f "data/${LANG}/table_applied_dictionary_${LANG}.jsonl"
  
  python -m src.run_llm \
    --model "$OLLAMA_MODEL" \
    --url "$OLLAMA_URL"

  python -m src.evaluate_pipeline

  echo "Finished language: $LANG"
done
