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
PIPELINE_ROOT="$PROJECT_ROOT/new_pipeline"
OLLAMA_ROOT="$PROJECT_ROOT/new_pipeline/tools/ollama"

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

BASE_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

set_ollama_backend() {
  local backend="$1"

  export OLLAMA_LLM_LIBRARY="$backend"
  export OLLAMA_LIBRARY_PATH="$OLLAMA_ROOT/ollama_package/lib:$OLLAMA_ROOT/ollama_package/lib/$backend:$OLLAMA_ROOT/$backend:$OLLAMA_ROOT"
  export LD_LIBRARY_PATH="$OLLAMA_ROOT/ollama_package/lib:$OLLAMA_ROOT/ollama_package/lib/$backend:$OLLAMA_ROOT/$backend:${BASE_LD_LIBRARY_PATH}"
}

test_ollama_backend() {
  local backend="$1"
  local smoke_log="$OLLAMA_ROOT/log/ollama_${SLURM_JOB_ID}_${backend}_smoke.log"
  local smoke_pid
  local status

  if [ ! -d "$OLLAMA_ROOT/ollama_package/lib/$backend" ] && [ ! -d "$OLLAMA_ROOT/$backend" ]; then
    echo "Skipping missing Ollama backend: $backend"
    return 1
  fi

  echo "Testing Ollama backend: $backend"
  set_ollama_backend "$backend"

  "$OLLAMA_ROOT/ollama" serve > "$smoke_log" 2>&1 &
  smoke_pid=$!

  sleep 15

  python - <<PY
import requests

payload = {
    "model": "${OLLAMA_MODEL}",
    "messages": [{"role": "user", "content": "Return only [\\"ok\\"]."}],
    "stream": False,
    "think": False,
    "options": {"temperature": 0, "num_predict": 8},
}

response = requests.post("${OLLAMA_URL}", json=payload, timeout=120)
response.raise_for_status()
print(response.json()["message"]["content"])
PY
  status=$?

  kill "$smoke_pid" 2>/dev/null || true
  wait "$smoke_pid" 2>/dev/null || true
  sleep 3

  if [ "$status" -eq 0 ]; then
    echo "Ollama backend smoke test passed: $backend"
  else
    echo "Ollama backend smoke test failed: $backend"
    tail -n 80 "$smoke_log" || true
  fi

  return "$status"
}

SELECTED_OLLAMA_BACKEND=""
for BACKEND in cuda_v13 cuda_v12; do
  if test_ollama_backend "$BACKEND"; then
    SELECTED_OLLAMA_BACKEND="$BACKEND"
    break
  fi
done

if [ -z "$SELECTED_OLLAMA_BACKEND" ]; then
  echo "No working Ollama CUDA backend found on this node."
  exit 1
fi

echo "Using Ollama backend: $SELECTED_OLLAMA_BACKEND"
set_ollama_backend "$SELECTED_OLLAMA_BACKEND"

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
