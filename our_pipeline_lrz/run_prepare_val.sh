#!/bin/bash
#SBATCH --job-name=prepare_val
#SBATCH --partition=lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=prepare_val_%j.out
#SBATCH --error=prepare_val_%j.err
#SBATCH --container-image=/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM/nvidia+pytorch+23.10-py3.sqsh
#SBATCH --container-mounts=/dss/dsshome1/01/ge65nus2:/dss/dsshome1/01/ge65nus2


PROJECT_ROOT="/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM"
PIPELINE_ROOT="$PROJECT_ROOT/our_pipeline_lrz"

source ~/miniconda3/bin/activate new_pipeline

cd "$PIPELINE_ROOT"

export PYTHONPATH="$PIPELINE_ROOT:${PYTHONPATH:-}"
export IS_VAL=true
export VAL_SPLIT_NAME="validation"

LANGUAGES=("de" "en" "hr" "id" "iden" "ja" "ko" "nl" "sl" "sr" "th" "vi")

echo "Node: $(hostname)"
echo "Preparing validation inputs through the LLM-prompt stage."

for LANG in "${LANGUAGES[@]}"; do
  export LANGUAGE="$LANG"
  echo "========================================"
  echo "Preparing validation language: $LANG"
  echo "========================================"

  python -m src.prepare_validation_data

    python external/machamp/predict.py \
        "models/machamp/detector_${LANG}_xlmr/model.pt" \
        "data/detector_train/detector_val_${LANG}.tsv" \
        "data/detector_output/detector_${LANG}_val.confidence.out" \
        --dataset "detector_${LANG}" \
        --device 0 \
        --topn 4

    python -m src.apply_dictionary
    python -m src.build_llm_prompts
done
