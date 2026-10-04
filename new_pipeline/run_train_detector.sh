#!/bin/bash
#SBATCH --job-name=train_detector
#SBATCH --partition=lrz-v100x2  
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=res_%j.log
#SBATCH -e log_%j.err 
#SBATCH --container-image=/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM/nvidia+pytorch+23.10-py3.sqsh
#SBATCH --container-mounts=/dss/dsshome1/01/ge65nus2:/dss/dsshome1/01/ge65nus2

PROJECT_ROOT="/dss/dsshome1/01/ge65nus2/projects/MultiLexNorm_LLM"
PIPELINE_ROOT="$PROJECT_ROOT/new_pipeline"

cd "$PIPELINE_ROOT"

source ~/miniconda3/bin/activate new_pipeline
export PYTHONPATH="$PIPELINE_ROOT:$PYTHONPATH"

LANGUAGES=("de" "hr" "id" "iden" "ja" "ko" "nl" "sl" "sr" "th" "vi")

for LANG in "${LANGUAGES[@]}"; do
  echo "=============================="
  echo "Running language: $LANG"
  echo "=============================="

  export LANGUAGE="$LANG"
  python -m src.prepare_detector_data

  python external/machamp/train.py \
  --dataset_configs models/machamp/configs/machamp_detector_"${LANG}".json \
  --parameters_config models/machamp/configs/machamp_params_detector.json \
  --model_dir models/machamp/detector_"${LANG}"_xlmr \
  --device 0

  python external/machamp/predict.py \
  models/machamp/detector_"${LANG}"_xlmr/model.pt \
  data/detector_train/detector_dev_"${LANG}".tsv \
  data/detector_output/detector_"${LANG}".confidence.out \
  --dataset detector_"${LANG}" \
  --device 0 \
  --topn 2

  python -m src.build_dictionary
  python -m src.apply_dictionary
  python -m src.build_llm_prompts

  echo "Finished language: $LANG"
done