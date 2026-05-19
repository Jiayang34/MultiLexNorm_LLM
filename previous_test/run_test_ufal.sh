#!/bin/bash
#SBATCH --job-name=test_ufal
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
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

cd $PROJECT_ROOT/previous_test/

source ~/miniconda3/bin/activate test_env
python test_ufal_validation.py
python test_ufal_test.py
