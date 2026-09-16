#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --gres=gpu:rtx_3090:4
#SBATCH --partition gpu
#SBATCH --time=1:00:00
#SBATCH --output=experiment_%j.out
#SBATCH --account=spai040604

source ../initMamba.sh
conda activate local_llm
echo "Job Started"
python src/lang_extract.py --chunk 0 --chunk-size 100