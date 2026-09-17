#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --nodes=1
#SBATCH --exclude=bp1-gpu[002-003,007-008,013-015,019-020,024-027]
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --output=experiment_%j.out
#SBATCH --account=spai04060

source ../initMamba.sh
conda activate local_llm
echo "Job Started"
python src/lang_extract.py