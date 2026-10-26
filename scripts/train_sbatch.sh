#!/usr/bin/env bash
# Submit from the repository root after creating logs/ and activating your environment.
#SBATCH --job-name=hinglish-ner
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
python -m src.train_flat_ner --model_name xlm-roberta-base \
  --output_dir outputs/xlmr-flat-gpu --batch_size 8 --num_train_epochs 5 \
  --gradient_accumulation_steps 4 --fp16
