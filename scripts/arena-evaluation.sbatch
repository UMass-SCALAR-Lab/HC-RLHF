#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=superpod-a100
#SBATCH --mem=240G
#SBATCH --mail-user=mailto:ychittepu@umass.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=sft_vs_baseline_eval
#SBATCH --output=./logs/eval/sft_vs_baseline_eval-%j.out
#SBATCH --error=./logs/eval/sft_vs_baseline_eval-%j.err
source ~/.bashrc
module load cuda/12.1
module load gcc/9.4.0
conda activate safe-rlhf
export HF_HOME=/scratch3/workspace/ychittepu_umass_edu-cql/cache/
export WANDB_API_KEY=351d5626da0ea4ccba23cb78278b35b4aaf7cb3c
bash scripts/arena-evaluation.sh \
    --red_corner_model_name_or_path /scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_qwen2.5/sft/pytorch_model.bin/ \
    --blue_corner_model_name_or_path /scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_qwen2.5/safe_baseline_bt/pytorch_model.bin/ \
    --reward_model_name_or_path /scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_qwen2.5/rm_bt/pytorch_model.bin/ \
    --cost_model_name_or_path /scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_qwen2.5/cm_bt/pytorch_model.bin/ \
    --output_dir /scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_qwen2.5/model_eval/sft_vs_baseline