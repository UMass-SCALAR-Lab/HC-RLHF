#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=superpod-a100
#SBATCH --mem=240G
#SBATCH --mail-user=mailto:ychittepu@umass.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=ppo-rlhf
#SBATCH --output=./logs/ppo-rlhf-%j.out
#SBATCH --error=./logs/ppo-rlhf-%j.err
source ~/.bashrc
module load cuda/12.1
module load gcc/9.4.0
conda activate safe-rlhf
export HF_HOME=/scratch3/workspace/ychittepu_umass_edu-cql/cache/
export WANDB_API_KEY=351d5626da0ea4ccba23cb78278b35b4aaf7cb3c
bash scripts/ppo.sh \
    --actor_model_name_or_path /scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf/sft_qwen2/pytorch_model.bin \
    --reward_model_name_or_path /scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf/rm_qwen2/pytorch_model.bin \
    --output_dir /scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf/ppo_1gpu_qwen2 \
    --log_run_name ppo-1gpu-qwen2