#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=superpod-a100
#SBATCH --mem=240G
#SBATCH --mail-user=mailto:ychittepu@umass.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=cm-qwen2.5-safe-rlhf
#SBATCH --output=./logs_qwen2.5/cm-qwen2.5-safe-rlhf-%j.out
#SBATCH --error=./logs_qwen2.5/cm-qwen2.5-safe-rlhf-%j.err
source ~/.bashrc
module load cuda/12.1
module load gcc/9.4.0   
conda activate safe-rlhf
export HF_HOME=/scratch3/workspace/ychittepu_umass_edu-cql/cache/
export WANDB_API_KEY=351d5626da0ea4ccba23cb78278b35b4aaf7cb3c
bash scripts/cost-model.sh \
    --model_name_or_path /scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_qwen2.5/sft/pytorch_model.bin \
    --output_dir /scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_qwen2.5/cm_bt \
    --log_run_name safe-rlhf-cm-bt-qwen2.5

# bash scripts/cost-model.sh \
#     --model_name_or_path /scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_llama3.2/sft/pytorch_model.bin \
#     --output_dir /scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_llama3.2/cm_bt \
#     --log_run_name safe-rlhf-cm-bt-llama3.2


# bash scripts/cost-model.sh \
#     --model_name_or_path /scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_qwen2.5/sft/pytorch_model.bin \
#     --output_dir /scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_qwen2.5/cm_bt \
#     --log_run_name safe-rlhf-cm-bt-qwen2.5