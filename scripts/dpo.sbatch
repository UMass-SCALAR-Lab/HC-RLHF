#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=superpod-a100
#SBATCH --mem=240G
#SBATCH --mail-user=mailto:ychittepu@umass.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=dpo-margin-ground-llama3.1-8b-uf
#SBATCH --output=./logs_pop_8b_2/dpo-margin-ground-llama3.1-8b-%j.out
#SBATCH --error=./logs_pop_8b_2/dpo-margin-ground-llama3.1-8b-%j.err
source ~/.bashrc
module load cuda/12.1
module load gcc/9.4.0   
conda activate safe-rlhf
export HF_HOME=/scratch3/workspace/ychittepu_umass_edu-cql/cache/
export WANDB_API_KEY=351d5626da0ea4ccba23cb78278b35b4aaf7cb3c
bash scripts/dpo_ground.sh \
    --model_name_or_path /scratch3/workspace/ychittepu_umass_edu-cql/pop_uf_llama3.1_8b/sft/pytorch_model.bin \
    --output_dir /scratch/workspace/ychittepu_umass_edu-pop/pop_uf_llama3.1_8b_2/dpo_margin_ground \
    --log_run_name dpo-margin-ground-llama3.1-8b-uf-2