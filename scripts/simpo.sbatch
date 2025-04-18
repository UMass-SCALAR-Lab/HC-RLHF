#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=superpod-a100
#SBATCH --mem=240G
#SBATCH --mail-user=mailto:ychittepu@umass.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=simpo-margin-ground-llama3.2-3b-uf
#SBATCH --output=./logs_pop/simpo-margin-ground-llama3.2-3b-%j.out
#SBATCH --error=./logs_pop/simpo-margin-ground-llama3.2-3b-%j.err
source ~/.bashrc
module load cuda/12.1
module load gcc/9.4.0   
conda activate safe-rlhf
export HF_HOME=/scratch3/workspace/ychittepu_umass_edu-cql/cache/
export WANDB_API_KEY=351d5626da0ea4ccba23cb78278b35b4aaf7cb3c
bash scripts/simpo.sh \
    --model_name_or_path /scratch3/workspace/ychittepu_umass_edu-cql/pop_uf_llama3.2/sft/pytorch_model.bin \
    --output_dir /scratch/workspace/ychittepu_umass_edu-pop/pop_uf_llama3.2/simpo_margin_ground \
    --log_run_name simpo-margin-ground-llama3.2-3b-uf \