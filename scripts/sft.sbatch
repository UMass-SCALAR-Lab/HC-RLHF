#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=superpod-a100
#SBATCH --mem=240G
#SBATCH --mail-user=mailto:ychittepu@umass.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=sft-llama3.1-8b-uf
#SBATCH --output=./logs_pop/sft-llama3.1-8b-%j.out
#SBATCH --error=./logs_pop/sft-llama3.1-8b-%j.err
source ~/.bashrc
module load cuda/12.1
module load gcc/9.4.0   
conda activate safe-rlhf
export HF_HOME=/scratch3/workspace/ychittepu_umass_edu-cql/cache/
export WANDB_API_KEY=351d5626da0ea4ccba23cb78278b35b4aaf7cb3c
bash scripts/sft.sh \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --output_dir /scratch3/workspace/ychittepu_umass_edu-cql/pop_uf_llama3.1_8b/sft \
    --log_run_name sft-llama3.1-8b-uf