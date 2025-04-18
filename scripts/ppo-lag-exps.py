
import os, shutil, glob
import socket
import sys
import argparse
import time
import subprocess
from itertools import product

THRESHOLDS = [-12, -9]

foldername = './temp/slurm_scripts/'
if not os.path.exists(foldername):
    os.makedirs(foldername, exist_ok=True)

slurm_files = []
partition = 'superpod-a100'
gpus = 4
days = 2
model_name = 'llama3.2'
log_dir = f'logs_{model_name}/'
# log_dir = 'logs/eval'

actor_path = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_{model_name}/sft/pytorch_model.bin'
rm_path = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_{model_name}/rm_bt/pytorch_model.bin'
cm_path = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_{model_name}/cm_bt/pytorch_model.bin'

# actor_path = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf/sft_qwen2/pytorch_model.bin'
# rm_path = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf/rm_qwen2/pytorch_model.bin'
# cm_path = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf/cm_bt_qwen2/pytorch_model.bin'


for threshold in THRESHOLDS:  
    
    wandb_run_name = f"{model_name}-safe-baseline-t_{threshold}"

    out_dir = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_{model_name}/safe_baseline_bt_thresh_lmbda_veryhigh/{wandb_run_name}'
    # out_dir = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_bt/reinforce_safe/{wandb_run_name}'

    slurm_file = os.path.join(foldername, f"{model_name}-{wandb_run_name}")
    job_name =  f"{wandb_run_name}"

    slurm_script = f"#!/bin/bash\n" \
                f"#SBATCH --time={days}-00:00:00\n" \
                f"#SBATCH --nodes=1\n" \
                f"#SBATCH --gpus={gpus}\n" \
                f"#SBATCH --partition={partition}\n" \
                f"#SBATCH --mem=240G\n" \
                f"#SBATCH --mail-user=mailto:ychittepu@umass.edu\n" \
                f"#SBATCH --mail-type=ALL\n" \
                f"#SBATCH --job-name={wandb_run_name}-safe-rlhf\n" \
                f"#SBATCH --output=./{log_dir}/{wandb_run_name}-%j.out\n" \
                f"#SBATCH --error=./{log_dir}/{wandb_run_name}-%j.err\n" \
                f"source ~/.bashrc\n" \
                f"module load cuda/12.1\n" \
                f"module load gcc/9.4.0\n" \
                f"conda activate safe-rlhf\n" \
                f"export HF_HOME=/scratch3/workspace/ychittepu_umass_edu-cql/cache/\n" \
                f"export WANDB_API_KEY=351d5626da0ea4ccba23cb78278b35b4aaf7cb3c\n" \
                f"bash scripts/ppo-lag.sh --actor_model_name_or_path {actor_path} --reward_model_name_or_path {rm_path} --cost_model_name_or_path {cm_path} --output_dir {out_dir} --log_run_name {wandb_run_name} --threshold {threshold} \n"
    
    sfile = open(slurm_file, 'w')
    sfile.write(slurm_script)
    sfile.close()
    subprocess.check_call(f"chmod 755 {slurm_file}",shell=True)
    slurm_files.append(slurm_file)

if __name__ == "__main__":

    user_input = input("Are you sure to kick off the batch script: y/n \n")
    if(user_input != 'y'):
        raise ValueError("Error: User did not confirm with 'y'...")
    for slurm_file in slurm_files: 
        os.system("sbatch %s" % slurm_file)


