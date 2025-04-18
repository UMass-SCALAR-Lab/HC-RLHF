
import os, shutil, glob
import socket
import sys
import argparse
import time
import subprocess
from itertools import product

# C1 = [1,2,4,6]
# C2 = [1,2,4,6]
C1 = [4]
C2 = [2]
# THRESHOLDS = [-12, -9, -7, -4]
THRESHOLDS = [0]
# BETAS = [0.1, 0.25, 0.5, 1.0, 0.01, 0.05]
BETAS=[0.1]
SAFETY_DSET_SIZE = 100
FAIL_PROB = 0.1
NUM_SAMPLES_PER_PROMPT = 2
RLOO = True
SEEDS = list(range(30))

foldername = './temp/slurm_scripts/'
if not os.path.exists(foldername):
    os.makedirs(foldername, exist_ok=True)

slurm_files = []
partition = 'superpod-a100'
gpus = 1
days = 1
model_name = 'seldonian'
log_dir = f'logs_{model_name}/eval_sl/1000/'
# log_dir = 'logs/eval'

# actor_path = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_{model_name}/sft/pytorch_model.bin'
# rm_path = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_{model_name}/rm_bt/pytorch_model.bin'
# cm_path = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_{model_name}/cm_bt/pytorch_model.bin'

actor_path = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf/sft_qwen2/pytorch_model.bin'
rm_path = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf/rm_qwen2/pytorch_model.bin'
cm_path = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf/cm_bt_qwen2/pytorch_model.bin'


for ix, (seed, c1, c2, beta, threshold) in enumerate(product(SEEDS, C1, C2, BETAS, THRESHOLDS)):  
    
    # actor_path = f"/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_10k/rl_qwen2/temp-reinforce-rloo-10k-2-c1-{c1}-c2-{c2}-beta-{beta}"
    actor_path = f"/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_seldonian/1000/rloo-seldonian-1000-seed-{seed}/pytorch_model.bin"
    # def copy_init(path):
    #     source_dir = path
    #     target_dir = os.path.join(path, 'pytorch_model.bin')

    #     # Copy config.json
    #     shutil.copy(os.path.join(source_dir, "config.json"), target_dir)

    #     # Copy files starting with "tokenizer"
    #     for file in glob.glob(os.path.join(source_dir, "tokenizer*")):
    #         shutil.copy(file, target_dir)

    # copy_init(actor_path)
    # actor_path = os.path.join(actor_path, 'pytorch_model.bin')

    # if(RLOO and NUM_SAMPLES_PER_PROMPT>1):
    #     wandb_run_name = f"{model_name}-reinforce-bt-rloo-{NUM_SAMPLES_PER_PROMPT}-t_{threshold}-c1-{c1}-c2-{c2}-beta-{beta}"
    # else:
    #     wandb_run_name = f"{model_name}-reinforce-bt-t_{threshold}-c1-{c1}-c2-{c2}-beta-{beta}"

    if(RLOO and NUM_SAMPLES_PER_PROMPT>1):
        wandb_run_name = f"rloo-seldonian-1000-seed-{ix}"
    else:
        wandb_run_name = f"seldonian-1000-seed-{ix}"

    # out_dir = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_{model_name}/1000/{wandb_run_name}'
    out_dir = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf/tmp'

    # out_dir = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_{model_name}/rl_bt_thresh_lmbda_veryhigh/{wandb_run_name}'
    # out_dir = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_{model_name}/rl_bt_multi_gpu_b16_thresh/{wandb_run_name}'
    # out_dir = f'/scratch3/workspace/ychittepu_umass_edu-cql/safe_rlhf_bt/reinforce_safe/{wandb_run_name}'

    slurm_file = os.path.join(foldername, f"{model_name}-{wandb_run_name}")
    job_name =  f"{wandb_run_name}"

    # f"#SBATCH --mail-user=mailto:ychittepu@umass.edu\n" \
    # f"#SBATCH --mail-type=ALL\n" \
    # f"export WANDB_API_KEY=351d5626da0ea4ccba23cb78278b35b4aaf7cb3c\n" \

    if(RLOO and NUM_SAMPLES_PER_PROMPT > 1):
        slurm_script = f"#!/bin/bash\n" \
                    f"#SBATCH --time={days}-00:00:00\n" \
                    f"#SBATCH --nodes=1\n" \
                    f"#SBATCH --gpus={gpus}\n" \
                    f"#SBATCH --partition={partition}\n" \
                    f"#SBATCH --mem=240G\n" \
                    f"#SBATCH --job-name={wandb_run_name}-safe-rlhf\n" \
                    f"#SBATCH --output=./{log_dir}/{wandb_run_name}-safe-rlhf-%j.out\n" \
                    f"#SBATCH --error=./{log_dir}/{wandb_run_name}-safe-rlhf-%j.err\n" \
                    f"source ~/.bashrc\n" \
                    f"module load cuda/12.1\n" \
                    f"module load gcc/9.4.0\n" \
                    f"conda activate safe-rlhf\n" \
                    f"export HF_HOME=/scratch3/workspace/ychittepu_umass_edu-cql/cache/\n" \
                    f"bash scripts/reinforce-ci-lag.sh --actor_model_name_or_path {actor_path} --reward_model_name_or_path {rm_path} --cost_model_name_or_path {cm_path} --output_dir {out_dir} --log_run_name {wandb_run_name} --kl_coeff {beta} --seed {seed} --threshold {threshold} --fail_prob {FAIL_PROB} --safety_dataset_size {SAFETY_DSET_SIZE} --students_c1 {c1} --students_c2 {c2} --num_samples_per_prompt {NUM_SAMPLES_PER_PROMPT} --rloo {RLOO} \n"
    else:
        slurm_script = f"#!/bin/bash\n" \
                f"#SBATCH --time={days}-00:00:00\n" \
                f"#SBATCH --nodes=1\n" \
                f"#SBATCH --gpus={gpus}\n" \
                f"#SBATCH --partition={partition}\n" \
                f"#SBATCH --mem=240G\n" \
                f"#SBATCH --mail-user=mailto:ychittepu@umass.edu\n" \
                f"#SBATCH --mail-type=ALL\n" \
                f"#SBATCH --job-name={wandb_run_name}-safe-rlhf\n" \
                f"#SBATCH --output=./{log_dir}/{wandb_run_name}-safe-rlhf-%j.out\n" \
                f"#SBATCH --error=./{log_dir}/{wandb_run_name}-safe-rlhf-%j.err\n" \
                f"source ~/.bashrc\n" \
                f"module load cuda/12.1\n" \
                f"module load gcc/9.4.0\n" \
                f"conda activate safe-rlhf\n" \
                f"export HF_HOME=/scratch3/workspace/ychittepu_umass_edu-cql/cache/\n" \
                f"export WANDB_API_KEY=351d5626da0ea4ccba23cb78278b35b4aaf7cb3c\n" \
                f"bash scripts/reinforce-ci-lag.sh --actor_model_name_or_path {actor_path} --reward_model_name_or_path {rm_path} --cost_model_name_or_path {cm_path} --output_dir {out_dir} --log_run_name {wandb_run_name} --kl_coeff {beta} --seed {seed} --threshold {threshold} --fail_prob {FAIL_PROB} --safety_dataset_size {SAFETY_DSET_SIZE} --students_c1 {c1} --students_c2 {c2} --rloo {False} \n" \
    
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


