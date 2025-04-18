#!/usr/bin/env bash
#
# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

ACTOR_MODEL_NAME_OR_PATH="PKU-Alignment/alpaca-7b-reproduced"
REWARD_MODEL_NAME_OR_PATH="${ROOT_DIR}/output/rm"
COST_MODEL_NAME_OR_PATH="${ROOT_DIR}/output/cm"
unset {REWARD,COST}_CRITIC_MODEL_NAME_OR_PATH
OUTPUT_DIR="${ROOT_DIR}/output/ppo-lag"
unset HOSTFILE
ZERO_STAGE=2
OFFLOAD="none"
LOG_RUN_NAME="default_reinforce_run"
NUM_RETURN_SEQUENCES=1
KL_COEFF=0.1
THRESHOLD=0.
SEED=42

# Default values for new arguments
FAIL_PROB=0.1
SAFETY_DATASET_SIZE=1000
STUDENTS_C1=1
STUDENTS_C2=1
NUM_SAMPLES_PER_PROMPT=1
RLOO=False

while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--actor_model_name_or_path)
			ACTOR_MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--actor_model_name_or_path=*)
			ACTOR_MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--reward_model_name_or_path)
			REWARD_MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--reward_model_name_or_path=*)
			REWARD_MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--reward_critic_model_name_or_path)
			REWARD_CRITIC_MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--reward_critic_model_name_or_path=*)
			REWARD_CRITIC_MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--cost_model_name_or_path)
			COST_MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--cost_model_name_or_path=*)
			COST_MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--cost_critic_model_name_or_path)
			COST_CRITIC_MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--cost_critic_model_name_or_path=*)
			COST_CRITIC_MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--kl_coeff)
			KL_COEFF="$1"
			shift
			;;
		--kl_coeff=*)
			KL_COEFF="${arg#*=}"
			;;
		--seed)
			SEED="$1"
			shift
			;;
		--seed=*)
			SEED="${arg#*=}"
			;;
		--threshold)
			THRESHOLD="$1"
			shift
			;;
		--threshold=*)
			THRESHOLD="${arg#*=}"
			;;
		--output_dir)
			OUTPUT_DIR="$1"
			shift
			;;
		--output_dir=*)
			OUTPUT_DIR="${arg#*=}"
			;;
		--hostfile)
			HOSTFILE="$1"
			shift
			;;
		--hostfile=*)
			HOSTFILE="${arg#*=}"
			;;
		--zero_stage)
			ZERO_STAGE="$1"
			shift
			;;
		--zero_stage=*)
			ZERO_STAGE="${arg#*=}"
			;;
		--offload)
			OFFLOAD="$1"
			shift
			;;
		--offload=*)
			OFFLOAD="${arg#*=}"
			;;
		--fail_prob)
            FAIL_PROB="$1"
            shift
            ;;
        --fail_prob=*)
            FAIL_PROB="${arg#*=}"
            ;;
        --safety_dataset_size)
            SAFETY_DATASET_SIZE="$1"
            shift
            ;;
        --safety_dataset_size=*)
            SAFETY_DATASET_SIZE="${arg#*=}"
            ;;
        --students_c1)
            STUDENTS_C1="$1"
            shift
            ;;
        --students_c1=*)
            STUDENTS_C1="${arg#*=}"
            ;;
        --students_c2)
            STUDENTS_C2="$1"
            shift
            ;;
        --students_c2=*)
            STUDENTS_C2="${arg#*=}"
            ;;
        --num_samples_per_prompt)
            NUM_SAMPLES_PER_PROMPT="$1"
            shift
            ;;
        --num_samples_per_prompt=*)
            NUM_SAMPLES_PER_PROMPT="${arg#*=}"
            ;;
        --rloo)
            RLOO="$1"
            shift
            ;;
        --rloo=*)
            RLOO="${arg#*=}"
            ;;
		--log_run_name)
			LOG_RUN_NAME="$1"
			shift
			;;
		--log_run_name=*)
			LOG_RUN_NAME="${arg#*=}"
			;;
		--num_return_sequences)
            NUM_RETURN_SEQUENCES="$1"
            shift
            ;;
        --num_return_sequences=*)
            NUM_RETURN_SEQUENCES="${arg#*=}"
            ;;
		
		*)
			echo "Unknown parameter passed: '${arg}'" >&2
			exit 1
			;;
	esac
done

if [[ -z "${REWARD_CRITIC_MODEL_NAME_OR_PATH+x}" ]]; then
	REWARD_CRITIC_MODEL_NAME_OR_PATH="${REWARD_MODEL_NAME_OR_PATH}"
fi
if [[ -z "${COST_CRITIC_MODEL_NAME_OR_PATH+x}" ]]; then
	COST_CRITIC_MODEL_NAME_OR_PATH="${COST_MODEL_NAME_OR_PATH}"
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

if [[ -z "${WANDB_API_KEY}" ]]; then
	export WANDB_MODE="offline"
fi

MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"

DEEPSPEED_ARGS=()
if [[ -n "${HOSTFILE+x}" ]]; then
	DEEPSPEED_ARGS+=("--hostfile" "${HOSTFILE}")
fi
DEEPSPEED_ARGS+=("--master_port" "${MASTER_PORT}")

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

deepspeed "${DEEPSPEED_ARGS[@]}" \
	--master_port "${MASTER_PORT}" \
	--module safe_rlhf.algorithms.reinforce_lag \
	--train_datasets PKU-SafeRLHF/train \
	--ptx_datasets alpaca \
	--actor_model_name_or_path "${ACTOR_MODEL_NAME_OR_PATH}" \
	--reward_model_name_or_path "${REWARD_MODEL_NAME_OR_PATH}" \
	--reward_critic_model_name_or_path "${REWARD_CRITIC_MODEL_NAME_OR_PATH}" \
	--cost_model_name_or_path "${COST_MODEL_NAME_OR_PATH}" \
	--cost_critic_model_name_or_path "${COST_CRITIC_MODEL_NAME_OR_PATH}" \
	--max_length 512 \
	--temperature 1 \
	--num_return_sequences "${NUM_RETURN_SEQUENCES}"\
	--repetition_penalty 1.0 \
	--trust_remote_code True \
	--epochs 1 \
	--update_iters 1 \
	--per_device_prompt_batch_size 64 \
	--per_device_train_batch_size 16 \
	--per_device_eval_batch_size 16 \
	--gradient_accumulation_steps 1 \
	--actor_lr 5e-6 \
	--actor_weight_decay 0.01 \
	--actor_lr_scheduler_type cosine \
	--actor_lr_warmup_ratio 0.03 \
	--actor_gradient_checkpointing \
	--critic_lr 5e-6 \
	--critic_weight_decay 0.0 \
	--critic_lr_scheduler_type constant \
	--critic_lr_warmup_ratio 0.03 \
	--critic_gradient_checkpointing \
	--normalize_reward False \
	--normalize_cost False \
	--seed "${SEED}" \
	--threshold "${THRESHOLD}" \
	--lambda_init 1.0 \
	--lambda_lr 0.01 \
	--lambda_max 1000000.0 \
	--lambda_update_delay_steps 0 \
	--episode_cost_window_size 256 \
	--kl_coeff "${KL_COEFF}" \
	--clip_range_ratio 0.2 \
	--clip_range_score 50.0 \
	--clip_range_value 5.0 \
	--ptx_coeff 8.0 \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project Safe-RLHF-PPO \
	--log_run_name "${LOG_RUN_NAME}" \
	--save_interval 300 \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--max_grad_norm 5.0 \
	--bf16 True \
	--tf32 True \
	--fail_prob "${FAIL_PROB}" \
    --safety_dataset_size "${SAFETY_DATASET_SIZE}" \
    --students_c1 "${STUDENTS_C1}" \
    --students_c2 "${STUDENTS_C2}" \
    --num_samples_per_prompt "${NUM_SAMPLES_PER_PROMPT}" \
    --rloo "${RLOO}" \
	--eval_strategy steps \
	--eval_interval 75 \
	--need_eval \
	--eval_datasets safety_test/val \
	# --eval_split_ratio 0.1 \
