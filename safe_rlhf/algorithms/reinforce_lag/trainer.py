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

from __future__ import annotations

import argparse
from collections import deque
from typing import Any
import os, sys

import deepspeed
import numpy as np
from scipy.stats import t
import torch
import torch.distributed as dist
from transformers import PreTrainedTokenizerBase

from safe_rlhf.models import AutoModelForScore, load_pretrained_models
from safe_rlhf.trainers import ReinforceTrainer
from safe_rlhf.utils import (
    batch_retokenize,
    gather_log_probabilities,
    get_all_reduce_max,
    get_all_reduce_mean,
    is_main_process,
    is_same_tokenizer,
    masked_mean,
)


class ReinforceCILagTrainer(ReinforceTrainer):
    TRAINING_TYPE = 'reinforce_ci_lag'

    cost_model: deepspeed.DeepSpeedEngine
    cost_critic_model: deepspeed.DeepSpeedEngine

    cost_tokenizer: PreTrainedTokenizerBase
    cost_critic_tokenizer: PreTrainedTokenizerBase

    def __init__(
        self,
        args: argparse.Namespace,
        ds_train_config: dict[str, Any],
        ds_eval_config: dict[str, Any],
    ) -> None:
        super().__init__(args=args, ds_train_config=ds_train_config, ds_eval_config=ds_eval_config)

        # Lagrange multiplier
        self.log_lambda = torch.nn.Parameter(
            torch.tensor(np.log(self.args.lambda_init), device=self.args.device),
            requires_grad=True,
        )

        self.log_lambda_max = np.log(self.args.lambda_max) if self.args.lambda_max else None
        self.log_lambda_optimizer = torch.optim.SGD([self.log_lambda], lr=self.args.lambda_lr)
        self.lambda_update_delay_steps = self.args.lambda_update_delay_steps
        self.episode_costs = deque(maxlen=self.args.episode_cost_window_size)
        self.threshold = self.args.threshold

        # Running mean of the reinforce objective to be used as the baseline
        # self.reinforce_mean_obj = torch.nn.Parameter(
        #     torch.tensor(0., device=self.args.device),
        #     requires_grad=False,
        # )
        # self.reinforce_tot_samples_seen = torch.nn.Parameter(
        #     torch.tensor(0., device=self.args.device),
        #     requires_grad=False,
        # )
        self.reinforce_mean_obj = torch.tensor(0., device=self.args.device)
        self.reinforce_tot_samples_seen = torch.tensor(0., device=self.args.device)

        # # Add parameters for mean and variance of return estimates
        # self.mean_return_estimate = torch.nn.Parameter(
        #     torch.tensor(0., device=self.args.device),
        #     requires_grad=False,
        # )
        # self.var_return_estimate = torch.nn.Parameter(
        #     torch.tensor(1., device=self.args.device),
        #     requires_grad=False,
        # )

    def init_models(self) -> None:
        super().init_models()
        self.cost_model, self.cost_tokenizer = load_pretrained_models(
            self.args.cost_model_name_or_path,
            model_max_length=self.args.max_length,
            auto_model_type=AutoModelForScore,
            padding_side='right',
            trust_remote_code=self.args.trust_remote_code,
            auto_model_kwargs={
                'score_type': 'cost',
                'do_normalize': self.args.normalize_cost,
            },
        )
        self.cost_model.set_normalize(self.args.normalize_cost)

        # DONT NEED A CRITIC
        # if self.args.cost_critic_model_name_or_path is None:
        #     self.args.cost_critic_model_name_or_path = self.args.cost_model_name_or_path
        # self.cost_critic_model, self.cost_critic_tokenizer = load_pretrained_models(
        #     self.args.cost_critic_model_name_or_path,
        #     model_max_length=self.args.max_length,
        #     auto_model_type=AutoModelForScore,
        #     padding_side='left',
        #     trust_remote_code=self.args.trust_remote_code,
        #     auto_model_kwargs={
        #         'score_type': 'critic',
        #         'do_normalize': False,
        #     },
        # )
        # self.cost_critic_model.set_normalize(False)

        if is_same_tokenizer(self.tokenizer, self.cost_tokenizer):
            self.cost_tokenizer = self.tokenizer
        # if not is_same_tokenizer(self.tokenizer, self.cost_critic_tokenizer):
        #     raise ValueError(
        #         (
        #             'Cost critic tokenizer must be the same as actor tokenizer. '
        #             'Expected {0.__module__}.{0.__qualname__}(vocab_size={1}), '
        #             'but got {2.__module__}.{2.__qualname__}(vocab_size={3}). '
        #             'Please consider pass `--cost_critic_model_name_or_path` from the command line.'
        #         ).format(
        #             type(self.tokenizer),
        #             len(self.tokenizer),
        #             type(self.cost_critic_tokenizer),
        #             len(self.cost_critic_tokenizer),
        #         ),
        #     )
        # self.cost_critic_tokenizer = self.tokenizer

    def init_engines(self) -> None:
        super().init_engines()

        # NO CRITIC NEEDED!!
        # self.cost_critic_model = self._init_train_engine(
        #     model=self.cost_critic_model,
        #     weight_decay=self.args.critic_weight_decay,
        #     lr=self.args.critic_lr,
        #     lr_scheduler_type=self.args.critic_lr_scheduler_type,
        #     lr_warmup_ratio=self.args.critic_lr_warmup_ratio,
        #     total_training_steps=self.args.total_training_steps,
        #     ds_config=self.ds_train_config,
        # )

        self.cost_model = self._init_eval_engine(
            model=self.cost_model,
            ds_config=self.ds_eval_config,
        )
        self.cost_model.eval()

        # NO CRITIC NEEDED!
        # if self.args.critic_gradient_checkpointing:
        #     self.cost_critic_model.gradient_checkpointing_enable()

    def set_train(self, mode: bool = True) -> None:
        super().set_train(mode=mode)

        # NO CRITIC NEEDED!
        # if mode:
        #     self.cost_critic_model.train()
        # else:
        #     self.cost_critic_model.eval()

    @torch.no_grad()
    def post_rollout(
        self,
        prompt: torch.Tensor,
        sequence: torch.Tensor,
        attention_mask: torch.BoolTensor,
    ) -> dict[str, Any]:
        if self.reward_tokenizer is not self.tokenizer:
            reward_tokenize_output = batch_retokenize(
                sequence,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.reward_tokenizer,
                skip_special_tokens=True,
                device=self.args.device,
            )
            reward_seq = reward_tokenize_output['input_ids']
            reward_attention_mask = reward_tokenize_output['attention_mask']
        else:
            reward_seq = sequence
            reward_attention_mask = attention_mask

        if self.cost_tokenizer is not self.tokenizer:
            cost_tokenize_output = batch_retokenize(
                sequence,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.cost_tokenizer,
                skip_special_tokens=True,
                device=self.args.device,
            )
            cost_seq = cost_tokenize_output['input_ids']
            cost_attention_mask = cost_tokenize_output['attention_mask']
        else:
            cost_seq = sequence
            cost_attention_mask = attention_mask

        logits = self.actor_model(sequence, attention_mask=attention_mask).logits
        ref_logits = self.actor_reference_model(sequence, attention_mask=attention_mask).logits

        reward = self.reward_model(reward_seq, attention_mask=reward_attention_mask).end_scores
        cost = self.cost_model(cost_seq, attention_mask=cost_attention_mask).end_scores
        # NO CRITIC MODELS IN REINFORCE!!
        # reward_values = self.reward_critic_model(sequence, attention_mask=attention_mask).scores
        # cost_values = self.cost_critic_model(sequence, attention_mask=attention_mask).scores

        reward = reward.squeeze(dim=-1)
        cost = cost.squeeze(dim=-1)
        # NO CRITIC MODELS IN REINFORCE!!
        # reward_values = reward_values.squeeze(dim=-1)[:, :-1]
        # cost_values = cost_values.squeeze(dim=-1)[:, :-1]

        log_probs = gather_log_probabilities(logits[:, :-1], sequence[:, 1:])
        ref_log_probs = gather_log_probabilities(ref_logits[:, :-1], sequence[:, 1:])
        
        # IS THIS NEEDED IN REINFORCE??
        self.episode_costs.extend(cost.tolist())

        return {
            'prompt': prompt,
            'log_probs': log_probs,
            'ref_log_probs': ref_log_probs,
            'reward': reward,
            'cost': cost,
            # NO CRITIC MODELS IN REINFORCE!!
            # 'reward_values': reward_values,
            # 'cost_values': cost_values,
            'input_ids': sequence,
            'attention_mask': attention_mask,
        }

    @torch.no_grad()
    def eval_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.BoolTensor,
    ) -> dict[str, torch.Tensor]:
        if self.reward_tokenizer is not self.tokenizer:
            reward_tokenize_output = batch_retokenize(
                input_ids,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.reward_tokenizer,
                skip_special_tokens=True,
                device=self.args.device,
            )
            reward_input_ids = reward_tokenize_output['input_ids']
            reward_attention_mask = reward_tokenize_output['attention_mask']
        else:
            reward_input_ids = input_ids
            reward_attention_mask = attention_mask

        if self.cost_tokenizer is not self.tokenizer:
            cost_tokenize_output = batch_retokenize(
                input_ids,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.cost_tokenizer,
                skip_special_tokens=True,
                device=self.args.device,
            )
            cost_input_ids = cost_tokenize_output['input_ids']
            cost_attention_mask = cost_tokenize_output['attention_mask']
        else:
            cost_input_ids = input_ids
            cost_attention_mask = attention_mask

        reward = self.reward_model(
            reward_input_ids,
            attention_mask=reward_attention_mask,
        ).end_scores.squeeze(dim=-1)
        cost = self.cost_model(
            cost_input_ids,
            attention_mask=cost_attention_mask,
        ).end_scores.squeeze(dim=-1)
        return {
            'eval/reward': reward,
            'eval/cost': cost,
            'eval/cost_success': (cost <= self.args.threshold).float(),
        }

    # We add the KL penalty to the rewards only
    # We compute a single value for the prompt+response pair, instead ofdoing token by token which PPO does
    # The costs will be used later to contruct the objective for the Seldonian Learning framework
    def add_kl_divergence_regularization(
        self,
        reward: torch.Tensor,  # size = (B,)
        cost: torch.Tensor,  # size = (B,)
        prompt: torch.LongTensor,  # size = (B, S) # pylint: disable=unused-argument
        log_probs: torch.Tensor,  # size = (B, L)
        ref_log_probs: torch.Tensor,  # size = (B, L)
        sequence_mask: torch.BoolTensor,  # size = (B, L)
    ) -> tuple[torch.Tensor, torch.Tensor]:  # size = (B, L)

        start = prompt.size(-1) - 1
        end_index = torch.cat([m.nonzero()[-1] for m in sequence_mask])  # size = (B,)
        mask = sequence_mask[:, start:]

        # size = (B, L)
        kl_divergence_estimate = log_probs - ref_log_probs
        kl_divergence_estimate_tot = torch.sum(kl_divergence_estimate[:,start:]*mask, dim=-1)
        kl_penalty_rewards = -self.kl_coeff * kl_divergence_estimate_tot
        net_rewards = reward + kl_penalty_rewards

        # rewards = torch.scatter_add(
        #     kl_penalty_rewards,
        #     dim=-1,
        #     index=end_index.unsqueeze(dim=-1),
        #     src=reward.to(kl_penalty_rewards.dtype).unsqueeze(dim=-1),
        # )
        # costs = torch.scatter_add(
        #     -kl_penalty_rewards,
        #     dim=-1,
        #     index=end_index.unsqueeze(dim=-1),
        #     src=cost.to(kl_penalty_rewards.dtype).unsqueeze(dim=-1),
        # )
        # return (
        #     torch.clamp(rewards, min=-self.clip_range_score, max=self.clip_range_score),
        #     torch.clamp(costs, min=-self.clip_range_score, max=self.clip_range_score),
        # )

        return torch.clamp(net_rewards, min=-self.clip_range_score, max=self.clip_range_score)

    # NEED TO ADD THE BASELINE HERE!!!
    def actor_loss_fn(
        self,
        log_probs: torch.Tensor,  # size = (B, L - S)
        old_log_probs: torch.Tensor,  # size = (B, L - S)
        rewards: torch.Tensor,  # size = (B,)
        costs: torch.Tensor,  # size = (B,)
        mask: torch.BoolTensor,  # size = (B, L - S)
    ) -> torch.Tensor:  # size = ()
        multiplier = self.log_lambda.exp().item()

        # size = (B, L - S)
        # advantages = (reward_advantages - multiplier * cost_advantages) / (1.0 + multiplier)
        # ratios = torch.exp(log_probs - old_log_probs)
        # surrogate1 = advantages * ratios
        # surrogate2 = advantages * torch.clamp(
        #     ratios,
        #     1.0 - self.clip_range_ratio,
        #     1.0 + self.clip_range_ratio,
        # )
        # surrogate = torch.minimum(surrogate1, surrogate2)
        # return -masked_mean(surrogate, mask)  # size = ()

        # WE CHANGE THIS TO THE REINFORCE LOSS
        # Should we normalize this by (1+multiplier) as the SAFE RLHF paper does??
        # Shape: (B,)
        net_return = (rewards - multiplier*costs)
        # Shape: (B,)
        traj_log_probs = torch.sum(log_probs*mask, dim=-1)
        if(self.args.rloo and self.args.num_samples_per_prompt > 1):
            returns_per_prompt = net_return.reshape((-1,self.args.num_samples_per_prompt))
            baseline_rloo = torch.mean(returns_per_prompt, dim=1, keepdim=True)
            obj_values = (returns_per_prompt-baseline_rloo)*(traj_log_probs.reshape(-1,self.args.num_samples_per_prompt))
            obj_values_per_prompt = obj_values.sum(dim=-1)/(self.args.num_samples_per_prompt - 1)
            return -obj_values_per_prompt.mean()
        else:
            return -((net_return-self.reinforce_mean_obj)*traj_log_probs).mean()


    # pylint: disable-next=too-many-locals
    def rl_step(self, rl_batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        
        # episode_cost = torch.tensor(self.episode_costs).mean().to(self.args.device)
        # dist.reduce(episode_cost, dst=0, op=dist.ReduceOp.AVG)

        # Assume `episode_cost` is defined and dist is initialized
        episode_cost_tensor = torch.tensor(self.episode_costs).to(self.args.device)
        # Prepare an empty tensor to hold gathered values
        gathered_costs = [torch.zeros_like(episode_cost_tensor) for _ in range(dist.get_world_size())]
        # Gather costs from all devices
        dist.all_gather(gathered_costs, episode_cost_tensor)
        # Concatenate the gathered tensors
        all_costs = torch.cat(gathered_costs, dim=0).to(self.args.device)

        # episode_cost_tensor = torch.tensor(self.episode_costs)
        # gathered_costs = None
        # if dist.get_rank() == 0:
        #     # Rank 0 will hold the gathered tensors
        #     gathered_costs = [torch.zeros_like(episode_cost_tensor) for _ in range(dist.get_world_size())]
        # # Gather costs from all devices to rank 0
        # dist.gather(episode_cost_tensor, gather_list=gathered_costs, dst=0)

        # SHOULD WE USE MINIBATCH OR BUFFER HERE TO COMPUTE UCB FOR THE DUAL GRADIENT???
        if is_main_process() and self.global_step >= self.lambda_update_delay_steps:
            # Threshold has to be a negative value if using this objective => YC
            # lambda_loss = -(episode_cost - self.threshold) * self.log_lambda.exp()
            # all_costs = torch.cat(gathered_costs, dim=0)
            
            # Probability (alpha/2 for two-tailed test)
            fail_probability = self.args.fail_prob
            # IF using buffer
            all_costs_dual = all_costs
            # IF using mini batch 
            # all_costs_dual = rl_batch['cost']

            # Degrees of freedom
            batch_dataset_size = len(all_costs_dual) # Since we use all the samples in the queue to compute the statistics
            safety_dataset_size = self.args.safety_dataset_size
            # Expand the bound
            C1, C2 = self.args.students_c1, self.args.students_c2
            # Calculate the t-value
            t_value_in = t.ppf(1 - fail_probability, batch_dataset_size-1)
            # Calculate the t-value
            t_value_out = t.ppf(1 - fail_probability, safety_dataset_size-1)
            students_coeff = (C1 * (t_value_in/np.sqrt(batch_dataset_size))) + (C2 * (t_value_out/np.sqrt(safety_dataset_size)))
            
            # torch uses bessels correction for stdev by default
            lambda_loss = -(all_costs_dual.mean() + (students_coeff*all_costs_dual.std()) - self.threshold) * self.log_lambda.exp()
            self.log_lambda_optimizer.zero_grad()
            lambda_loss.backward()
            self.log_lambda_optimizer.step()
            if self.log_lambda_max is not None:
                with torch.no_grad():
                    self.log_lambda.clamp_(max=self.log_lambda_max)

        dist.broadcast(self.log_lambda, src=0)
        # dist.barrier(device_ids=int(os.environ["LOCAL_RANK"]))

        dist.barrier()

        prompt = rl_batch['prompt']
        old_log_probs = rl_batch['log_probs']
        ref_log_probs = rl_batch['ref_log_probs']
        reward = rl_batch['reward']
        cost = rl_batch['cost']
        # NO CRITIC MODEL!!
        # old_reward_values = rl_batch['reward_values']
        # old_cost_values = rl_batch['cost_values']
        input_ids = rl_batch['input_ids']
        attention_mask = rl_batch['attention_mask']

        start = prompt.size(-1) - 1
        sequence_mask = attention_mask[:, 1:]

        with torch.no_grad():
            # THIS IS THE OLD VARIANT, THE NEW VARIANT RETURN THE KL REGULARIZED RETURNS OF SHAPE (B,)
            # old_rewards, old_costs = self.add_kl_divergence_regularization(
            #     reward,
            #     cost,
            #     prompt,
            #     old_log_probs,
            #     ref_log_probs,
            #     sequence_mask,
            # )
            old_rewards = self.add_kl_divergence_regularization(
                reward,
                cost,
                prompt,
                old_log_probs,
                ref_log_probs,
                sequence_mask,
            )

            # NO CRITIC MODELS IN REINFORCE!!
            # reward_advantages, reward_returns = self.get_advantages_and_returns(
            #     old_reward_values,
            #     old_rewards,
            #     sequence_mask,
            #     start,
            # )
            # cost_advantages, cost_returns = self.get_advantages_and_returns(
            #     old_cost_values,
            #     old_costs,
            #     sequence_mask,
            #     start,
            # )

            # MODIFY OLD_COSTS TO STORE THE AUGMENTED COST IN THE SELDONIAN SETTING 
            
            # Probability (alpha/2 for two-tailed test)
            fail_probability = self.args.fail_prob
            # Degrees of freedom
            batch_dataset_size = self.args.per_device_train_batch_size * self.args.num_samples_per_prompt
            safety_dataset_size = self.args.safety_dataset_size
            # Expand the bound
            C1, C2 = self.args.students_c1, self.args.students_c2

            # Get Estimate of Mean and Standard Deviation
            # We are using all the samples in our replay buffer to compute these estimates
            # This is obviously biased, but is consistent as we converge to a stationary policy
            mean_estimate = all_costs.mean()
            std_estimate = all_costs.std()

            # Calculate the t-value
            t_value_in = t.ppf(1 - fail_probability, batch_dataset_size-1)
            # Calculate the t-value
            t_value_out = t.ppf(1 - fail_probability, safety_dataset_size-1)

            scale_coeff = (C1 * (t_value_in/np.sqrt(batch_dataset_size))) + (C2 * (t_value_out/np.sqrt(safety_dataset_size)))
            old_costs = cost + (scale_coeff* (cost**2 - (2*cost*mean_estimate))/(2*std_estimate))

        # if is_main_process():
        with torch.no_grad():
            multiplier = self.log_lambda.exp().item()
            aug_return = (old_rewards - multiplier*old_costs).to(self.args.device)
            num_local_samples = torch.tensor(len(aug_return), device=self.args.device)
            aug_return_tot = aug_return.sum()
            dist.reduce(aug_return_tot, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(num_local_samples, dst=0, op=dist.ReduceOp.SUM)

            self.reinforce_mean_obj = ((self.reinforce_mean_obj * self.reinforce_tot_samples_seen) + aug_return_tot) / (self.reinforce_tot_samples_seen + num_local_samples)
            self.reinforce_tot_samples_seen += num_local_samples

        dist.broadcast(self.reinforce_mean_obj, src=0)
    
        dist.broadcast(self.reinforce_tot_samples_seen, src=0)

        # Barrier to ensure all ranks have received the updated reinforce_mean_obj
        # dist.barrier(device_ids=int(os.environ["LOCAL_RANK"]))
        dist.barrier()

        logits = self.actor_model(input_ids, attention_mask=attention_mask, use_cache=False).logits
        
        log_probs = gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
        actor_loss = self.actor_loss_fn(
            log_probs[:, start:],
            old_log_probs[:, start:],
            old_rewards,
            old_costs,
            sequence_mask[:, start:],
        )

        self.actor_model.backward(actor_loss)
        self.actor_model.step()

        # NO CRITIC MODELS IN REINFORCE!!
        # reward_values = self.reward_critic_model(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     use_cache=False,
        # ).scores
        # reward_values = reward_values.squeeze(dim=-1)[:, :-1]
        # reward_critic_loss = self.critic_loss_fn(
        #     reward_values[:, start:],
        #     old_reward_values[:, start:],
        #     reward_returns,
        #     sequence_mask[:, start:],
        # )
        # self.reward_critic_model.backward(reward_critic_loss)
        # self.reward_critic_model.step()

        # cost_values = self.cost_critic_model(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     use_cache=False,
        # ).scores
        # cost_values = cost_values.squeeze(dim=-1)[:, :-1]
        # cost_critic_loss = self.critic_loss_fn(
        #     cost_values[:, start:],
        #     old_cost_values[:, start:],
        #     cost_returns,
        #     sequence_mask[:, start:],
        # )
        # self.cost_critic_model.backward(cost_critic_loss)
        # self.cost_critic_model.step()

        with torch.no_grad():
            mask = sequence_mask[:, start:]
            kl_divergence = ((old_log_probs - ref_log_probs)[:, start:] * mask).sum(dim=-1).mean()
            mean_generated_length = mask.sum(dim=-1).float().mean()
            max_generated_length = mask.sum(dim=-1).float().max()

            cost_success = (cost <= self.args.threshold).float().mean()
            reward = reward.mean()
            cost = cost.mean()

            # reward_with_kl_penalty = (old_rewards[:, start:] * mask).sum(dim=-1).mean()
            # reward_advantage = masked_mean(reward_advantages, mask)
            # reward_return = masked_mean(reward_returns, mask)
            # reward_value = masked_mean(reward_values[:, start:], mask)
            # cost_with_kl_penalty = (old_costs[:, start:] * mask).sum(dim=-1).mean()
            # cost_advantage = masked_mean(cost_advantages, mask)
            # cost_return = masked_mean(cost_returns, mask)
            # cost_value = masked_mean(cost_values[:, start:], mask)

            reward_with_kl_penalty = old_rewards.mean()
            cost_with_seldonian_constraint = old_costs.mean()

            actor_loss = get_all_reduce_mean(actor_loss)
            # reward_critic_loss = get_all_reduce_mean(reward_critic_loss)
            # cost_critic_loss = get_all_reduce_mean(cost_critic_loss)
            cost_success = get_all_reduce_mean(cost_success)
            reward = get_all_reduce_mean(reward)
            cost = get_all_reduce_mean(cost)
            reward_with_kl_penalty = get_all_reduce_mean(reward_with_kl_penalty)
            # reward_advantage = get_all_reduce_mean(reward_advantage)
            # reward_return = get_all_reduce_mean(reward_return)
            # reward_value = get_all_reduce_mean(reward_value)
            cost_with_seldonian_constraint = get_all_reduce_mean(cost_with_seldonian_constraint)
            # cost_advantage = get_all_reduce_mean(cost_advantage)
            # cost_return = get_all_reduce_mean(cost_return)
            # cost_value = get_all_reduce_mean(cost_value)
            kl_divergence = get_all_reduce_mean(kl_divergence)
            mean_generated_length = get_all_reduce_mean(mean_generated_length)
            max_generated_length = get_all_reduce_max(max_generated_length)

        dist.barrier()

        return {
            'train/actor_loss': actor_loss.item(),
            # 'train/reward_critic_loss': reward_critic_loss.item(),
            # 'train/cost_critic_loss': cost_critic_loss.item(),
            'train/lambda': self.log_lambda.exp().item(),
            # 'train/episode_cost': episode_cost.item(),
            'train/mean_cost_estimate': all_costs.mean().item(),
            'train/stdev_cost_estimate': all_costs.std().item(),
            'train/reward': reward.item(),
            'train/cost': cost.item(),
            'train/cost_success': cost_success.item(),
            'train/reward_with_kl_penalty': reward_with_kl_penalty.item(),
            # 'train/reward_advantage': reward_advantage.item(),
            # 'train/reward_return': reward_return.item(),
            # 'train/reward_value': reward_value.item(),
            'train/cost_with_seldonian_constraint': cost_with_seldonian_constraint.item(),
            # 'train/cost_advantage': cost_advantage.item(),
            # 'train/cost_return': cost_return.item(),
            # 'train/cost_value': cost_value.item(),
            'train/kl_divergence': kl_divergence.item(),
            'train/actor_lr': self.actor_model.optimizer.param_groups[0]['lr'],
            # 'train/reward_critic_lr': self.reward_critic_model.optimizer.param_groups[0]['lr'],
            # 'train/cost_critic_lr': self.cost_critic_model.optimizer.param_groups[0]['lr'],
            'train/mean_generated_length': mean_generated_length.item(),
            'train/max_generated_length': max_generated_length.item(),
        }
