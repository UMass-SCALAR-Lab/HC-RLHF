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
from typing import Any

from collections import deque
import deepspeed
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from safe_rlhf.datasets import PreferenceDataset, PrefOverPrefDataset
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.trainers import SupervisedTrainer
from safe_rlhf.utils import gather_log_probabilities, get_all_reduce_mean, is_main_process


class DPO_PoP_Fixed_Trainer(SupervisedTrainer):
    TRAINING_TYPE = 'dpo-pop-fixed'
    DATASET_TYPE = PrefOverPrefDataset

    model: deepspeed.DeepSpeedEngine
    reference_model: deepspeed.DeepSpeedEngine

    ds_train_config: dict[str, Any]
    ds_eval_config: dict[str, Any]

    def __init__(
        self,
        args: argparse.Namespace,
        ds_train_config: dict[str, Any],
        ds_eval_config: dict[str, Any],
    ) -> None:
        """Initialize trainer."""
        self.args = args
        self.ds_train_config = ds_train_config
        self.ds_eval_config = ds_eval_config
        self.scale_coeff = args.scale_coeff
        super().__init__(args, ds_train_config)

        # # Lagrange multiplier
        # self.log_lambda = torch.nn.Parameter(
        #     torch.tensor(np.log(self.args.lambda_init), device=self.args.device),
        #     requires_grad=True,
        # )

        # self.log_lambda_max = np.log(self.args.lambda_max) if self.args.lambda_max else None
        # self.log_lambda_optimizer = torch.optim.SGD([self.log_lambda], lr=self.args.lambda_lr)
        # self.lambda_update_delay_steps = self.args.lambda_update_delay_steps
        # self.constraint_hist = deque(maxlen=self.args.constraint_window_size)
        # self.threshold = self.args.threshold


    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if (
            self.ds_train_config is not None
            and self.ds_train_config['zero_optimization']['stage'] == 3
        ):
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_config)

        if (
            self.ds_eval_config is not None
            and self.ds_eval_config['zero_optimization']['stage'] == 3
        ):
            self.dsechf_eval = HfDeepSpeedConfig(self.ds_eval_config)

        self.model, self.tokenizer = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='left',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
        )
        self.reference_model, _ = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='left',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
        )

    def init_engines(self) -> None:
        super().init_engines()
        self.reference_model, *_ = deepspeed.initialize(
            model=self.reference_model,
            config=self.ds_eval_config,
        )

    @staticmethod
    def compute_log_probs(
        model: AutoModelForCausalLM,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        logits = model(input_ids, attention_mask=attention_mask).logits
        return gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])

    # def loss(  # pylint: disable=too-many-locals
    #     self,
    #     better_input_ids: torch.LongTensor,  # size = (B, L)
    #     better_attention_mask: torch.BoolTensor,  # size = (B, L)
    #     worse_input_ids: torch.LongTensor,  # size = (B, L)
    #     worse_attention_mask: torch.BoolTensor,  # size = (B, L)
    # ) -> dict[str, torch.Tensor]:
    def loss(
        self,
        pref_chosen_input_ids,
        pref_chosen_attention_mask,
        pref_rejected_input_ids,
        pref_rejected_attention_mask,
        unpref_chosen_input_ids,
        unpref_chosen_attention_mask,
        unpref_rejected_input_ids,
        unpref_rejected_attention_mask,
    ):
        """Loss function for the DPO algorithm.

        Args:
            better_input_ids (torch.LongTensor): The input ids of the better answer.
            better_attention_mask (torch.BoolTensor): The attention mask of the better answer.
            worse_input_ids (torch.LongTensor): The input ids of the worse answer.
            worse_attention_mask (torch.BoolTensor): The attention mask of the worse answer.

        Returns:
            dict[str, torch.Tensor]: loss, reward, better sample reward, worse sample reward
        """
        # import ipdb; ipdb.set_trace()
        assert pref_chosen_input_ids.size(0) == pref_rejected_input_ids.size(0) == unpref_chosen_input_ids.size(0) == unpref_rejected_input_ids.size(0), 'batch size mismatch!'
        batch_size = pref_chosen_input_ids.size(0)

        sequence_log_probs = self.compute_log_probs(  # size = (4 * B, L - 1)
            self.model.module,
            input_ids=torch.cat([pref_chosen_input_ids, pref_rejected_input_ids, unpref_chosen_input_ids, unpref_rejected_input_ids], dim=0),
            attention_mask=torch.cat([pref_chosen_attention_mask, pref_rejected_attention_mask, unpref_chosen_attention_mask, unpref_rejected_attention_mask], dim=0),
        )
        (
            pref_chosen_log_probs,  # size = (B, L - 1)
            pref_rejected_log_probs,  # size = (B, L - 1)
            unpref_chosen_log_probs,  # size = (B, L - 1)
            unpref_rejected_log_probs,  # size = (B, L - 1)
        ) = sequence_log_probs.chunk(chunks=4, dim=0)

        with torch.no_grad():
            ref_sequence_log_probs = self.compute_log_probs(  # size = (2 * B, L - 1)
                self.reference_model.module,
                input_ids=torch.cat([pref_chosen_input_ids, pref_rejected_input_ids, unpref_chosen_input_ids, unpref_rejected_input_ids], dim=0),
                attention_mask=torch.cat([pref_chosen_attention_mask, pref_rejected_attention_mask, unpref_chosen_attention_mask, unpref_rejected_attention_mask], dim=0),
            )
            (
                ref_pref_chosen_log_probs,  # size = (B, L - 1)
                ref_pref_rejected_log_probs,  # size = (B, L - 1)
                ref_unpref_chosen_log_probs,  # size = (B, L - 1)
                ref_unpref_rejected_log_probs,  # size = (B, L - 1)
            ) = ref_sequence_log_probs.chunk(chunks=4, dim=0)

        losses = []
        pref_chosen_sample_rewards, pref_rejected_sample_rewards = [], []
        unpref_chosen_sample_rewards, unpref_rejected_sample_rewards = [], []
        for i in range(batch_size):
            
            pref_chosen_end_index = pref_chosen_attention_mask[i].nonzero()[-1].squeeze().item()
            pref_rejected_end_index = pref_rejected_attention_mask[i].nonzero()[-1].squeeze().item()
            unpref_chosen_end_index = unpref_chosen_attention_mask[i].nonzero()[-1].squeeze().item()
            unpref_rejected_end_index = unpref_rejected_attention_mask[i].nonzero()[-1].squeeze().item()

            if(torch.all(torch.eq(pref_chosen_input_ids[i], pref_rejected_input_ids[i]),).item()):
                # import ipdb; ipdb.set_trace()
                continue
            if(torch.all(torch.eq(unpref_chosen_input_ids[i], unpref_rejected_input_ids[i]),).item()):
                # import ipdb; ipdb.set_trace()
                continue
            # assert not torch.all(
            #     torch.eq(pref_chosen_input_ids[i], pref_rejected_input_ids[i]),
            # ).item(), f"The better and worse answers are the same for Pref!  \n\n Chosen: \n\n {self.tokenizer.batch_decode(pref_chosen_input_ids,skip_special_tokens=True)} \n\n Rejected: \n\n {self.tokenizer.batch_decode(pref_rejected_input_ids,skip_special_tokens=True)} \n"
            # assert not torch.all(
            #     torch.eq(unpref_chosen_input_ids[i], unpref_rejected_input_ids[i]),
            # ).item(), f"The better and worse answers are the same for Unpref! \n\n Chosen: \n\n {self.tokenizer.batch_decode(unpref_chosen_input_ids,skip_special_tokens=True)} \n\n Rejected: \n\n {self.tokenizer.batch_decode(unpref_rejected_input_ids,skip_special_tokens=True)} \n"

            pref_diverge_index = (
                (pref_chosen_input_ids[i] != pref_rejected_input_ids[i]).nonzero()[0].squeeze().item()
            )
            assert 0 <= pref_diverge_index <= pref_chosen_end_index, 'pref diverge index is out of range!'
            assert 0 <= pref_diverge_index <= pref_rejected_end_index, 'pref diverge index is out of range!'
            unpref_diverge_index = (
                (unpref_chosen_input_ids[i] != unpref_rejected_input_ids[i]).nonzero()[0].squeeze().item()
            )
            assert 0 <= unpref_diverge_index <= unpref_chosen_end_index, 'unpref diverge index is out of range!'
            assert 0 <= unpref_diverge_index <= unpref_rejected_end_index, 'unpref diverge index is out of range!'

            pref_chosen_seq_slice = slice(pref_diverge_index, pref_chosen_end_index + 1)
            pref_rejected_seq_slice = slice(pref_diverge_index, pref_rejected_end_index + 1)
            unpref_chosen_seq_slice = slice(unpref_diverge_index, unpref_chosen_end_index + 1)
            unpref_rejected_seq_slice = slice(unpref_diverge_index, unpref_rejected_end_index + 1)

            # size = ()
            pref_chosen_log_prob_ = pref_chosen_log_probs[i, pref_chosen_seq_slice].sum(dim=-1)
            pref_rejected_log_prob_ = pref_rejected_log_probs[i, pref_rejected_seq_slice].sum(dim=-1)
            ref_pref_chosen_log_prob_ = ref_pref_chosen_log_probs[i, pref_chosen_seq_slice].sum(dim=-1)
            ref_pref_rejected_log_prob_ = ref_pref_rejected_log_probs[i, pref_rejected_seq_slice].sum(dim=-1)
            pref_chosen_log_ratio = pref_chosen_log_prob_ - ref_pref_chosen_log_prob_
            pref_rejected_log_ratio = pref_rejected_log_prob_ - ref_pref_rejected_log_prob_

            unpref_chosen_log_prob_ = unpref_chosen_log_probs[i, unpref_chosen_seq_slice].sum(dim=-1)
            unpref_rejected_log_prob_ = unpref_rejected_log_probs[i, unpref_rejected_seq_slice].sum(dim=-1)
            ref_unpref_chosen_log_prob_ = ref_unpref_chosen_log_probs[i, unpref_chosen_seq_slice].sum(dim=-1)
            ref_unpref_rejected_log_prob_ = ref_unpref_rejected_log_probs[i, unpref_rejected_seq_slice].sum(dim=-1)
            unpref_chosen_log_ratio = unpref_chosen_log_prob_ - ref_unpref_chosen_log_prob_
            unpref_rejected_log_ratio = unpref_rejected_log_prob_ - ref_unpref_rejected_log_prob_

            # loss_self = (-F.logsigmoid(self.scale_coeff * (pref_chosen_log_ratio-pref_rejected_log_ratio)) - F.logsigmoid(self.scale_coeff * (unpref_chosen_log_ratio-unpref_rejected_log_ratio)))/2.0
            # pop_gap_func = -F.logsigmoid(self.scale_coeff * ((pref_chosen_log_ratio-pref_rejected_log_ratio)-(unpref_chosen_log_ratio-unpref_rejected_log_ratio))) - np.log(2)
            # loss_cross = self.log_lambda.exp().item()*pop_gap_func
            # loss_ = loss_self+loss_cross
            # loss_ = -F.logsigmoid(self.scale_coeff * ((pref_chosen_log_ratio-pref_rejected_log_ratio)-(unpref_chosen_log_ratio.detach()-unpref_rejected_log_ratio.detach())))
            loss_ = -F.logsigmoid(self.scale_coeff * ((pref_chosen_log_ratio-pref_rejected_log_ratio)-torch.clamp((unpref_chosen_log_ratio.detach()-unpref_rejected_log_ratio.detach()),min=self.args.margin_min,max=self.args.margin_max)))
            losses.append(loss_)
            # self.constraint_hist.extend(pop_gap_func.detach().flatten().tolist())
            pref_chosen_sample_rewards.append(self.scale_coeff * pref_chosen_log_ratio.detach())
            pref_rejected_sample_rewards.append(self.scale_coeff * pref_rejected_log_ratio.detach())
            unpref_chosen_sample_rewards.append(self.scale_coeff * unpref_chosen_log_ratio.detach())
            unpref_rejected_sample_rewards.append(self.scale_coeff * unpref_rejected_log_ratio.detach())

        loss = torch.stack(losses).mean()  # size = ()
        pref_chosen_sample_reward_ = torch.stack(pref_chosen_sample_rewards)  # size = (B,)
        pref_rejected_sample_reward_ = torch.stack(pref_rejected_sample_rewards)  # size = (B,)
        # unpref_chosen_sample_reward_ = torch.stack(unpref_chosen_sample_rewards)  # size = (B,)
        # unpref_rejected_sample_reward_ = torch.stack(unpref_rejected_sample_rewards)  # size = (B,)
        # avg_reward = (pref_chosen_sample_reward_ + pref_rejected_sample_reward_ + unpref_chosen_sample_reward_ + unpref_rejected_sample_reward_)/4.0  # size = (B,)
        # reward_self_accuracy = ((pref_chosen_sample_reward_ > pref_rejected_sample_reward_).float().mean() + (unpref_chosen_sample_reward_ > unpref_rejected_sample_reward_).float().mean())/2.0  # size = ()
        # reward_pop_accuracy = ((pref_chosen_sample_reward_ - pref_rejected_sample_reward_) > (unpref_chosen_sample_reward_ - unpref_rejected_sample_reward_)).float().mean()
        acc = (pref_chosen_sample_reward_ > pref_rejected_sample_reward_).float().mean()
        pref_reward_margin = pref_chosen_sample_reward_ - pref_rejected_sample_reward_  # size = (B,)
        # unpref_reward_margin = unpref_chosen_sample_reward_ - unpref_rejected_sample_reward_  # size = (B,)
        # pop_margin = pref_reward_margin - unpref_reward_margin

        return {
            'loss': loss,
            # 'avg_reward': avg_reward,
            'pref_chosen_sample_reward': pref_chosen_sample_reward_,
            'pref_rejected_sample_reward': pref_rejected_sample_reward_,
            # 'unpref_chosen_sample_reward': unpref_chosen_sample_reward_,
            # 'unpref_rejected_sample_reward': unpref_rejected_sample_reward_,
            # 'reward_self_accuracy': reward_self_accuracy,
            # 'reward_pop_accuracy': reward_pop_accuracy,
            'accuracy': acc,
            'pref_reward_margin': pref_reward_margin,
            # 'unpref_reward_margin': unpref_reward_margin,
            # 'pop_margin': pop_margin,
        }

    # pref_chosen_input_ids
    # pref_chosen_attention_mask
    # pref_rejected_input_ids
    # pref_rejected_attention_mask
    # unpref_chosen_input_ids
    # unpref_chosen_attention_mask
    # unpref_rejected_input_ids
    # unpref_rejected_attention_mask
    # def train_step(
    #     self,
    #     better_input_ids: torch.LongTensor,  # size = (B, L)
    #     better_attention_mask: torch.BoolTensor,  # size = (B, L)
    #     worse_input_ids: torch.LongTensor,  # size = (B, L)
    #     worse_attention_mask: torch.BoolTensor,  # size = (B, L)
    # ) -> dict[str, Any]:
    
    def train_step(
        self,
        pref_chosen_input_ids,
        pref_chosen_attention_mask,
        pref_rejected_input_ids,
        pref_rejected_attention_mask,
        unpref_chosen_input_ids,
        unpref_chosen_attention_mask,
        unpref_rejected_input_ids,
        unpref_rejected_attention_mask,
        **score_batch_kwargs,
    ):
        """Perform a single training step.

        Args:
            better_input_ids (torch.LongTensor): The input ids of the better answer.
            better_attention_mask (torch.BoolTensor): The attention mask of the better answer.
            worse_input_ids (torch.LongTensor): The input ids of the worse answer.
            worse_attention_mask (torch.BoolTensor): The attention mask of the worse answer.

        Returns:
            dict[str, Any]: training loss, reward, learning rate
        """

        # Get samples for the Constraint for 2 batches
        # if(self.global_step < 2):
        #     with torch.no_grad():
        #         loss_dict = self.loss(
        #             pref_chosen_input_ids=pref_chosen_input_ids,
        #             pref_chosen_attention_mask=pref_chosen_attention_mask,
        #             pref_rejected_input_ids=pref_rejected_input_ids,
        #             pref_rejected_attention_mask=pref_rejected_attention_mask,
        #             unpref_chosen_input_ids=unpref_chosen_input_ids,
        #             unpref_chosen_attention_mask=unpref_chosen_attention_mask,
        #             unpref_rejected_input_ids=unpref_rejected_input_ids,
        #             unpref_rejected_attention_mask=unpref_rejected_attention_mask,
        #         )
        #         loss = loss_dict['loss']
        # else:
            
        # constraint_ = torch.tensor(self.constraint_hist).mean().to(self.args.device)

        # dist.reduce(constraint_, dst=0, op=dist.ReduceOp.AVG)

        # if is_main_process() and self.global_step >= self.lambda_update_delay_steps:
        #     lambda_loss = -(constraint_ - self.threshold) * self.log_lambda.exp()
        #     self.log_lambda_optimizer.zero_grad()
        #     lambda_loss.backward()
        #     self.log_lambda_optimizer.step()
        #     if self.log_lambda_max is not None:
        #         with torch.no_grad():
        #             self.log_lambda.clamp_(max=self.log_lambda_max)

        # dist.broadcast(self.log_lambda, src=0)

        # dist.barrier()

        loss_dict = self.loss(
            pref_chosen_input_ids=pref_chosen_input_ids,
            pref_chosen_attention_mask=pref_chosen_attention_mask,
            pref_rejected_input_ids=pref_rejected_input_ids,
            pref_rejected_attention_mask=pref_rejected_attention_mask,
            unpref_chosen_input_ids=unpref_chosen_input_ids,
            unpref_chosen_attention_mask=unpref_chosen_attention_mask,
            unpref_rejected_input_ids=unpref_rejected_input_ids,
            unpref_rejected_attention_mask=unpref_rejected_attention_mask,
        )
        loss = loss_dict['loss']
        self.model.backward(loss)
        self.model.step()

        with torch.no_grad():
            # avg_reward = loss_dict['avg_reward'].mean()
            pref_chosen_sample_reward = loss_dict['pref_chosen_sample_reward'].mean()
            pref_rejected_sample_reward = loss_dict['pref_rejected_sample_reward'].mean()
            # unpref_chosen_sample_reward = loss_dict['unpref_chosen_sample_reward'].mean()
            # unpref_rejected_sample_reward = loss_dict['unpref_rejected_sample_reward'].mean()
            accuracy = loss_dict['accuracy']
            # reward_self_accuracy = loss_dict['reward_self_accuracy']
            # reward_pop_accuracy = loss_dict['reward_pop_accuracy']
            pref_reward_margin = loss_dict['pref_reward_margin'].mean()
            # unpref_reward_margin = loss_dict['unpref_reward_margin'].mean()
            # pop_margin = loss_dict['pop_margin'].mean()

            loss = get_all_reduce_mean(loss)
            # avg_reward = get_all_reduce_mean(avg_reward)
            pref_chosen_sample_reward = get_all_reduce_mean(pref_chosen_sample_reward)
            pref_rejected_sample_reward = get_all_reduce_mean(pref_rejected_sample_reward)
            # unpref_chosen_sample_reward = get_all_reduce_mean(unpref_chosen_sample_reward)
            # unpref_rejected_sample_reward = get_all_reduce_mean(unpref_rejected_sample_reward)
            accuracy = get_all_reduce_mean(accuracy)
            # reward_self_accuracy = get_all_reduce_mean(reward_self_accuracy)
            # reward_pop_accuracy = get_all_reduce_mean(reward_pop_accuracy)
            pref_reward_margin = get_all_reduce_mean(pref_reward_margin)
            # unpref_reward_margin = get_all_reduce_mean(unpref_reward_margin)
            # pop_margin = get_all_reduce_mean(pop_margin)

        return {
            'train/loss': loss.item(),
            # 'train/avg_reward': avg_reward.item(),
            'train/pref_chosen_sample_reward': pref_chosen_sample_reward.item(),
            'train/pref_rejected_sample_reward': pref_rejected_sample_reward.item(),
            # 'train/unpref_chosen_sample_reward': unpref_chosen_sample_reward.item(),
            # 'train/unpref_rejected_sample_reward': unpref_rejected_sample_reward.item(),
            'train/accuracy': accuracy.item(),
            # 'train/reward_self_accuracy': reward_self_accuracy.item(),
            # 'train/reward_pop_accuracy': reward_pop_accuracy.item(),
            'train/pref_reward_margin': pref_reward_margin.item(),
            # 'train/unpref_reward_margin': unpref_reward_margin.item(),
            # 'train/pop_margin': pop_margin.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
            # 'train/lambda': self.log_lambda.exp().item(),
        }
