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
"""Trainer base class for supervised training."""

from __future__ import annotations

import abc
import argparse
from copy import deepcopy
from typing import Any, ClassVar
import matplotlib.pyplot as plt
import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, get_scheduler
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from safe_rlhf.configs import ADAM_BETAS
from safe_rlhf.datasets import TokenizedDataset
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.trainers.base import TrainerBase
from safe_rlhf.utils import get_optimizer_grouped_parameters, is_main_process, to_device


class SupervisedTrainerEMA(TrainerBase):
    """Trainer base class for supervised training.

    Abstract methods:
        loss: Compute supervised training loss.
        train_step: Perform a single training step.
    """

    TRAINING_TYPE: ClassVar[str] = 'supervised'
    DATASET_TYPE: ClassVar[type[TokenizedDataset]]
    MODEL_TYPE = AutoModelForCausalLM

    model: deepspeed.DeepSpeedEngine
    ds_config: dict[str, Any]

    extra_model_kwargs: dict[str, Any] | None = None
    extra_tokenizer_kwargs: dict[str, Any] | None = None

    def __init__(self, args: argparse.Namespace, ds_config: dict[str, Any]) -> None:
        """Initialize trainer."""
        self.args = args
        self.ds_config = ds_config
        self.global_step = 0

        self.init_models()
        dist.barrier()
        self.init_datasets()
        dist.barrier()
        self.init_engines()
        dist.barrier()
        self.init_logger()

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_config is not None and self.ds_config['zero_optimization']['stage'] == 3:
            self.dstchf = HfDeepSpeedConfig(self.ds_config)

        self.model, self.tokenizer = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='right',
            auto_model_type=self.MODEL_TYPE,
            trust_remote_code=self.args.trust_remote_code,
            auto_model_kwargs=self.extra_model_kwargs,
            auto_tokenizer_kwargs=self.extra_tokenizer_kwargs,
        )
        # self.ema_model = deepcopy(self.model)
        # self.ema_model.eval()
        # for param in self.ema_model.parameters():
        #     param.requires_grad = False

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        
        train_dataset = self.DATASET_TYPE(
            self.args.train_datasets,
            tokenizer=self.tokenizer,
        )

        if self.args.need_eval:
            if self.args.eval_datasets is None and self.args.eval_split_ratio is not None:
                train_dataset, eval_dataset = train_dataset.split_train_test(
                    split_ratio=self.args.eval_split_ratio,
                )
            elif self.args.eval_datasets is not None and self.args.eval_split_ratio is None:
                eval_dataset = self.DATASET_TYPE(
                    self.args.eval_datasets,
                    tokenizer=self.tokenizer,
                )
            else:
                raise ValueError('Either `eval_datasets` or `eval_split_ratio` should be provided.')

            self.eval_dataloader = DataLoader(
                eval_dataset,
                collate_fn=eval_dataset.get_collator(),
                sampler=DistributedSampler(eval_dataset, shuffle=True),
                batch_size=self.args.per_device_eval_batch_size,
            )
        else:
            self.eval_dataloader = None
        
        self.train_dataloader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.get_collator(),
            sampler=DistributedSampler(train_dataset, shuffle=True),
            batch_size=self.args.per_device_train_batch_size,
        )

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.args.num_update_steps_per_epoch = (
            len(self.train_dataloader) + self.args.gradient_accumulation_steps - 1
        ) // self.args.gradient_accumulation_steps
        self.args.total_training_steps = self.args.epochs * self.args.num_update_steps_per_epoch

        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            self.model,
            self.args.weight_decay,
        )
        if (
            self.ds_config['zero_optimization'].get('offload_optimizer', {}).get('device', 'none')
            != 'none'
        ):
            optimizer = DeepSpeedCPUAdam(
                optimizer_grouped_parameters,
                lr=self.args.lr,
                betas=ADAM_BETAS,
            )
        else:
            optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=self.args.lr,
                betas=ADAM_BETAS,
            )

        num_warmup_steps = int(self.args.lr_warmup_ratio * self.args.total_training_steps)
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.args.total_training_steps,
        )

        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            args=self.args,
            config=self.ds_config,
            lr_scheduler=lr_scheduler,
            dist_init_required=True,
        )

        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    @abc.abstractmethod
    def loss(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute supervised training loss."""
        raise NotImplementedError

    @abc.abstractmethod
    def train_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Perform a single training step."""
        raise NotImplementedError

    
    def eval(self):
        """Evaluate the Model"""
        return {}

        # with torch.no_grad():
        #     pref_margins_res, unpref_margins_res = None, None
        #     gt_margins_res = None
        #     for i, batch in tqdm(enumerate(self.train_dataloader)):
        #         # if(i > 100):
        #         #     break
        #         info = self.loss(
        #             **to_device(batch, self.args.device)
        #         )
        #         # Prepare an empty tensor to hold gathered values
        #         unpref_margins_ = [torch.zeros_like(info['unpref_reward_margin']) for _ in range(dist.get_world_size())]
        #         # Gather costs from all devices
        #         dist.all_gather(unpref_margins_, info['unpref_reward_margin'])
        #         # Concatenate the gathered tensors
        #         unpref_margins_all = torch.cat(unpref_margins_, dim=0).to(self.args.device)

        #         # Prepare an empty tensor to hold gathered values
        #         pref_margins_ = [torch.zeros_like(info['pref_reward_margin']) for _ in range(dist.get_world_size())]
        #         # Gather costs from all devices
        #         dist.all_gather(pref_margins_, info['pref_reward_margin'])
        #         # Concatenate the gathered tensors
        #         pref_margins_all = torch.cat(pref_margins_, dim=0).to(self.args.device)

        #         gt = batch['unpref_gap'].to(self.args.device)
        #         gt_margins_ = [torch.zeros_like(gt) for _ in range(dist.get_world_size())]
        #         # Gather costs from all devices
        #         dist.all_gather(gt_margins_, gt)
        #         # Concatenate the gathered tensors
        #         gt_margins_all = torch.cat(gt_margins_, dim=0).to(self.args.device)

        #         # info = self.train_step(**to_device(batch, self.args.device))
        #         torch.cuda.empty_cache()
        #         if(pref_margins_res is None):
        #             pref_margins_res = pref_margins_all.flatten().detach().cpu()
        #         else:
        #             pref_margins_res = torch.cat((pref_margins_res, pref_margins_all.flatten().detach().cpu()))
        #         if(unpref_margins_res is None):
        #             unpref_margins_res = unpref_margins_all.flatten().detach().cpu()
        #         else:
        #             unpref_margins_res = torch.cat((unpref_margins_res, unpref_margins_all.flatten().detach().cpu()))
        #         if(gt_margins_res is None):
        #             gt_margins_res = gt_margins_all.flatten().detach().cpu()
        #         else:
        #             gt_margins_res = torch.cat((gt_margins_res, gt_margins_all.flatten().detach().cpu()))
            
        # pref_np = pref_margins_res.float().numpy()
        # unpref_np = unpref_margins_res.float().numpy()
        # clamped_margins_np = torch.clamp(unpref_margins_res, min=0, max=10).float().numpy()
        # gt_np = gt_margins_res.float().numpy()
        # np.savez("margins.npz", pref=pref_np, unpref=unpref_np, clamped=clamped_margins_np, gt=gt_np)
        # BINS=25

        # print(f"Pref: {pref_np.shape}, Unpref: {unpref_np.shape} \n\n")
        # plt.figure(figsize=(8, 5))
        # plt.hist(pref_np, bins=BINS, alpha=0.6, label="Preferred Margins", density=True)
        # plt.hist(unpref_np, bins=BINS, alpha=0.6, label="Unpreferred Margins", density=True)
        # plt.hist(clamped_margins_np, bins=BINS, alpha=0.6, label="Clamped Margins", density=True)
        # plt.hist(gt_np, bins=BINS, alpha=0.6, label="GT Margins", density=True)

        # plt.title("Normalized Histogram of Margins")
        # plt.xlabel("Margin")
        # plt.ylabel("Density")
        # plt.legend()
        # # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig('hist_margins.png')

        # fig, axs = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
        # # Preferred margins histogram
        # axs[0].hist(pref_np, bins=BINS, density=True, color='red', alpha=0.7)
        # axs[0].set_title("Preferred Margins")
        # axs[0].set_xlabel("Margin")
        # axs[0].set_ylabel("Density")
        # # axs[0].grid(True)

        # # Unpreferred margins histogram
        # axs[1].hist(unpref_np, bins=BINS, density=True, color='salmon', alpha=0.7)
        # axs[1].set_title("Unpreferred Margins")
        # axs[1].set_xlabel("Margin")
        # # axs[1].grid(True)

        # axs[2].hist(clamped_margins_np, bins=BINS, density=True, color='green', alpha=0.7)
        # axs[2].set_title("Clamped Margins")
        # axs[2].set_xlabel("Margin")

        # axs[3].hist(gt_np, bins=BINS, density=True, color='blue', alpha=0.7)
        # axs[3].set_title("GT Margins")
        # axs[3].set_xlabel("Margin")

        # plt.suptitle("Normalized Histograms of Margins", fontsize=14)
        # plt.tight_layout()
        # plt.savefig('hist_margins_subplot.png')



    def train(self) -> None:
        """Train the model."""

        self.logger.print('***** Running training *****')

        progress_bar = tqdm(
            total=self.args.epochs * len(self.train_dataloader),
            desc=f'Training 1/{self.args.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )

        if self.args.need_eval:
            self.logger.print('\n***** Evaluating at the beginning *****')
            self.logger.log(self.eval(), step=0)

        for epoch in range(self.args.epochs):
            self.model.train()

            for batch in self.train_dataloader:
                info = self.train_step(**to_device(batch, self.args.device))
                torch.cuda.empty_cache()

                self.global_step += 1
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.args.epochs} epoch '
                    f'(loss {info["train/loss"]:.4f})',
                )
                progress_bar.update(1)

                info['train/epoch'] = self.global_step / len(self.train_dataloader)
                self.logger.log(info, step=self.global_step)

                if self.global_step % self.args.save_interval == 0:
                    self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                    self.model.save_checkpoint(self.args.output_dir, tag=self.global_step)
                    self.logger.print('Checkpoint saved.')

                if (
                    self.args.need_eval
                    and self.args.eval_strategy == 'steps'
                    and self.global_step % self.args.eval_interval == 0
                ):
                    self.logger.print(f'\n***** Evaluating at step {self.global_step} *****')
                    self.logger.log(self.eval(), step=self.global_step)

            if self.args.need_eval and self.args.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.args.epochs} *****',
                )
                self.logger.log(self.eval(), step=self.global_step)

            self.model.tput_timer.update_epoch_count()

    def set_train(self, mode: bool = True) -> None:
        """Set training mode for model."""
        if mode:
            self.model.train()
            if self.args.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
        else:
            self.model.eval()
            if self.args.gradient_checkpointing:
                self.model.gradient_checkpointing_disable()
