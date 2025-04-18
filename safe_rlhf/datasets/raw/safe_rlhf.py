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
"""Safe-RLHF preference datasets."""

from __future__ import annotations

from typing import ClassVar

import random
from datasets import load_dataset
from safe_rlhf.datasets.base import RawDataset, RawSample


__all__ = [
    'SafeRLHFDataset',
    'SafeRLHFTrainDataset',
    'SafeRLHFTestDataset',
    'SafeRLHF30KTrainDataset',
    'SafeRLHF30KTestDataset',
    'SafeRLHF10KTrainDataset',
    'SafeRLHF_YC_Val',
    'SafeRLHF_YC_Safety'
]

class SafeRLHFDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or self.PATH, split=self.SPLIT)

        # For train
        # seed_value = random.randint(0, 1000000)
        # downsample_size = 1000
        # old_size = self.data.num_rows
        # print(f"Shuffling the dataset with seed: {seed_value}")
        # self.data = self.data.shuffle(seed=seed_value)
        # self.data = self.data.select(range(downsample_size))
        # print(f"Truncated the Dataset from size {old_size} to {self.data.num_rows}")

        # For Eval
        # downsample_size = 100
        # old_size = self.data.num_rows
        # print(f"Shuffling the dataset with seed: {42}")
        # self.data = self.data.shuffle(seed=42)
        # self.data = self.data.select(range(downsample_size))
        # print(f"Truncated the Dataset from size {old_size} to {self.data.num_rows}")

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['prompt'],
            answer=data['response_0'],
            other_answer=data['response_1'],
            better=int(data['better_response_id']) == 0,
            safer=int(data['safer_response_id']) == 0,
            is_safe=bool(data['is_response_0_safe']),
            is_other_safe=bool(data['is_response_1_safe']),
        )

    def __len__(self) -> int:
        return len(self.data)


class SafeRLHFTrainDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF/train',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF'
    SPLIT: str = 'train'


class SafeRLHFTestDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF/test'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF/test',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF'
    SPLIT: str = 'test'


class SafeRLHF30KTrainDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF-30K/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-30K/train',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF-30K'
    SPLIT: str = 'train'


class SafeRLHF30KTestDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF-30K/test'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-30K/test',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF-30K'
    SPLIT: str = 'test'


class SafeRLHF10KTrainDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF-10K/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-10K/train',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF-10K'
    SPLIT: str = 'train'

    # def __init__(self, path: str | None = None) -> None:
    #     self.data = load_dataset(path or self.PATH, split=self.SPLIT)
    #     self.data = self.data.shuffle(seed=42)
    #     self.data = self.data.select(range(100))

class SafeRLHF_YC_Val(SafeRLHFDataset):

    NAME: str = 'safety_test/val'
    ALIASES: tuple[str, ...] = ('yaswanthchittepu/safe_rlhf_safety_test/val',)
    PATH: str = 'yaswanthchittepu/safe_rlhf_safety_test'
    SPLIT: str = 'val'

class SafeRLHF_YC_Safety(SafeRLHFDataset):

    NAME: str = 'safety_test/safety'
    ALIASES: tuple[str, ...] = ('yaswanthchittepu/safe_rlhf_safety_test/safety',)
    PATH: str = 'yaswanthchittepu/safe_rlhf_safety_test'
    SPLIT: str = 'safety'
    