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
"""Stanford Alpaca dataset for supervised instruction fine-tuning."""

from __future__ import annotations
from typing import ClassVar

from datasets import load_dataset, concatenate_datasets
from safe_rlhf.datasets.base import RawDataset, RawSample


__all__ = ['UFDataset']


class UF_SFT_Dataset(RawDataset):
    NAME: str = 'ultrafeedback-sft'
    ALIASES: tuple[str, ...] = ('ulrafeedback/sft',)

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or 'HuggingFaceH4/ultrafeedback_binarized', split='train_sft')
        self.data = self.data.filter(
            lambda x: len(x['messages'][-1]['content']) > 0
        )

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        # input = f"BEGINNING OF CONVERSATION: USER: {data['prompt']} ASSISTANT:"
        input = data['prompt']
        answer = data['messages'][-1]['content']
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)

class UF_Pref_Filt_Dataset(RawDataset):

    SPLIT: ClassVar[str]

    def __init__(self, path: str | None = None, ) -> None:
        
        # import ipdb; ipdb.set_trace()
        self.data = load_dataset(path or 'HuggingFaceH4/ultrafeedback_binarized', split=self.SPLIT)
        self.data = self.data.filter(
            lambda x: len(x['chosen'][-1]['content']) > 0 and len(x['rejected'][-1]['content']) > 0
        )

        # Define batched filter function
        def filter_invalid_rows(batch):
            mask = [c[-1]['content'] != r[-1]['content'] for c, r in zip(batch["chosen"], batch["rejected"])]
            return mask

        # Apply batched filtering
        self.data = self.data.filter(filter_invalid_rows, batched=True)
        
        # Filter rows with lowest margin
        # Step 1: Compute margins
        margins = [c - r for c, r in zip(self.data["score_chosen"], self.data["score_rejected"])]

        # Step 2: Find the minimum margin
        min_margin = min(margins)

        # Step 3: Filter out rows with that minimum margin
        self.data = self.data.filter(
            lambda x: (x["score_chosen"] - x["score_rejected"]) != min_margin
        )

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = data['prompt']
        answer = data['chosen'][-1]['content']
        other_answer = data['rejected'][-1]['content']
        better = True
        sc = data['score_chosen']
        sr = data['score_rejected']
        return RawSample(input=input, answer=answer, other_answer=other_answer, better=better, score_chosen=sc, score_rejected=sr)

    def __len__(self) -> int:
        return len(self.data)


class UF_Pref_Dataset(RawDataset):
    # NAME: str = 'ultrafeedback-pref'
    # ALIASES: tuple[str, ...] = ('ulrafeedback/pref',)
    SPLIT: ClassVar[str]

    def __init__(self, path: str | None = None, ) -> None:
        
        # import ipdb; ipdb.set_trace()
        self.data = load_dataset(path or 'HuggingFaceH4/ultrafeedback_binarized', split=self.SPLIT)
        self.data = self.data.filter(
            lambda x: len(x['chosen'][-1]['content']) > 0 and len(x['rejected'][-1]['content']) > 0
        )

        # Define batched filter function
        def filter_invalid_rows(batch):
            mask = [c[-1]['content'] != r[-1]['content'] for c, r in zip(batch["chosen"], batch["rejected"])]
            return mask

        # Apply batched filtering
        self.data = self.data.filter(filter_invalid_rows, batched=True)
        # data__ = self.data.filter(lambda x: x['prompt'].startswith('can you rewrite and explain this?'))
        # sdata = self.data.select(list(range(10)))
        # self.data = concatenate_datasets([data__, sdata])
        # Filter only a subset of the data here
        # self.data = self.data.shuffle(seed=42)
        # self.data = self.data.select(range(min(4000,self.data.num_rows)))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = data['prompt']
        answer = data['chosen'][-1]['content']
        other_answer = data['rejected'][-1]['content']
        better = True
        sc = data['score_chosen']
        sr = data['score_rejected']
        return RawSample(input=input, answer=answer, other_answer=other_answer, better=better, score_chosen=sc, score_rejected=sr)

    def __len__(self) -> int:
        return len(self.data)

class UF_Pref_Train_Dataset(UF_Pref_Dataset):
    NAME: str = 'ultrafeedback-pref/train'
    # ALIASES: tuple[str, ...] = ('ultrafeedback/pref/train',)
    SPLIT: str = 'train_prefs'

class UF_Pref_Test_Dataset(UF_Pref_Dataset):
    NAME: str = 'ultrafeedback-pref/test'
    # ALIASES: tuple[str, ...] = ('ultrafeedback/pref/test',)
    SPLIT: str = 'test_prefs'

class UF_Pref_Filt_Train_Dataset(UF_Pref_Filt_Dataset):
    NAME: str = 'ultrafeedback-pref-filt/train'
    # ALIASES: tuple[str, ...] = ('ultrafeedback/pref/train',)
    SPLIT: str = 'train_prefs'

class UF_Pref_Filt_Test_Dataset(UF_Pref_Filt_Dataset):
    NAME: str = 'ultrafeedback-pref-filt/test'
    # ALIASES: tuple[str, ...] = ('ultrafeedback/pref/train',)
    SPLIT: str = 'test_prefs'

# class UF_Pref_Test_Dataset(UF_Pref_Dataset):
#     NAME: str = 'ultrafeedback-pref/test'
#     # ALIASES: tuple[str, ...] = ('ultrafeedback/pref/test',)
#     SPLIT: str = 'test_prefs'

class UF_Pref_No_Margin_Dataset(RawDataset):
    # NAME: str = 'ultrafeedback-pref'
    # ALIASES: tuple[str, ...] = ('ulrafeedback/pref',)
    SPLIT: ClassVar[str]

    def __init__(self, path: str | None = None, ) -> None:
        
        # import ipdb; ipdb.set_trace()
        self.data = load_dataset(path or 'HuggingFaceH4/ultrafeedback_binarized', split=self.SPLIT)
        self.data = self.data.filter(
            lambda x: len(x['chosen'][-1]['content']) > 0 and len(x['rejected'][-1]['content']) > 0
        )

        # Define batched filter function
        def filter_invalid_rows(batch):
            mask = [c[-1]['content'] != r[-1]['content'] for c, r in zip(batch["chosen"], batch["rejected"])]
            return mask

        # Apply batched filtering
        self.data = self.data.filter(filter_invalid_rows, batched=True)
        self.data = self.data.remove_columns(['score_chosen','score_rejected'])
        # data__ = self.data.filter(lambda x: x['prompt'].startswith('can you rewrite and explain this?'))
        # sdata = self.data.select(list(range(10)))
        # self.data = concatenate_datasets([data__, sdata])
        # Filter only a subset of the data here
        # self.data = self.data.shuffle(seed=42)
        # self.data = self.data.select(range(min(4000,self.data.num_rows)))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = data['prompt']
        answer = data['chosen'][-1]['content']
        other_answer = data['rejected'][-1]['content']
        better = True
        return RawSample(input=input, answer=answer, other_answer=other_answer, better=better)

    def __len__(self) -> int:
        return len(self.data)

class UF_Pref_No_Margin_Train_Dataset(UF_Pref_No_Margin_Dataset):
    NAME: str = 'ultrafeedback-pref-nomargin/train'
    # ALIASES: tuple[str, ...] = ('ultrafeedback/pref/train',)
    SPLIT: str = 'train_prefs'


class UF_Pref_No_Margin_Test_Dataset(UF_Pref_No_Margin_Dataset):
    NAME: str = 'ultrafeedback-pref-nomargin/test'
    # ALIASES: tuple[str, ...] = ('ultrafeedback/pref/test',)
    SPLIT: str = 'test_prefs'

class UF_Pref_Filt_No_Margin_Dataset(RawDataset):
    # NAME: str = 'ultrafeedback-pref'
    # ALIASES: tuple[str, ...] = ('ulrafeedback/pref',)
    SPLIT: ClassVar[str]

    def __init__(self, path: str | None = None, ) -> None:
        
        # import ipdb; ipdb.set_trace()
        self.data = load_dataset(path or 'HuggingFaceH4/ultrafeedback_binarized', split=self.SPLIT)
        self.data = self.data.filter(
            lambda x: len(x['chosen'][-1]['content']) > 0 and len(x['rejected'][-1]['content']) > 0
        )

        # Define batched filter function
        def filter_invalid_rows(batch):
            mask = [c[-1]['content'] != r[-1]['content'] for c, r in zip(batch["chosen"], batch["rejected"])]
            return mask

        # Apply batched filtering
        self.data = self.data.filter(filter_invalid_rows, batched=True)

        # Filter rows with lowest margin
        # Step 1: Compute margins
        margins = [c - r for c, r in zip(self.data["score_chosen"], self.data["score_rejected"])]

        # Step 2: Find the minimum margin
        min_margin = min(margins)

        # Step 3: Filter out rows with that minimum margin
        self.data = self.data.filter(
            lambda x: (x["score_chosen"] - x["score_rejected"]) != min_margin
        )

        self.data = self.data.remove_columns(['score_chosen','score_rejected'])

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = data['prompt']
        answer = data['chosen'][-1]['content']
        other_answer = data['rejected'][-1]['content']
        better = True
        return RawSample(input=input, answer=answer, other_answer=other_answer, better=better)

    def __len__(self) -> int:
        return len(self.data)

class UF_Pref_Filt_No_Margin_Train_Dataset(UF_Pref_Filt_No_Margin_Dataset):
    NAME: str = 'ultrafeedback-pref-filt-nomargin/train'
    # ALIASES: tuple[str, ...] = ('ultrafeedback/pref/train',)
    SPLIT: str = 'train_prefs'


class UF_Pref_Filt_No_Margin_Test_Dataset(UF_Pref_Filt_No_Margin_Dataset):
    NAME: str = 'ultrafeedback-pref-filt-nomargin/test'
    # ALIASES: tuple[str, ...] = ('ultrafeedback/pref/test',)
    SPLIT: str = 'test_prefs'