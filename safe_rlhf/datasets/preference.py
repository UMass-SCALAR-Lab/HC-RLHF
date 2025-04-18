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
"""Dataset class for preference training."""

from __future__ import annotations

from typing import Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch

from safe_rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset, RawSamplePoP
from safe_rlhf.datasets.utils import format_prompt, right_padding


__all__ = [
    'PreferenceDataset',
    'PreferenceCollator',
    'PreferenceSample',
    'PreferenceBatch',
]


class PreferenceSample(TypedDict, total=True):
    better_input_ids: torch.LongTensor  # size = (L,)
    worse_input_ids: torch.LongTensor  # size = (L,)


class PreferenceBatch(TypedDict, total=True):
    better_input_ids: torch.LongTensor  # size = (B, L)
    better_attention_mask: torch.BoolTensor  # size = (B, L)

    worse_input_ids: torch.LongTensor  # size = (B, L)
    worse_attention_mask: torch.BoolTensor  # size = (B, L)

class PrefOverPrefSample(TypedDict, total=True):
    pref_chosen_input_ids: torch.LongTensor # size = (L,)
    pref_rejected_input_ids: torch.LongTensor
    unpref_chosen_input_ids: torch.LongTensor
    unpref_rejected_input_ids: torch.LongTensor
    pref_score_chosen: float
    pref_score_rejected: float
    unpref_score_chosen: float
    unpref_score_rejected: float
    pref_gap: float
    unpref_gap: float
    pop_gap: float


class PrefOverPrefBatch(TypedDict, total=True):

    pref_chosen_input_ids: torch.LongTensor # size = (B,L)
    pref_chosen_attention_mask: torch.BoolTensor # size = (B,L)
    pref_rejected_input_ids: torch.LongTensor 
    pref_rejected_attention_mask: torch.BoolTensor
    unpref_chosen_input_ids: torch.LongTensor
    unpref_chosen_attention_mask: torch.BoolTensor
    unpref_rejected_input_ids: torch.LongTensor
    unpref_rejected_attention_mask: torch.BoolTensor
    pref_score_chosen: torch.FloatTensor # size = (B,)
    pref_score_rejected: torch.FloatTensor
    unpref_score_chosen: torch.FloatTensor
    unpref_score_rejected: torch.FloatTensor
    pref_gap: torch.FloatTensor
    unpref_gap: torch.FloatTensor
    pop_gap: torch.FloatTensor

class PreferenceDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> PreferenceSample:
        
        prompt = format_prompt(input=raw_sample['input'], eos_token=self.tokenizer.eos_token)
        better_answer = raw_sample['answer']
        worse_answer = raw_sample['other_answer']
        score_chosen, score_rejected = None, None
        try:
            score_chosen = raw_sample['score_chosen']
            score_rejected = raw_sample['score_rejected']
        except:
            pass
        better = raw_sample['better']
        if not better:
            better_answer, worse_answer = worse_answer, better_answer

        better_input_ids = self.tokenize(prompt + better_answer + self.tokenizer.eos_token)
        worse_input_ids = self.tokenize(prompt + worse_answer + self.tokenizer.eos_token)
        # if (
        #     better_input_ids.size() == worse_input_ids.size()
        #     and torch.all(torch.eq(better_input_ids, worse_input_ids)).item()
        # ):
        #     raise ValueError(
        #         'Two responses get the same `input_ids` after tokenization.\n\n'
        #         f'Prompt: {prompt}\n\n'
        #         f'Better answer: {better_answer}\n\n'
        #         f'Worse answer: {worse_answer}',
        #     )
        if(score_chosen is None or score_rejected is None):
            return {
                'better_input_ids': better_input_ids,  # size = (L,)
                'worse_input_ids': worse_input_ids,  # size = (L,)
            }
        else:
            return {
                'better_input_ids': better_input_ids,  # size = (L,)
                'worse_input_ids': worse_input_ids,  # size = (L,)
                'margin': score_chosen-score_rejected,
            }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PreferenceCollator(self.tokenizer.pad_token_id)


class PreferenceCollator(CollatorBase):
    def __call__(self, samples: list[PreferenceSample]) -> PreferenceBatch:
        input_ids = [sample['better_input_ids'] for sample in samples] + [
            sample['worse_input_ids'] for sample in samples
        ]  # size = (2 * B, L)
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]  # size = (2 * B, L)

        input_ids = right_padding(input_ids, padding_value=self.pad_token_id)  # size = (2 * B, L)
        attention_mask = right_padding(attention_mask, padding_value=0)  # size = (2 * B, L)

        (
            better_input_ids,  # size = (B, L)
            worse_input_ids,  # size = (B, L)
        ) = input_ids.chunk(chunks=2, dim=0)
        (
            better_attention_mask,  # size = (B, L)
            worse_attention_mask,  # size = (B, L)
        ) = attention_mask.chunk(chunks=2, dim=0)

        margin = [sample.get('margin', None) for sample in samples]
        if(margin[0] is None):
            return {
                'better_input_ids': better_input_ids,  # size = (B, L)
                'better_attention_mask': better_attention_mask,  # size = (B, L)
                'worse_input_ids': worse_input_ids,  # size = (B, L)
                'worse_attention_mask': worse_attention_mask,  # size = (B, L)
            }
        else:
            return {
                'better_input_ids': better_input_ids,  # size = (B, L)
                'better_attention_mask': better_attention_mask,  # size = (B, L)
                'worse_input_ids': worse_input_ids,  # size = (B, L)
                'worse_attention_mask': worse_attention_mask,  # size = (B, L)
                'margin': torch.tensor(margin),
            }

class PrefOverPrefDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSamplePoP) -> PrefOverPrefSample:

        # pref_prompt = data['pref_prompt'],
        # pref_chosen = data['pref_chosen'],
        # pref_rejected = data['pref_rejected'],
        # unpref_prompt = data['unpref_prompt'],
        # unpref_chosen = data['unpref_chosen'],
        # unpref_rejected = data['unpref_rejected'],
        # pref_score_chosen = data['pref_score_chosen'],
        # pref_score_rejected = data['pref_score_rejected'],
        # unpref_score_chosen = data['unpref_score_chosen'],
        # unpref_score_rejected = data['unpref_score_rejected'],
        # pref_gap = data['pref_gap'],
        # unpref_gap = data['unpref_gap'],
        # pop_gap = data['pop_gap'],

        pref_prompt = format_prompt(input=raw_sample['pref_prompt'], eos_token=self.tokenizer.eos_token)
        unpref_prompt = format_prompt(input=raw_sample['unpref_prompt'], eos_token=self.tokenizer.eos_token)
        pref_chosen_input_ids = self.tokenize(pref_prompt + raw_sample['pref_chosen'] + self.tokenizer.eos_token)
        pref_rejected_input_ids = self.tokenize(pref_prompt + raw_sample['pref_rejected'] + self.tokenizer.eos_token)
        unpref_chosen_input_ids = self.tokenize(unpref_prompt + raw_sample['unpref_chosen'] + self.tokenizer.eos_token)
        unpref_rejected_input_ids = self.tokenize(unpref_prompt + raw_sample['unpref_rejected'] + self.tokenizer.eos_token)

        # prompt = format_prompt(input=raw_sample['prompt'], eos_token=self.tokenizer.eos_token)
        # better_answer = raw_sample['answer']
        # worse_answer = raw_sample['other_answer']
        # better = raw_sample['better']
        # if not better:
        #     better_answer, worse_answer = worse_answer, better_answer

        # better_input_ids = self.tokenize(prompt + better_answer + self.tokenizer.eos_token)
        # worse_input_ids = self.tokenize(prompt + worse_answer + self.tokenizer.eos_token)
        # if (
        #     better_input_ids.size() == worse_input_ids.size()
        #     and torch.all(torch.eq(better_input_ids, worse_input_ids)).item()
        # ):
        #     raise ValueError(
        #         'Two responses get the same `input_ids` after tokenization.\n\n'
        #         f'Prompt: {prompt}\n\n'
        #         f'Better answer: {better_answer}\n\n'
        #         f'Worse answer: {worse_answer}',
        #     )
        # return {
        #     'better_input_ids': better_input_ids,  # size = (L,)
        #     'worse_input_ids': worse_input_ids,  # size = (L,)
        # }

        return {
            'pref_chosen_input_ids': pref_chosen_input_ids,
            'pref_rejected_input_ids': pref_rejected_input_ids,
            'unpref_chosen_input_ids': unpref_chosen_input_ids,
            'unpref_rejected_input_ids': unpref_rejected_input_ids,
            'pref_score_chosen': raw_sample['pref_score_chosen'],
            'pref_score_rejected': raw_sample['pref_score_rejected'],
            'unpref_score_chosen': raw_sample['unpref_score_chosen'],
            'unpref_score_rejected': raw_sample['unpref_score_rejected'],
            'pref_gap': raw_sample['pref_gap'],
            'unpref_gap': raw_sample['unpref_gap'],
            'pop_gap': raw_sample['pop_gap'],
        }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PrefOverPrefCollator(self.tokenizer.pad_token_id)

class PrefOverPrefCollator(CollatorBase):
    def __call__(self, samples: list[PrefOverPrefSample]) -> PrefOverPrefBatch:
        
        ## Pref
        input_ids = [sample['pref_chosen_input_ids'] for sample in samples] + [
            sample['pref_rejected_input_ids'] for sample in samples
        ]  + [sample['unpref_chosen_input_ids'] for sample in samples] + [
            sample['unpref_rejected_input_ids'] for sample in samples
        ]  # size = (4 * B, L)
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]  # size = (4 * B, L)

        input_ids = right_padding(input_ids, padding_value=self.pad_token_id)  # size = (2 * B, L)
        attention_mask = right_padding(attention_mask, padding_value=0)  # size = (2 * B, L)

        (
            pref_chosen_input_ids,  # size = (B, L)
            pref_rejected_input_ids,  # size = (B, L)
            unpref_chosen_input_ids,  # size = (B, L)
            unpref_rejected_input_ids,  # size = (B, L)
        ) = input_ids.chunk(chunks=4, dim=0)
        (
            pref_chosen_attention_mask,  # size = (B, L)
            pref_rejected_attention_mask,  # size = (B, L)
            unpref_chosen_attention_mask,  # size = (B, L)
            unpref_rejected_attention_mask,  # size = (B, L)
        ) = attention_mask.chunk(chunks=4, dim=0)

        return {
            'pref_chosen_input_ids': pref_chosen_input_ids,  # size = (B, L)
            'pref_chosen_attention_mask': pref_chosen_attention_mask,  # size = (B, L)
            'pref_rejected_input_ids': pref_rejected_input_ids,  # size = (B, L)
            'pref_rejected_attention_mask': pref_rejected_attention_mask,  # size = (B, L)
            'unpref_chosen_input_ids': unpref_chosen_input_ids,  # size = (B, L)
            'unpref_chosen_attention_mask': unpref_chosen_attention_mask,  # size = (B, L)
            'unpref_rejected_input_ids': unpref_rejected_input_ids,  # size = (B, L)
            'unpref_rejected_attention_mask': unpref_rejected_attention_mask,  # size = (B, L)
            'pref_score_chosen': torch.tensor([sample['pref_score_chosen'] for sample in samples]),
            'pref_score_rejected': torch.tensor([sample['pref_score_rejected'] for sample in samples]),
            'unpref_score_chosen': torch.tensor([sample['unpref_score_chosen'] for sample in samples]),
            'unpref_score_rejected': torch.tensor([sample['unpref_score_rejected'] for sample in samples]),
            'pref_gap': torch.tensor([sample['pref_gap'] for sample in samples]),
            'unpref_gap': torch.tensor([sample['unpref_gap'] for sample in samples]),
            'pop_gap': torch.tensor([sample['pop_gap'] for sample in samples]),
        }