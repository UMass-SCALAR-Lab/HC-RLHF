from __future__ import annotations

from typing import ClassVar

import random
from datasets import load_dataset, load_from_disk
from safe_rlhf.datasets.base import RawDataset, RawSamplePoP


__all__ = [
    'UF_PoP_Train_Dataset',
    'UF_PoP_Test_Dataset',
    'UF_PoP_4k_Train_Dataset',
    'UF_PoP_Random_Train_Dataset',
]
## Columns

# ['pref_prompt', 'unpref_prompt', 'pref_chosen', 'unpref_chosen', 'pref_rejected', 'unpref_rejected', 'pref_score_chosen', 
# 'unpref_score_chosen', 'pref_score_rejected', 'unpref_score_rejected', 'pref_gap', 'unpref_gap', 'pop_gap']
class UF_PoP_Dataset(RawDataset):

    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:

        self.data = load_from_disk(path or self.PATH)
        
        # Define batched filter function
        def filter_invalid_rows(batch):
            mask = [(c != r and  uc != ur) for c, r, uc, ur in zip(batch["pref_chosen"], batch["pref_rejected"], batch["unpref_chosen"], batch["unpref_rejected"])]
            return mask

        # Apply batched filtering
        self.data = self.data.filter(filter_invalid_rows, batched=True)

    def __getitem__(self, index: int):
        data = self.data[index]
        return RawSamplePoP(
            pref_prompt = data['pref_prompt'],
            pref_chosen = data['pref_chosen'],
            pref_rejected = data['pref_rejected'],
            unpref_prompt = data['unpref_prompt'],
            unpref_chosen = data['unpref_chosen'],
            unpref_rejected = data['unpref_rejected'],
            pref_score_chosen = data['pref_score_chosen'],
            pref_score_rejected = data['pref_score_rejected'],
            unpref_score_chosen = data['unpref_score_chosen'],
            unpref_score_rejected = data['unpref_score_rejected'],
            pref_gap = data['pref_gap'],
            unpref_gap = data['unpref_gap'],
            pop_gap = data['pop_gap'],
        )

    def __len__(self) -> int:
        return len(self.data)
    
class UF_PoP_Train_Dataset(UF_PoP_Dataset):
    NAME: str = 'UF-PoP/train'
    PATH: str = '/project/pi_sniekum_umass_edu/ychittepu/PoP/data/pref_over_pref/train/full/'


class UF_PoP_Test_Dataset(UF_PoP_Dataset):
    NAME: str = 'UF-PoP/test'
    PATH: str = '/project/pi_sniekum_umass_edu/ychittepu/PoP/data/pref_over_pref/eval/'

class UF_PoP_4k_Train_Dataset(UF_PoP_Dataset):
    NAME: str = 'UF-PoP-4k/train'
    PATH: str = '/project/pi_sniekum_umass_edu/ychittepu/PoP/data/pref_over_pref_4k/train/full/'

class UF_PoP_Random_Train_Dataset(UF_PoP_Dataset):
    NAME: str = 'UF-PoP-Random/train'
    PATH: str = '/project/pi_sniekum_umass_edu/ychittepu/PoP/data/pref_over_pref_random/train/full/'
