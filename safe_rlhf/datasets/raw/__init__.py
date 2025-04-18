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
"""Raw datasets."""

from safe_rlhf.datasets.raw.alpaca import AlpacaDataset
from safe_rlhf.datasets.raw.firefly import FireflyDataset
from safe_rlhf.datasets.raw.hh_rlhf import (
    HhRLHFDialogueDataset,
    HhRLHFHarmlessDialogueDataset,
    HhRLHFHelpfulDialogueDataset,
)
from safe_rlhf.datasets.raw.moss import MOSS002SFT, MOSS003SFT
from safe_rlhf.datasets.raw.safe_rlhf import (
    SafeRLHF10KTrainDataset,
    SafeRLHF30KTestDataset,
    SafeRLHF30KTrainDataset,
    SafeRLHFDataset,
    SafeRLHFTestDataset,
    SafeRLHFTrainDataset,
    SafeRLHF_YC_Val,
    SafeRLHF_YC_Safety,
)
from safe_rlhf.datasets.raw.pop import (
    UF_PoP_Train_Dataset,
    UF_PoP_Test_Dataset,
    UF_PoP_4k_Train_Dataset,
    UF_PoP_Random_Train_Dataset,
)

from safe_rlhf.datasets.raw.uf import (
    UF_SFT_Dataset,
    UF_Pref_Dataset,
    UF_Pref_Train_Dataset,
    UF_Pref_Test_Dataset,
    UF_Pref_No_Margin_Dataset,
    UF_Pref_No_Margin_Test_Dataset,
    UF_Pref_No_Margin_Train_Dataset,
    UF_Pref_Filt_Dataset,
    UF_Pref_Filt_Test_Dataset,
    UF_Pref_Filt_Train_Dataset,
    UF_Pref_Filt_No_Margin_Dataset,
    UF_Pref_Filt_No_Margin_Test_Dataset,
    UF_Pref_Filt_No_Margin_Train_Dataset,
)


__all__ = [
    'AlpacaDataset',
    'FireflyDataset',
    'HhRLHFDialogueDataset',
    'HhRLHFHarmlessDialogueDataset',
    'HhRLHFHelpfulDialogueDataset',
    'MOSS002SFT',
    'MOSS003SFT',
    'SafeRLHFDataset',
    'SafeRLHFTrainDataset',
    'SafeRLHFTestDataset',
    'SafeRLHF30KTrainDataset',
    'SafeRLHF30KTestDataset',
    'SafeRLHF10KTrainDataset',
    'SafeRLHF_YC_Val',
    'SafeRLHF_YC_Safety',
    'UF_PoP_Train_Dataset',
    'UF_PoP_Test_Dataset',
    'UF_SFT_Dataset',
    'UF_Pref_Dataset',
    'UF_Pref_Train_Dataset',
    'UF_Pref_Test_Dataset',
    'UF_PoP_4k_Train_Dataset',
    'UF_PoP_Random_Train_Dataset',
    'UF_Pref_No_Margin_Dataset',
    'UF_Pref_No_Margin_Test_Dataset',
    'UF_Pref_No_Margin_Train_Dataset',
    'UF_Pref_Filt_Dataset',
    'UF_Pref_Filt_Test_Dataset',
    'UF_Pref_Filt_Train_Dataset',
    'UF_Pref_Filt_No_Margin_Dataset',
    'UF_Pref_Filt_No_Margin_Test_Dataset',
    'UF_Pref_Filt_No_Margin_Train_Dataset',
]

# __all__ = []
