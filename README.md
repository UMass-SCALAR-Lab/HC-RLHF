<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->


This repository was built on top of [Safe-RLHF](https://github.com/PKU-Alignment/safe-rlhf). We added methods to do Reinforce on a Constrained Optimizatation objective. Additionally techniques to use Adaptive margins in the DPO loss have also been implemented

## Training

`safe-rlhf` supports a complete pipeline from Supervised Fine-Tuning (SFT) to preference model training to RLHF alignment training.

0. Follow the instructions in section [Installation](#installation) to setup the training environment properly.

```bash
conda activate safe-rlhf
export WANDB_API_KEY="..."  # your W&B API key here
```

or

```bash
make docker-run
export WANDB_API_KEY="..."  # your W&B API key here
```

1. Supervised Fine-Tuning (SFT)

```bash
bash scripts/sft.sh \
    --model_name_or_path <your-model-name-or-checkpoint-path> \
    --output_dir output/sft
```

NOTE: You may need to update some of the parameters in the script according to your machine setup, such as the number of GPUs for training, the training batch size, etc.

2. Value Models (reward model & cost model)

```bash
bash scripts/reward-model.sh \
    --model_name_or_path output/sft \
    --output_dir output/rm
```

```bash
bash scripts/cost-model.sh \
    --model_name_or_path output/sft \
    --output_dir output/cm
```

3. RLHF (Optional)

```bash
bash scripts/ppo.sh \
    --actor_model_name_or_path output/sft \
    --reward_model_name_or_path output/rm \
    --output_dir output/ppo
```

4. Safe-RLHF

```bash
bash scripts/ppo-lag.sh \
    --actor_model_name_or_path output/sft \
    --reward_model_name_or_path output/rm \
    --cost_model_name_or_path output/cm \
    --output_dir output/ppo-lag
```

5. HC-RLHF

For other hyperparameters, that you might wish to change, refer to scripts/reinforce-ci-lag.sh

```bash
bash scripts/reinforce-ci-lag.sh \
    --actor_model_name_or_path output/sft \
    --reward_model_name_or_path output/rm \
    --cost_model_name_or_path output/cm \
    --output_dir output/ppo-lag \
    --num_samples_per_prompt 2 \
    --rloo True \
```

An example of commands to run the whole pipeline with [LLaMA-7B](https://ai.facebook.com/blog/large-language-model-llama-meta-ai):

```bash
conda activate safe-rlhf
bash scripts/sft.sh --model_name_or_path ~/models/llama-7b --output_dir output/sft
bash scripts/reward-model.sh --model_name_or_path output/sft --output_dir output/rm
bash scripts/cost-model.sh --model_name_or_path output/sft --output_dir output/cm
bash scripts/ppo-lag.sh \
    --actor_model_name_or_path output/sft \
    --reward_model_name_or_path output/rm \
    --cost_model_name_or_path output/cm \
    --output_dir output/ppo-lag
```


## Custom Datasets

`safe-rlhf` provides an abstraction to create datasets for all of the Supervised Fine-Tuning, preference model training, and RL training stages.

```python
class RawSample(TypedDict, total=False):
    """Raw sample type.

    For SupervisedDataset, should provide (input, answer) or (dialogue).
    For PreferenceDataset, should provide (input, answer, other_answer, better).
    For SafetyPreferenceDataset, should provide (input, answer, other_answer, safer, is_safe, is_other_safe).
    For PromptOnlyDataset, should provide (input).
    """

    # Texts
    input: NotRequired[str]  # either `input` or `dialogue` should be provided
    """User input text."""
    answer: NotRequired[str]
    """Assistant answer text."""
    other_answer: NotRequired[str]
    """Other assistant answer text via resampling."""
    dialogue: NotRequired[list[str]]  # either `input` or `dialogue` should be provided
    """Dialogue history."""

    # Flags
    better: NotRequired[bool]
    """Whether ``answer`` is better than ``other_answer``."""
    safer: NotRequired[bool]
    """Whether ``answer`` is safer than ``other_answer``."""
    is_safe: NotRequired[bool]
    """Whether ``answer`` is safe."""
    is_other_safe: NotRequired[bool]
    """Whether ``other_answer`` is safe."""
```

Here is an example to implement a custom dataset (see [safe_rlhf/datasets/raw](safe_rlhf/datasets/raw) for more examples):

```python
import argparse
from datasets import load_dataset
from safe_rlhf.datasets import RawDataset, RawSample, parse_dataset


class MyRawDataset(RawDataset):
    NAME = 'my-dataset-name'

    def __init__(self, path=None) -> None:
        # Load a dataset from Hugging Face
        self.data = load_dataset(path or 'my-organization/my-dataset')['train']

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        # Construct a `RawSample` dictionary from your custom dataset item
        return RawSample(
            input=data['col1'],
            answer=data['col2'],
            other_answer=data['col3'],
            better=float(data['col4']) > float(data['col5']),
            ...
        )

    def __len__(self) -> int:
        return len(self.data)  # dataset size


def parse_arguments():
    parser = argparse.ArgumentParser(...)
    parser.add_argument(
        '--datasets',
        type=parse_dataset,
        nargs='+',
        metavar='DATASET[:PROPORTION[:PATH]]',
    )
    ...
    return parser.parse_args()


def main():
    args = parse_arguments()
    ...


if __name__ == '__main__':
    main()
```

Then you can pass this dataset to the training scripts as:

```bash
python3 train.py --datasets my-dataset-name
```

You may also pass multiple datasets with optionally additional dataset proportions (separated by a colon `:`). For example:

```bash
python3 train.py --datasets alpaca:0.75 my-dataset-name:0.5
```

This will use randomly split 75% of the Stanford Alpaca dataset and 50% of your custom dataset.

In addition, the dataset argument can also be followed by a local path (separated by a colon `:`) if you have already cloned the dataset repository from Hugging Face.

```bash
git lfs install
git clone https://huggingface.co/datasets/my-organization/my-dataset ~/path/to/my-dataset/repository
python3 train.py --datasets alpaca:0.75 my-dataset-name:0.5:~/path/to/my-dataset/repository
```

NOTE: The dataset class must be imported before the training script begins to parse the command line arguments.

## Benchmark and Evaluation

### Arena via Reward and Cost Models

```bash
scripts/arena-evaluation.sh \
    --red_corner_model_name_or_path output/sft \
    --blue_corner_model_name_or_path output/ppo-lag \
    --reward_model_name_or_path output/rm \
    --cost_model_name_or_path output/cm \
    --output_dir output/arena-evaluation
```

### BIG-bench

```bash
# Install BIG-bench
git clone https://github.com/google/BIG-bench.git
(
    cd BIG-bench
    python3 setup.py sdist
    python3 -m pip install -e .
)

# BIG-bench evaluation
python3 -m safe_rlhf.evaluate.bigbench \
    --model_name_or_path output/ppo-lag \
    --task_name <BIG-bench-task-name>
```

### GPT-4 Evaluation

```bash
# Install OpenAI Python API
pip3 install openai
export OPENAI_API_KEY="..."  # your OpenAI API key here

# GPT-4 evaluation
python3 -m safe_rlhf.evaluate.gpt4 \
    --red_corner_model_name_or_path output/sft \
    --blue_corner_model_name_or_path output/ppo-lag
```
