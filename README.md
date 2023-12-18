# Black Box Adversarial Prompting for Foundation Models

## Introduction
This project is a replication and modification of experiments from the paper ["Black Box Adversarial Prompting for Foundation Models"](https://openreview.net/pdf?id=aI5QPjTRbS). It focuses on exploring adversarial prompting in foundation models using Google Colab for its GPU capabilities, as the experiments require significant computational resources.

## Installation

### Prerequisites
- Create a Huggingface account for access tokens [here](https://huggingface.co/settings/tokens).
- Obtain a W&B access token.

### Environment Setup
Use Google Colab to leverage GPU support for running experiments. Check if GPU is available with:

```python
import torch

# Check if GPU is available
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
else:
    print("CPU is being used.")
```

Install required dependencies:

```bash
!pip install transformers torch nltk pandas wandb gpytorch botorch diffusers torchvision
```

## Replication Steps

### Text-to-Text Generation
Run the following commands for **text-to-text generation** with smaller models that are suitable for lower RAM systems:

```bash
!time python3 run_text_exp.py --loss_type perplexity --seed 0 --language_model facebook/opt-350m --embedding_model tinybert --seed_text "Explain list comprehension in Python."
!time python3 run_text_exp.py --loss_type perplexity --seed 0 --language_model facebook/opt-125m --embedding_model tinybert --seed_text "Explain list comprehension in Python."
```

To run the adversarial prompt: It is related on the output observed from the previous command

```bash
!time python3 run_text_exp.py --loss_type perplexity --seed 0 --language_model facebook/opt-350m --embedding_model tinybert --seed_text "usc consumer hen finals Explain list comprehension in Python."
```

### Text-to-Image Generation
For **text-to-image generation**, adjust the query size based on PC requirements. The optimal class (e.g., 'bus') can be changed based on specific needs or as mentioned in the paper:

#### Unrestricted Prompts

```bash
!time python3 image_optimization.py --optimal_class bus --max_allowed_calls_without_progress 1000 --max_n_calls 5000 --seed 0
```

#### Restricted Prompts

```bash
!time python3 image_optimization.py --optimal_class bus --max_allowed_calls_without_progress 1000 --max_n_calls 5000 --seed 0 --exclude_high_similarity_tokens True
```

#### Restricted Prepending Prompts

```bash
!time python3 image_optimization.py --optimal_class bus --max_allowed_calls_without_progress 3000 --max_n_calls 10000 --seed 0 --exclude_high_similarity_tokens True --prepend_task True --prepend_task_version 1
```

To use the Square Attack optimization method, add `--square_attack True` to the command.

## Troubleshooting

### Fixing Bugs in Code
Remove `PerplexityWithSeedLoss` from `run_text_exp.py`. Replace `.cuda()` with `.cpu()` in various files for non-NVIDIA GPUs.

### Avoiding W&B Login Prompt

```bash
! wandb disabled
```

## Conclusion
Methodology alterations may lead to different results from the original paper. This README provides a guide for replicating experiments under specific technical constraints and computational resources.
