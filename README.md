# Foundation-Model-Tools

A collection of scripts and utilities for fine-tuning genomic foundation models, with a focus on DNABERT-2 for sequence classification tasks on SLURM-based HPC clusters.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Fine-Tuning with DNABERT-2](#fine-tuning-with-dnabert-2)
  - [Data Format](#data-format)
  - [SLURM Job Submission](#slurm-job-submission)
  - [Key Training Parameters](#key-training-parameters)
- [Model & Training Details](#model--training-details)
  - [LoRA Support](#lora-support)
  - [Evaluation Metrics](#evaluation-metrics)
- [Citation](#citation)

---

## Overview

This repository provides tools for fine-tuning [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2), a genomic foundation model trained on large-scale multi-species genomes. DNABERT-2 achieves state-of-the-art performance across 28 tasks in the Genome Understanding Evaluation (GUE) benchmark. It replaces k-mer tokenization with Byte Pair Encoding (BPE) and uses Attention with Linear Bias (ALiBi) in place of standard positional embeddings.

The fine-tuning pipeline here supports:
- Single sequence classification
- Sequence-pair classification
- Optional k-mer tokenization (legacy mode)
- Optional LoRA (Low-Rank Adaptation) for parameter-efficient training

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/your-org/Foundation-Model-Tools.git
cd Foundation-Model-Tools
```

### 2. Clone the DNABERT-2 repository

```bash
git clone https://github.com/MAGICS-LAB/DNABERT_2.git
cd DNABERT_2
```

### 3. Create and activate a conda environment

```bash
conda create -n dna python=3.8
conda activate dna
```

### 4. (Optional) Install Flash Attention via Triton

Flash Attention can significantly speed up training. Skip this block if you don't need it.

```bash
git clone https://github.com/openai/triton.git
cd triton/python
pip install cmake        # build-time dependency
pip install -e .
cd ../..
```

### 5. Install required Python packages

```bash
python3 -m pip install -r requirements.txt
```

### 6. Verify the installation

You can quickly verify DNABERT-2 loads correctly with the following snippet:

```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M",
    trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "zhihan1996/DNABERT-2-117M",
    trust_remote_code=True
)

dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors="pt")["input_ids"]
hidden_states = model(inputs)[0]

embedding_mean = torch.mean(hidden_states[0], dim=0)
print(f"Embedding shape: {embedding_mean.shape}")  # Expected: torch.Size([768])
```

> **Note:** The pre-trained model weights are hosted on HuggingFace as [`zhihan1996/DNABERT-2-117M`](https://huggingface.co/zhihan1996/DNABERT-2-117M) and will be downloaded automatically on first use.

---

## Repository Structure

```
Foundation-Model-Tools/
├── DNABERT_2/
│   ├── finetune/
│   │   └── train.py          # Main fine-tuning script
│   ├── sample_data/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   └── test.csv
│   ├── requirements.txt
│   └── ...
├── slurm_finetune.sh          # SLURM job submission script
├── JSON_LOGGER.py             # Post-training JSON logging utility
└── README.md
```

---

## Fine-Tuning with DNABERT-2

### Data Format

Training data should be provided as `.csv` files with a header row, placed in a single directory containing `train.csv`, `dev.csv`, and `test.csv`.

**Single sequence classification** (2 columns):
```
sequence,label
ACGT...,0
TGCA...,1
```

**Sequence-pair classification** (3 columns):
```
sequence1,sequence2,label
ACGT...,TGCA...,0
```

Labels must be integers starting from `0`.

### SLURM Job Submission

The `slurm_finetune.sh` script submits a fine-tuning job to a SLURM cluster. Edit the configuration block at the top to match your cluster's partition and resource availability, then submit with:

```bash
sbatch slurm_finetune.sh
```

Key configuration variables in the script:

| Variable | Description | Default |
|---|---|---|
| `DATA_PATH` | Path to the directory containing your CSV data files | `./sample_data` |
| `MAX_LENGTH` | Maximum token length — set to `0.25 × sequence length` | `50` |
| `LR` | Learning rate | `3e-5` |
| `--partition` | SLURM partition name | `nvidia-L40S-20` |
| `--cpus-per-task` | Number of CPU threads | `8` |
| `--mem` | Memory allocation | `62gb` |
| `--gres=gpu:1` | GPU resource request | 1 GPU |

After training completes, `JSON_LOGGER.py` is called automatically to log results.

### Key Training Parameters

The table below summarizes the main arguments passed to `train.py`:

| Argument | Value | Description |
|---|---|---|
| `--model_name_or_path` | `zhihan1996/DNABERT-2-117M` | Pre-trained model to fine-tune |
| `--kmer` | `-1` | `-1` disables k-mer tokenization (recommended for DNABERT-2) |
| `--model_max_length` | `50` | Max sequence length in tokens |
| `--per_device_train_batch_size` | `8` | Training batch size per GPU |
| `--per_device_eval_batch_size` | `16` | Evaluation batch size per GPU |
| `--learning_rate` | `3e-5` | AdamW learning rate |
| `--num_train_epochs` | `25` | Number of training epochs |
| `--fp16` | `True` | Mixed-precision (FP16) training |
| `--save_steps` | `200` | Checkpoint save frequency |
| `--eval_steps` | `200` | Evaluation frequency |
| `--warmup_steps` | `50` | Linear warmup steps |
| `--output_dir` | `output/dnabert2` | Directory for checkpoints and results |

---

## Model & Training Details

### LoRA Support

The `train.py` script has built-in support for LoRA (Low-Rank Adaptation), which allows parameter-efficient fine-tuning by training only a small fraction of the model's weights. To enable it, add the following flags to your training command:

```bash
--use_lora True \
--lora_r 8 \
--lora_alpha 32 \
--lora_dropout 0.05 \
--lora_target_modules "query,value"
```

This is recommended when GPU memory is limited or when fine-tuning on small datasets to reduce overfitting.

### Evaluation Metrics

At the end of training, the model is evaluated on `test.csv` and the following metrics are saved to `output/dnabert2/results/<run_name>/eval_results.json`:

- **Accuracy**
- **F1 Score** (macro-averaged)
- **Matthews Correlation Coefficient (MCC)**
- **Precision** (macro-averaged)
- **Recall** (macro-averaged)

---

## Citation

If you use DNABERT-2 in your work, please cite the original paper:

```bibtex
@article{zhou2023dnabert2,
    author = {Zhou, Zhihan and Ji, Yanrong and Li, Weijian and Dutta, Pratik and Davuluri, Ramana and Liu, Han},
    title  = {DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome},
    journal = {ICLR},
    year   = {2024}
}
```
