## Installation
 
### 1. Clone this repository
 
```bash
git clone https://github.com/your-org/Foundation-Model-Tools.git
cd Foundation-Model-Tools
```
 
### 2. Clone DNABERT-2
 
```bash
git clone https://github.com/MAGICS-LAB/DNABERT_2.git
```
 
### 3. Create and activate a conda environment
 
```bash
conda create -n dna python=3.8
conda activate dna
```
 
### 4. (Optional) Install Flash Attention via Triton
 
Flash Attention can significantly speed up training on compatible GPUs. Skip if not needed.
 
```bash
git clone https://github.com/openai/triton.git
cd triton/python
pip install cmake       # build-time dependency
pip install -e .
cd ../..
```
 
### 5. Install required packages
 
```bash
python3 -m pip install -r requirements.txt
```
 
A minimal `requirements.txt` for all scripts in this repository:
 
```
torch>=2.0
transformers>=4.38
peft>=0.9
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
tqdm
```
 
### 6. Verify the installation
 
```python
import torch
from transformers import AutoTokenizer, AutoModel
 
tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True
)
 
dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors="pt")["input_ids"]
hidden_states = model(inputs)[0]
embedding_mean = torch.mean(hidden_states[0], dim=0)
print(f"Embedding shape: {embedding_mean.shape}")  # torch.Size([768])
```
 
> The pre-trained weights are hosted on HuggingFace as [`zhihan1996/DNABERT-2-117M`](https://huggingface.co/zhihan1996/DNABERT-2-117M) and downloaded automatically on first use.
 
---
 
## Repository Structure
 
```
Foundation-Model-Tools/
├── DNABERT_2/
│   ├── finetune/
│   │   └── train.py               # DNABERT-2 fine-tuning script
│   ├── sample_data/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   └── test.csv
│   └── requirements.txt
├── slurm_finetune.sh              # SLURM job submission script
├── run_motif_injection.py         # Motif injection & flip-rate analysis
├── run_logistic_baseline.py       # k-mer logistic regression baselines
├── log_eval_results.py            # Append eval metrics to summary CSV
└── README.md
```
 
---
 
## Fine-Tuning with DNABERT-2
 
### Data Format
 
Place `train.csv`, `dev.csv`, and `test.csv` in a single directory. Each file requires a header row.
 
**Single sequence classification** (2 columns):
 
```
sequence,label
ACGTACGTACGT...,0
TGCATGCATGCA...,1
```
 
**Sequence-pair classification** (3 columns):
 
```
sequence1,sequence2,label
ACGT...,TGCA...,0
```
 
Labels must be integers starting from `0`.
 
### SLURM Job Submission
 
Edit the configuration variables at the top of `slurm_finetune.sh` to match your cluster and data paths, then submit with:
 
```bash
sbatch slurm_finetune.sh
```
 
Key configuration variables:
 
| Variable | Description | Default |
|---|---|---|
| `DATA_PATH` | Directory containing your CSV splits | `./sample_data` |
| `MAX_LENGTH` | Max token length — set to `0.25 × sequence length` | `50` |
| `LR` | Learning rate | `3e-5` |
| `--partition` | SLURM partition name | `nvidia-L40S-20` |
| `--cpus-per-task` | CPU thread count | `8` |
| `--mem` | Memory request | `62gb` |
| `--gres=gpu:1` | Number of GPUs | `1` |
 
After training, `log_eval_results.py` is called automatically to append metrics to the summary CSV.
 
### Key Training Parameters
 
| Argument | Value | Description |
|---|---|---|
| `--model_name_or_path` | `zhihan1996/DNABERT-2-117M` | Pre-trained model checkpoint |
| `--kmer` | `-1` | Disables legacy k-mer tokenization (use BPE) |
| `--model_max_length` | `50` | Max token sequence length |
| `--per_device_train_batch_size` | `8` | Training batch size per GPU |
| `--per_device_eval_batch_size` | `16` | Evaluation batch size per GPU |
| `--learning_rate` | `3e-5` | AdamW optimizer learning rate |
| `--num_train_epochs` | `25` | Total training epochs |
| `--fp16` | `True` | Mixed-precision (FP16) training |
| `--save_steps` | `200` | Checkpoint save frequency (steps) |
| `--eval_steps` | `200` | Evaluation frequency (steps) |
| `--warmup_steps` | `50` | Linear warmup steps |
| `--output_dir` | `output/dnabert2` | Checkpoint and results directory |
 
---
 
## Model & Training Details
 
### LoRA Support
 
`train.py` supports LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning — useful when GPU memory is limited or datasets are small. Add these flags to your training command:
 
```bash
--use_lora True \
--lora_r 8 \
--lora_alpha 32 \
--lora_dropout 0.05 \
--lora_target_modules "query,value"
```
 
### Evaluation Metrics
 
After training, the model is evaluated on `test.csv`. Metrics are saved to `output/dnabert2/results/<run_name>/eval_results.json`:
 
- **Accuracy**
- **F1 Score** (macro-averaged)
- **Matthews Correlation Coefficient (MCC)**
- **Precision** (macro-averaged)
- **Recall** (macro-averaged)
---
 
## Motif Injection Analysis
 
> **If you use the motif injection analysis in your work, please cite:**
> Qureshi et al., *Heterogeneous epigenetic variation converges on splicing dysregulation in opioid addiction*, medRxiv (2025). https://doi.org/10.64898/2025.12.20.25342745
 
`run_motif_injection.py` probes model sensitivity to specific short DNA sequences by injecting motifs into test sequences and measuring how often the predicted class flips.
 
**What it does:**
 
1. Loads a fine-tuned DNABERT-2 model and a test CSV.
2. Runs inference on all original sequences.
3. For each motif, replaces a sub-string at a specified position (default: sequence centre) with the motif.
4. Runs inference on each modified sequence.
5. Reports per-motif **flip rates** in both directions (0→1 and 1→0).
**Usage:**
 
```bash
python run_motif_injection.py \
    --input_csv     data/test.csv \
    --model_path    output/dnabert2 \
    --output_csv    results/motif_results.csv \
    --motifs        ATCG,GCTA,TTAA \
    --motif_position 25     # optional; defaults to sequence centre
```
 
**Outputs:**
 
| File | Contents |
|---|---|
| `results/motif_results.csv` | Per-sequence predictions for original and all motif-modified sequences |
| `results/motif_results_flip_stats.csv` | Per-motif flip-rate summary (counts + rates for 0→1 and 1→0) |
 
**Flip-rate summary columns:**
 
| Column | Description |
|---|---|
| `Motif` | The injected motif string |
| `Total_1s` / `Total_0s` | Sequences originally predicted as class 1 / 0 |
| `Flips_1_to_0` / `Flips_0_to_1` | Count of label changes after injection |
| `FlipRate_1_to_0` / `FlipRate_0_to_1` | Fraction of sequences that flipped |
 
A high `FlipRate_1_to_0` for a given motif suggests the model associates that motif strongly with class 0 — biologically interpretable as a regulatory signal or binding site.
 
---
 
## Logistic Regression Baselines
 
`run_logistic_baseline.py` trains classical logistic regression models using k-mer frequency vectors and GC content as features, providing a principled baseline to contextualise DNABERT-2 performance.
 
**Feature set:**
 
- **k-mer relative frequencies** — each sequence is represented as a vector of normalised k-mer counts. Multiple values of k can be combined (default: 3-mer, 4-mer, 6-mer).
- **GC content** — the fraction of G/C nucleotides in the sequence (single scalar).
Two model variants are evaluated: raw features and z-score scaled features. If a DNABERT-2 `eval_results.json` is provided, it is included in the comparison table automatically.
 
**Usage:**
 
```bash
# Basic baseline
python run_logistic_baseline.py \
    --train_csv  data/train.csv \
    --test_csv   data/test.csv  \
    --output_csv results/baseline_results.csv \
    --kmers      3,4,6
 
# Side-by-side comparison with DNABERT-2
python run_logistic_baseline.py \
    --train_csv       data/train.csv \
    --test_csv        data/test.csv  \
    --output_csv      results/baseline_results.csv \
    --dnabert_results output/dnabert2/results/DNABERT2_run/eval_results.json
```
 
**Key arguments:**
 
| Argument | Default | Description |
|---|---|---|
| `--kmers` | `3,4,6` | Comma-separated k values for feature construction |
| `--no_gc` | `False` | Exclude GC-content feature |
| `--C` | `1.0` | Inverse regularisation strength |
| `--max_iter` | `1000` | Solver iteration limit |
| `--dnabert_results` | `None` | Path to DNABERT-2 JSON for comparison |
 
**Example output:**
 
```
=== Model Comparison ===
model                                accuracy  f1_macro  precision_macro  recall_macro  matthews_correlation
LogReg_raw_k3_4_6                       0.712     0.698            0.721         0.698                 0.421
LogReg_scaled_k3_4_6                    0.731     0.719            0.738         0.719                 0.461
DNABERT-2 (fine-tuned)                  0.893     0.887            0.901         0.887                 0.786
```
 
---
 
## Logging Evaluation Results
 
`log_eval_results.py` replaces the original `JSON_LOGGER.py` with a more robust version that supports configurable paths, run annotations, and automatic CSV header creation.
 
**Usage:**
 
```bash
python log_eval_results.py \
    --json_path  output/dnabert2/results/DNABERT2_run1/eval_results.json \
    --csv_path   SUMMARY_STATS.csv \
    --run_name   DNABERT2_run1 \
    --extra      "epochs=25,lr=3e-5,data=NAc_QTL"
```
 
If `SUMMARY_STATS.csv` does not exist it is created with a header row. Each subsequent call appends one row, making it suitable for tracking many experiments over time. The `--extra` flag lets you annotate any run with arbitrary key=value metadata (data version, hyperparameters, etc.) that becomes additional columns in the CSV.
 
This script is called automatically at the end of `slurm_finetune.sh`.
 
---
 
## Citation
 
If you use the **motif injection analysis** from this repository, please cite:
 
```bibtex
@article{qureshi2025opioid,
    author  = {Qureshi, Fatir and Apere, Chesna and Okeke, Chidera and
               Kassim, Bibi S. and Iskhakova, Marina and Sallari, Richard and
               Chakraborty, Maharshi and Morgan, Laura and Barnard, Zia and
               Luna, Xochitl and Madden, Megan and Rotti, Pavanna G. and
               Hoang, An and Ramcharan, Hannah K. and Quach, Bryan and
               Willis, Caryn and Maher, Brion S. and Mash, Deborah and
               Scacheri, Peter C. and Johnson, Eric O. and Akbarian, Schahram
               and Corradin, Olivia},
    title   = {Heterogeneous epigenetic variation converges on splicing
               dysregulation in opioid addiction},
    journal = {medRxiv},
    year    = {2025},
    doi     = {10.64898/2025.12.20.25342745},
    url     = {https://www.medrxiv.org/content/10.64898/2025.12.20.25342745v1}
}
```
 
If you use **DNABERT-2** in your work, please also cite:
 
```bibtex
@article{zhou2023dnabert2,
    author  = {Zhou, Zhihan and Ji, Yanrong and Li, Weijian and Dutta, Pratik
               and Davuluri, Ramana and Liu, Han},
    title   = {DNABERT-2: Efficient Foundation Model and Benchmark for
               Multi-Species Genome},
    journal = {ICLR},
    year    = {2024}
}
```
