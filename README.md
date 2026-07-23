# DTFformer

Dual-Granularity Time-Frequency Interaction Transformer for Time Series Forecasting

## Overview

Time series forecasting often struggles to balance time-domain local dynamics and frequency-domain global patterns. **DTFformer** addresses this challenge with a dual-granularity interaction mechanism.

Instead of simple concatenation, the model progressively fuses information from both domains:

- **Micro-level interaction:** time-domain and frequency-domain features provide mutual context during attention computation.
- **Macro-level interaction:** inter-layer gating networks dynamically control information exchange between the two branches.
- **Adaptive noise reduction:** the AmpT-Filter enhances informative frequency components and suppresses noise.

## Model architecture

<p align="center">
  <img src="./DTFformer_architecture.png" alt="DTFformer Architecture" width="900">
</p>

<p align="center">
  <em>Overall architecture of DTFformer.</em>
</p>

## Tested environment

The released implementation was tested with the following environment:

| Component | Version |
| --- | --- |
| Operating system | Windows |
| Python | 3.8.20 |
| PyTorch | 1.12.1+cu116 |
| CUDA | 11.6 |
| cuDNN | 8.3.2 |
| GPU | NVIDIA GeForce RTX 3080 |

## Installation

Clone the repository and create a Python 3.8 environment:

```bash
git clone https://github.com/time-series-Labo/DTFformer.git
cd DTFformer
conda create -n dtfformer python=3.8.20 -y
conda activate dtfformer
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first line of `requirements.txt` points pip to the official PyTorch CUDA 11.6 wheel index. If a different CUDA or CPU-only build is required, install the appropriate PyTorch build first and then install the remaining dependencies.

## Data preparation

Place datasets under the following directory structure:

```text
dataset/
├── ETT/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── weather.csv
├── wind1.csv
└── wind2.csv
```

The ETT datasets are available from the [ETDataset repository](https://github.com/zhouhaoyi/ETDataset). Please prepare the Weather and Wind datasets in the same CSV format expected by `data_provider/data_loader.py`.

## Quick start

Train and evaluate DTFformer on ETTh1 with an input length of 96 and a prediction length of 96:

```bash
python run.py \
  --is_training 1 \
  --model_id ETTh1_96_96 \
  --model DTFformer \
  --data ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --itr 3
```

Available models are `DTFformer`, `DLinear`, `FEDformer`, `FilterTS`, `PatchTST`, `TimeMixer`, `WPMixer`, and `iTransformer`.

The shared experimental protocol uses an input length of 96, prediction lengths in `{96, 192, 336, 720}`, batch size 64, learning rate `5e-5`, and three repeated runs. DTFformer uses two encoder layers, model dimension 512, and eight attention heads; baseline architecture parameters are recorded separately in their model configurations. For each dataset-and-horizon setting, the random number generators are initialized once with seed `2021` before the repetition loop and are not reseeded between repetitions.

## Training and testing DTFformer

The recommended entry point is `run_config.py`, which loads the shared protocol from `configs/common.yaml`, applies the DTFformer settings from `configs/models/DTFformer.yaml`, and then invokes `run.py`.

### Train and evaluate one configuration

The following command trains DTFformer on ETTh1 and evaluates each of the three repeated runs with an input length of 96 and a prediction length of 96:

```bash
python run_config.py \
  --model DTFformer \
  --data ETTh1 \
  --pred_len 96
```

Change `--data` and `--pred_len` to run another dataset or prediction horizon. Supported prediction lengths are `96`, `192`, `336`, and `720`.

### Run all DTFformer main experiments

From the repository root, run:

```bash
bash scripts/long_term_forecast/DTFformer.sh
```

This script runs all seven datasets and all four prediction lengths using three repetitions per setting. Bash, Git Bash, or WSL can be used to execute the script.

### Test an existing checkpoint

To evaluate a previously trained checkpoint without retraining, use the same model, dataset, prediction length, and configuration as the corresponding training run:

```bash
python run_config.py \
  --model DTFformer \
  --data ETTh1 \
  --pred_len 96 \
  --is_training 0 \
  --itr 1
```

The test-only command above loads the checkpoint for run index `0`. The checkpoint must already exist and its experiment setting must match the testing command.

### Checkpoints and evaluation outputs

The best validation checkpoint selected by early stopping is saved as:

```text
checkpoints/<experiment_setting>/checkpoint.pth
```

Testing saves the following files:

```text
results/<experiment_setting>/metrics.npy
results/<experiment_setting>/pred.npy
results/<experiment_setting>/true.npy
```

- `metrics.npy` stores `[MAE, MSE, RMSE, MAPE, MSPE]`.
- `pred.npy` stores the model predictions.
- `true.npy` stores the corresponding ground-truth values.

The experiment setting records the model, dataset, input length, prediction length, architecture parameters, patch length, stride, base seed, and run index.

### Repeated-run reporting

Each main experiment contains three runs. Python, NumPy, and PyTorch are initialized once with the base seed `2021` before the repetition loop, and the random-number-generator states then advance naturally without reseeding between runs.

For the three MSE values and three MAE values:

- the reported mean is the arithmetic mean over the three runs;
- the reported standard deviation is the population standard deviation (`ddof=0`);
- Best MSE is the minimum MSE among the three runs;
- Best MAE is the minimum MAE among the three runs.

## Reproduction scripts

Model-specific scripts for DTFformer and all seven baselines are provided under [`scripts/long_term_forecast`](scripts/long_term_forecast). They cover the seven paper datasets and all four prediction lengths with the shared training protocol documented above.

The authoritative common and model-specific parameters are stored under [`configs/`](configs/). `run_config.py` merges common, model, dataset, and prediction-horizon settings before invoking `run.py`; the reproduction scripts use this resolver automatically.

Run one model or the complete suite with Bash, Git Bash, or WSL:

```bash
bash scripts/long_term_forecast/DTFformer.sh
bash scripts/long_term_forecast/run_all.sh
```

See the [script documentation](scripts/long_term_forecast/README.md) for model-specific settings and commands for running smaller subsets.
