# Long-term forecasting scripts

These scripts reproduce the main comparison protocol for DTFformer and the seven baseline models reported in the paper.

## Models

- DTFformer
- DLinear
- FEDformer
- FilterTS
- PatchTST
- TimeMixer
- WPMixer
- iTransformer

## Shared experimental protocol

To make the comparison transparent, `_run_model.sh` resolves the common protocol and model-specific settings from the YAML files under `configs/`:

| Setting | Value |
| --- | --- |
| Input length | 96 |
| Prediction lengths | 96, 192, 336, 720 |
| Datasets | ETTh1, ETTh2, ETTm1, ETTm2, Weather, Wind1, Wind2 |
| Repeated runs | 3 |
| Random seed | Initialized once with 2021 before the repetitions |
| Batch size | 64 for ETT; 32 for Weather and Wind |
| Learning rate | 5e-5 |
| Maximum epochs | 10 |
| Early-stopping patience | 3 |
| Loss | MSE |

Architecture parameters are model-specific rather than part of the shared training budget. DTFformer uses `d_model=512`, `n_heads=8`, `e_layers=2`, `d_ff=2048`, `dropout=0.1`, and patch length/stride `16/8`. FilterTS additionally uses the filtering values published in its official scripts: `quantile=0.9`, `bandwidth=1`, `top_K_static_freqs=10`, and `filter_type=all`. TimeMixer explicitly records its average-pooling down-sampling configuration. Dataset- and horizon-specific values are documented in each model YAML.

For each dataset-and-horizon setting, the script starts one `run.py` process. Python, NumPy, and PyTorch are seeded once with `2021` before that process enters its repetition loop, matching the repetition-loop behavior of the original local runners. The generators are not reseeded between the three repetitions; their states advance naturally as the runs execute.

## Usage

Run one model from the repository root with Bash, Git Bash, or WSL:

```bash
bash scripts/long_term_forecast/DTFformer.sh
bash scripts/long_term_forecast/PatchTST.sh
```

Run the full comparison suite:

```bash
bash scripts/long_term_forecast/run_all.sh
```

The full suite contains 224 experiment settings before accounting for the three repeated runs, so it can take a long time. Environment variables can be used for a smaller reproducibility check:

```bash
DTF_DATASETS="ETTh1 weather" \
DTF_PRED_LENS="96 192" \
DTF_ITR=1 \
bash scripts/long_term_forecast/PatchTST.sh
```

Use a specific Python executable when needed:

```bash
PYTHON_BIN="/path/to/python" bash scripts/long_term_forecast/DTFformer.sh
```

Dataset paths and dimensions are selected from `configs/common.yaml` by `run_config.py`. Prepare the files according to the data layout in the root README before running these scripts.

Preview the fully resolved command without starting training:

```bash
python run_config.py --model PatchTST --data ETTh1 --pred_len 96 --dry_run
```

## Upstream script references

The baseline command structure and model-specific options were checked against:

- [Time-Series-Library long-term forecasting scripts](https://github.com/thuml/Time-Series-Library/tree/main/scripts/long_term_forecast)
- [FilterTS official scripts](https://github.com/wyl010607/FilterTS/tree/main/scripts)

The files in this repository are adapted to the common protocol reported for DTFformer; they are not verbatim copies of upstream scripts.
