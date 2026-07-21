# Experiment configurations

This directory contains the shared paper protocol and one model-specific configuration for DTFformer and each baseline.

## Files

- `common.yaml`: datasets, forecasting protocol, training budget, repeated-run seed behavior, device, and environment metadata.
- `models/DTFformer.yaml`
- `models/DLinear.yaml`
- `models/FEDformer.yaml`
- `models/FilterTS.yaml`
- `models/PatchTST.yaml`
- `models/TimeMixer.yaml`
- `models/WPMixer.yaml`
- `models/iTransformer.yaml`

## Resolution order

`run_config.py` resolves an experiment in the following order, with later values overriding earlier ones:

1. argument sections in `common.yaml`;
2. dataset metadata in `common.yaml`;
3. `default_args` in the selected model configuration;
4. `dataset_args.<dataset>.default_args`;
5. `dataset_args.<dataset>.horizon_args.<pred_len>`;
6. explicit command-line overrides.

Preview a resolved command without starting training:

```bash
python run_config.py --model PatchTST --data ETTh1 --pred_len 96 --dry_run
```

Run the same setting:

```bash
python run_config.py --model PatchTST --data ETTh1 --pred_len 96
```

The Bash scripts under `scripts/long_term_forecast/` use this resolver automatically.

## Configuration policy

- The paper protocol uses input length 96, prediction lengths 96, 192, 336, and 720, Adam with learning rate 5e-5, batch size 64, at most 10 epochs, patience 3, and three repeated runs.
- Public-dataset architecture parameters use official scripts where compatible and available.
- Missing public-dataset settings fall back to recovered local settings or current implementation defaults, recorded in each file's `provenance` section.
- Wind1 and Wind2 are private datasets. Applicable baseline parameters follow the DTFformer protocol, while model-specific operations retain their baseline defaults.
- Official WPMixer scripts use `seq_len=512`; the active paper configuration instead uses `seq_len=96` and recovered local/current defaults.
- TimeMixer inherits `label_len=48` from the paper protocol instead of the `label_len=0` used by current upstream scripts.

## Sources

- [Time-Series-Library long-term forecasting scripts](https://github.com/thuml/Time-Series-Library/tree/main/scripts/long_term_forecast)
- [FilterTS official scripts](https://github.com/wyl010607/FilterTS/tree/main/scripts)
