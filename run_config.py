import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = REPO_ROOT / "configs"
COMMON_ARG_SECTIONS = (
    "forecast_args",
    "training_args",
    "reproducibility_args",
    "evaluation_args",
    "device_args",
    "output_args",
)


def read_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return data


def merge_args(target: Dict, values: Optional[Dict]) -> None:
    if values:
        target.update(values)


def normalize_dataset(common_config: Dict, dataset: str) -> str:
    datasets = common_config["datasets"]
    if dataset in datasets:
        return dataset
    aliases = {name.casefold(): name for name in datasets}
    normalized = aliases.get(dataset.casefold())
    if normalized is None:
        raise ValueError(f"Unsupported dataset {dataset!r}")
    return normalized


def resolve_args(model: str, dataset: str, pred_len: int) -> Dict:
    model_path = CONFIG_ROOT / "models" / f"{model}.yaml"
    if not model_path.exists():
        raise FileNotFoundError(f"Model configuration not found: {model_path}")

    model_config = read_yaml(model_path)
    base_path = (model_path.parent / model_config.get("base_config", "../common.yaml")).resolve()
    common_config = read_yaml(base_path)

    if pred_len not in common_config["prediction_lengths"]:
        raise ValueError(
            f"Unsupported prediction length {pred_len}; "
            f"choose from {common_config['prediction_lengths']}"
        )
    dataset = normalize_dataset(common_config, dataset)

    resolved = {}
    for section in COMMON_ARG_SECTIONS:
        merge_args(resolved, common_config.get(section))
    merge_args(resolved, common_config["datasets"][dataset])
    merge_args(resolved, model_config.get("default_args"))

    dataset_config = model_config.get("dataset_args", {}).get(dataset, {})
    merge_args(resolved, dataset_config.get("default_args"))
    horizon_args = dataset_config.get("horizon_args", {})
    merge_args(resolved, horizon_args.get(pred_len) or horizon_args.get(str(pred_len)))

    resolved.update(
        model=model,
        model_id=f"{dataset}_{resolved['seq_len']}_{pred_len}",
        pred_len=pred_len,
    )
    return resolved


def append_argument(command: List[str], key: str, value) -> None:
    if value is None:
        return
    if key == "use_gpu":
        if not value:
            command.append("--no_use_gpu")
        return
    if isinstance(value, bool):
        if value:
            command.append(f"--{key}")
        return
    if isinstance(value, list):
        command.append(f"--{key}")
        command.extend(str(item) for item in value)
        return
    command.extend((f"--{key}", str(value)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one paper configuration")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--pred_len", required=True, type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--itr", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry_run", action="store_true")
    args, extra_args = parser.parse_known_args()

    resolved = resolve_args(args.model, args.data, args.pred_len)
    for key in ("seq_len", "itr", "seed"):
        value = getattr(args, key)
        if value is not None:
            resolved[key] = value
    resolved["model_id"] = f"{resolved['data']}_{resolved['seq_len']}_{args.pred_len}"

    command = [sys.executable, "-u", str(REPO_ROOT / "run.py")]
    for key, value in resolved.items():
        append_argument(command, key, value)
    command.extend(extra_args)

    print("Resolved command:")
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
