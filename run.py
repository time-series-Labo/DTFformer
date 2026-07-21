import argparse
import os
import random

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_dataset_defaults(args):
    dataset = args.data

    if dataset in {"ETTh1", "ETTh2"}:
        args.root_path = args.root_path or "dataset/ETT/"
        args.data_path = args.data_path or f"{dataset}.csv"
        args.enc_in = args.dec_in = args.c_out = 7
        args.freq = args.freq or "h"
    elif dataset in {"ETTm1", "ETTm2"}:
        args.root_path = args.root_path or "dataset/ETT/"
        args.data_path = args.data_path or f"{dataset}.csv"
        args.enc_in = args.dec_in = args.c_out = 7
        args.freq = args.freq or "t"
    elif dataset == "weather":
        args.root_path = args.root_path or "dataset/"
        args.data_path = args.data_path or "weather.csv"
        args.enc_in = args.dec_in = args.c_out = 21
        args.freq = args.freq or "t"
    elif dataset == "exchange_rate":
        args.root_path = args.root_path or "dataset/"
        args.data_path = args.data_path or "exchange_rate.csv"
        args.enc_in = args.dec_in = args.c_out = 8
        args.freq = args.freq or "d"
    elif dataset == "electricity":
        args.root_path = args.root_path or "dataset/"
        args.data_path = args.data_path or "electricity.csv"
        args.enc_in = args.dec_in = args.c_out = 321
        args.freq = args.freq or "h"
    elif dataset in {"wind1", "wind2"}:
        args.root_path = args.root_path or "dataset/"
        args.data_path = args.data_path or f"{dataset}.csv"
        args.enc_in = args.dec_in = args.c_out = 26
        args.freq = args.freq or "t"
        args.target = "C_VALUE"
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. "
            "Use one of: ETTh1, ETTh2, ETTm1, ETTm2, weather, exchange_rate, electricity, wind1, wind2."
        )

    return args


def build_setting(args, run_index: int = 0) -> str:
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}"
        f"_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
        f"_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}"
        f"_df{args.d_ff}_lr{args.learning_rate}_do{args.dropout}"
        f"_pln{args.patch_len}_st{args.stride}"
        f"_expand{args.expand}_dc{args.d_conv}_fc{args.factor}"
        f"_eb{args.embed}_dt{args.distil}_seed{args.seed}_{args.des}_{run_index}"
    )


def clear_device_cache(args) -> None:
    if args.use_gpu and args.gpu_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif args.use_gpu and args.gpu_type == "mps" and hasattr(torch.backends, "mps"):
        torch.backends.mps.empty_cache()


def get_args():
    parser = argparse.ArgumentParser(description="Single long-term forecasting run")

    # basic config
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, default=1, help="1: train+test, 0: test only")
    parser.add_argument("--model_id", type=str, default="train")
    parser.add_argument(
        "--model",
        type=str,
        default="DTFformer",
        choices=[
            "DTFformer",
            "DLinear",
            "FEDformer",
            "FilterTS",
            "PatchTST",
            "TimeMixer",
            "WPMixer",
            "iTransformer",
        ],
    )
    parser.add_argument("--des", type=str, default="test")

    # data
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default=None)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")
    parser.add_argument("--inverse", action="store_true", default=False)

    # model
    parser.add_argument("--enc_in", type=int, default=None)
    parser.add_argument("--dec_in", type=int, default=None)
    parser.add_argument("--c_out", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--embed", type=str, default="learned")
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--wavelet", type=str, default="db2")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--tfactor", type=int, default=5)
    parser.add_argument("--dfactor", type=int, default=5)
    parser.add_argument("--no_decomposition", action="store_true", default=False)

    # shared options used by other available models
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--distil", action="store_false", default=True)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--version", type=str, default="fourier")
    parser.add_argument("--mode_select", type=str, default="random")
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--channel_independence", type=int, default=1)
    parser.add_argument("--decomp_method", type=str, default="moving_avg")
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--down_sampling_layers", type=int, default=2)
    parser.add_argument("--down_sampling_window", type=int, default=2)
    parser.add_argument("--down_sampling_method", type=str, default="avg")
    parser.add_argument("--seg_len", type=int, default=96)
    parser.add_argument("--individual", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--pos", type=int, choices=[0, 1], default=1)

    # FilterTS options
    parser.add_argument("--filter_type", type=str, default="all")
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--bandwidth", type=float, default=0.1)
    parser.add_argument("--embedding", type=str, default="fourier_interpolate")
    parser.add_argument("--top_K_static_freqs", type=int, default=50)

    # optimization
    parser.add_argument("--itr", type=int, default=3)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--use_dtw", action="store_true", default=False)

    # GPU
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--no_use_gpu", action="store_false", dest="use_gpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_type", type=str, default="cuda", choices=["cuda", "mps"])
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    # compatibility options expected by some modules
    parser.add_argument("--p_hidden_dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--p_hidden_layers", type=int, default=2)
    parser.add_argument("--mask_rate", type=float, default=0.25)
    parser.add_argument("--anomaly_ratio", type=float, default=0.25)
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--extra_tag", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # Match the original local experiment runners: initialize the random
    # generators once, then let their state advance across repetitions.
    set_seed(args.seed)
    args = apply_dataset_defaults(args)

    if args.gpu_type == "cuda" and not torch.cuda.is_available():
        args.use_gpu = False
        print("CUDA is not available. Falling back to CPU.")
    elif args.gpu_type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        args.use_gpu = False
        print("MPS is not available. Falling back to CPU.")

    if args.use_gpu and args.gpu_type == "cuda":
        args.device = torch.device(f"cuda:{args.gpu}")
    elif args.use_gpu and args.gpu_type == "mps":
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(device_id) for device_id in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    os.makedirs(args.checkpoints, exist_ok=True)

    print("Args in experiment:")
    print_args(args)

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        mses, maes = [], []
        for ii in range(args.itr):
            exp = Exp(args)
            setting = build_setting(args, ii)

            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)

            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            mse, mae = exp.test(setting)
            mses.append(mse)
            maes.append(mae)
            clear_device_cache(args)

        if mses:
            avg_mse, std_mse = float(np.mean(mses)), float(np.std(mses))
            avg_mae, std_mae = float(np.mean(maes)), float(np.std(maes))
            print("\n" + "=" * 60)
            print(f"Summary: {args.model} | {args.data} | pred_len={args.pred_len}")
            print(f"MSE: {avg_mse:.4f} +/- {std_mse:.4f}")
            print(f"MAE: {avg_mae:.4f} +/- {std_mae:.4f}")
            print(f"Best MSE: {min(mses):.4f}")
            print(f"Best MAE: {min(maes):.4f}")
            print("=" * 60)

            with open("result_long_term_forecast_sensitive.txt", "a", encoding="utf-8") as f:
                f.write(build_setting(args, args.itr - 1) + "\n")
                f.write(f"MSE: {avg_mse:.4f} +/- {std_mse:.4f}, MAE: {avg_mae:.4f} +/- {std_mae:.4f}\n\n")
    else:
        exp = Exp(args)
        setting = build_setting(args, 0)
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test=1)
        clear_device_cache(args)
