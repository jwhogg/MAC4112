# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["POLARS_MAX_THREADS"] = "1"
import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import polars as pl
from scipy.signal import stft
from scipy.stats import kurtosis, skew
from tqdm import tqdm

"""
The Class for the Medalian Pipeline

Expects a str with the path to the folder containing the files to process

Expects that the files are .parquet
"""


class Pipeline:
    def __init__(self, dir_path: str):
        self.dir_path = dir_path

    def bronze_check_missing_data(self, file_path: str):
        df = pl.read_parquet(file_path)
        null_count = df.null_count().sum_horizontal().item()
        nan_count = (
            df.select(pl.col(pl.Float32, pl.Float64).is_nan().sum())
            .sum_horizontal()
            .item()
        )
        return null_count + nan_count

    def compute_spectral_kurtosis(
        self, x: np.ndarray, fs: float, nperseg: int
    ) -> np.ndarray:
        """Returns K(f): one spectral kurtosis value per frequency bin,
        averaged over time per Antoni's definition."""
        _, _, S = stft(x, fs=fs, nperseg=nperseg)
        mag = np.abs(S)  # shape: (freq, time)
        mag2 = mag**2
        mag4 = mag**4
        mean_mag2 = mag2.mean(axis=1)
        mean_mag4 = mag4.mean(axis=1)
        K_f = mean_mag4 / (mean_mag2**2) - 2
        return K_f

    def compute_row_stats(self, df: pl.DataFrame, value_col: str, SAMPLE_RATE) -> dict:
        NPERSEG = int(len(df) / 25)
        x = df[value_col].to_numpy().astype(np.float64)

        if len(x) < 25 or np.all(x == 0) or np.std(x) == 0:
            logging.warning(
                f"{value_col}: degenerate signal, len={len(x)}, std={np.std(x)}"
            )

        mean_ = x.mean()
        std_ = x.std()
        rms = np.sqrt(np.mean(x**2))
        kurt = kurtosis(x, fisher=True, bias=False)
        skw = skew(x, bias=False)
        p2p = x.max() - x.min()
        abs_x = np.abs(x)
        crest = abs_x.max() / rms
        shape_factor = rms / abs_x.mean()
        impulse_factor = abs_x.max() / abs_x.mean()
        margin_factor = abs_x.max() / (np.mean(np.sqrt(abs_x)) ** 2)
        energy = np.sum(x**2)

        K_f = self.compute_spectral_kurtosis(x, SAMPLE_RATE, NPERSEG)
        sk_mean = K_f.mean()
        sk_std = K_f.std()
        sk_skew = skew(K_f, bias=False)
        sk_kurt = kurtosis(K_f, fisher=True, bias=False)

        stats = {
            "mean": mean_,
            "std": std_,
            "rms": rms,
            "kurtosis": kurt,
            "skewness": skw,
            "peak_to_peak": p2p,
            "crest_factor": crest,
            "shape_factor": shape_factor,
            "impulse_factor": impulse_factor,
            "margin_factor": margin_factor,
            "energy": energy,
            "spectral_kurtosis_mean": sk_mean,
            "spectral_kurtosis_std": sk_std,
            "spectral_kurtosis_skewness": sk_skew,
            "spectral_kurtosis_kurtosis": sk_kurt,
        }
        return {f"{value_col}_{k}": v for k, v in stats.items()}

    def process_file(self, file, cols):
        logging.info(f"about to read file {file}")
        df = pl.read_parquet(file)  # only load needed cols
        stats = {}
        for col in cols:
            logging.info(f"doing col {col} for file {file}")
            if col not in df.columns:
                continue
            rate = 1000 if col == "Power" else 51_20
            stats |= self.compute_row_stats(df, col, SAMPLE_RATE=rate)
        logging.info(f"done {file}")
        return str(file).split("/")[-1].split(".")[0], stats

    def bronze_layer(self):
        logging.info("BRONZE LAYER: beginning checks...")

        files = list(Path(self.dir_path).glob("*.parquet"))

        # ---- Check data is not empty
        check_missing_results = [self.bronze_check_missing_data(str(f)) for f in files]

        if sum(check_missing_results) > 0:
            print("WARNING: there is missing data!")
            # TODO: if missing data, specify where it is coming from and fix it via interpolation

    def silver_layer(
        self,
        cols: list,
        sampling_rate=-1,
        run_override=False,
        path="code/silver_data/silver_layer.parquet",
    ):
        if (Path(path).exists()) and (not run_override):
            logging.info("Silver data already generated, skipping...")
            return None
        logging.info("SILVER LAYER: beginning checks...")
        files = list(Path(self.dir_path).glob("*.parquet"))
        # ---- Calculate Summary Statistics

        results = []
        for file in files:
            results.append(self.process_file(file, cols))

        summary_df = pl.DataFrame(
            [{"trial": trial, **stats} for trial, stats in results]
        )
        summary_df.write_parquet(path)
        logging.info("SILVER LAYER: Done! Saved File")

    def gold_layer(self, silver_path="code/silver_data/silver_layer.parquet"):
        if Path(silver_path).exists():
            logging.info("Silver layer data not found, running Silver layer...")
            self.silver_layer(
                cols=[
                    "SpindleAccX",
                    "SpindleAccY",
                    "SpindleAccZ",
                    "PlateLFAccX",
                    "PlateLFAccY",
                    "PlateLFAccZ",
                    "PlateHFAccZ",
                    "Power",
                ],
            )

        df = pl.read_parquet(silver_path)

        return None


# ---- command line running
# example usage:
# > python pipeline.py
# > python pipeline.py --path code/other_output --run-override
# > python pipeline.py --cols SpindleAccX Power --skip-bronze
def main():
    parser = argparse.ArgumentParser(description="Run Pipeline")
    parser.add_argument(
        "--data_path", default="code/parquet_output", help="Pipeline output path"
    )
    parser.add_argument(
        "--cols",
        nargs="+",
        default=[
            "SpindleAccX",
            "SpindleAccY",
            "SpindleAccZ",
            "PlateLFAccX",
            "PlateLFAccY",
            "PlateLFAccZ",
            "PlateHFAccZ",
            "Power",
        ],
        help="Columns to process in silver layer",
    )
    parser.add_argument(
        "--silver-data-path",
        default="code/silver_data/silver_layer.parquet",
        help="Where the silver layer saves its output data file to",
    )
    parser.add_argument(
        "--silver-run-override",
        action="store_true",
        help="Force re-run even if output exists",
    )
    parser.add_argument(
        "--skip-bronze-checks",
        action="store_true",
        help="Skip the bronze layer step",
    )
    parser.add_argument("--verbose", action="store_true", help="show logs in terminal")

    args = parser.parse_args()

    handlers = [logging.FileHandler("pipeline.log")]
    if args.verbose:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )

    pipeline = Pipeline(args.data_path)

    if not args.skip_bronze_checks:
        pipeline.bronze_layer()

    pipeline.silver_layer(
        cols=args.cols,
        run_override=args.silver_run_override,
        path=args.silver_data_path,
    )


if __name__ == "__main__":
    main()
