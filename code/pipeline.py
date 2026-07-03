import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import argparse
import logging
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

parser = argparse.ArgumentParser()
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
        return df.null_count().sum_horizontal().item()

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
        x = df[value_col].to_numpy().astype(float)

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
        df = pl.read_parquet(file, columns=cols)  # only load needed cols
        stats = {}
        for col in cols:
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

    def silver_layer(self, cols: list, sampling_rate=-1):
        logging.info("SILVER LAYER: beginning checks...")
        files = list(Path(self.dir_path).glob("*.parquet"))
        # ---- Calculate Summary Statistics

        # get a df of each trial
        # for each df, make concurrent for each signal (should be 7), and pass through the summary stats function
        # this returns a dict each, plug all those dicts in to a new summary stats for that trial
        # combine all the trial dfs into one df, using the trial name as below
        # str(file).split("/")[-1].split(".")[0]

        # all_stats = {}  # {col_name: {stat: value}}

        # total_iters = len(files) * len(cols)
        # pbar = tqdm(total=total_iters)

        func = partial(self.process_file, cols=cols)

        with ProcessPoolExecutor(max_workers=4) as ex:
            results = list(tqdm(ex.map(func, files), total=len(files)))
            print(results)

        # for file in files:
        #     stats = {}
        #     df = pl.read_parquet(file)
        #     for col in cols:
        #         if col not in df.columns:
        #             logging.warning(f"{col} missing in {file.name}, skipping")
        #             pbar.update(1)
        #             continue
        #         # SAMPLING RATES:
        #         # FOR VIBRATION: 51.2 kHz
        #         # FOR POWER: 1kHz
        #         if sampling_rate == -1:
        #             sampling_rate = 1000 if col == "Power" else 51_20
        #         stats = stats | self.compute_row_stats(
        #             df, col, SAMPLE_RATE=sampling_rate
        #         )
        #         pbar.update(1)
        #     trial_name = str(file).split("/")[-1].split(".")[0]
        #     all_stats[trial_name] = stats
        #     pbar.set_description(f"{trial_name}...")
        # print(all_stats)
        # normalise values

    def gold_layer(self):
        return None


pipeline = Pipeline("code/parquet_output")

pipeline.bronze_layer()
pipeline.silver_layer(
    cols=[
        "SpindleAccX",
        "SpindleAccY",
        "SpindleAccZ",
        "PlateLFAccX",
        "PlateLFAccY",
        "PlateLFAccZ",
        "PlateHFAccZ",
        "Power",
    ]
)
