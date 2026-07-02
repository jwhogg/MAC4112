import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import polars as pl

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

    def bronze_layer(self):
        logging.info("BRONZE LAYER: beginning checks...")

        files = list(Path(self.dir_path).glob("*.parquet"))

        # ---- Check data is not empty
        check_missing_results = [self.bronze_check_missing_data(str(f)) for f in files]

        if sum(check_missing_results) > 0:
            print("WARNING: there is missing data!")
            # TODO: if missing data, specify where it is coming from and fix it via interpolation

    def silver_layer(self):
        # ---- Calculate Summary Statistics

        # normalise values
        return None

    def gold_layer(self):
        return None


pipeline = Pipeline("code/parquet_output")

pipeline.bronze_layer()
