"""
Convert MATLAB v7.3 .mat files to Parquet.

Usage:
    python mat_to_parquet.py --data_dir ../data --out_dir ./parquet_output
    python mat_to_parquet.py --data_dir ../data --out_dir ./parquet_output --file Segmented_Linear_Baseline.mat

Writes one run at a time via PyArrow's ParquetWriter so peak RAM stays
low (~112k rows per flush rather than the entire file).

Output columns:
    routine    -- e.g. Linear, Spindle5000, Machining
    fault_mode -- e.g. Baseline, ToolWear
    run        -- 1-indexed run number within the file
    sample     -- 1-indexed sample index within the run
    <signal>   -- raw signal value for each channel

Power is sampled at 1 kHz; accelerometers at 51.2 kHz. Power values are
repeated to match the accelerometer sample rate (nearest-neighbour).

Requires: pip install pyarrow h5py
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FP_SIGNALS = [
    "SpindleAccX", "SpindleAccY", "SpindleAccZ",
    "PlateLFAccX", "PlateLFAccY", "PlateLFAccZ",
    "PlateHFAccZ",
    "Power",
]

MACH_SIGNAL_MAP = {
    "SpindleAccX": "SpindleX",
    "SpindleAccY": "SpindleY",
    "SpindleAccZ": "SpindleZ",
    "PlateLFAccX": "PlateLFAccX",
    "PlateLFAccY": "PlateLFAccY",
    "PlateLFAccZ": "PlateLFAccZ",
    "Power":       "Power",
}


def _parse_filename(stem: str) -> tuple[str, str]:
    parts = stem.split("_")
    routine = parts[1] if len(parts) > 1 else "Unknown"
    fault   = "_".join(parts[2:]) if len(parts) > 2 else "Unknown"
    return routine, fault


def _make_table(routine: str, fault_mode: str, run_idx: int,
                signals: dict[str, np.ndarray]) -> pa.Table:
    """Build a PyArrow Table for one run (low-memory path)."""
    acc_len   = len(signals["SpindleAccX"])
    power_arr = signals["Power"]
    power_len = len(power_arr)

    # Repeat power values to match acc sample rate (nearest-neighbour)
    indices = np.minimum(
        (np.arange(acc_len) * power_len / acc_len).astype(np.int32),
        power_len - 1,
    )
    power_resampled = power_arr[indices]

    cols: dict[str, pa.Array] = {
        "routine":    pa.array([routine]    * acc_len, type=pa.string()),
        "fault_mode": pa.array([fault_mode] * acc_len, type=pa.string()),
        "run":        pa.array(np.full(acc_len, run_idx + 1, dtype=np.int32)),
        "sample":     pa.array(np.arange(1, acc_len + 1, dtype=np.int32)),
    }
    for name, arr in signals.items():
        if name == "Power":
            cols["Power"] = pa.array(power_resampled.astype(np.float32))
        else:
            cols[name] = pa.array(arr.astype(np.float32))

    return pa.table(cols)


def convert_file(mat_path: Path, out_path: Path) -> int:
    stem = mat_path.stem
    routine, fault_mode = _parse_filename(stem)
    is_machining = routine == "Machining"
    signal_map = MACH_SIGNAL_MAP if is_machining else {s: s for s in FP_SIGNALS}
    canonical_names = list(signal_map.keys())

    total_rows = 0
    writer = None

    with h5py.File(mat_path, "r") as f:
        grp_key = next(k for k in f.keys() if not k.startswith("#"))
        grp = f[grp_key]
        n_runs = grp[signal_map[canonical_names[0]]].shape[0]

        for run_idx in range(n_runs):
            signals: dict[str, np.ndarray] = {}
            for canonical, file_key in signal_map.items():
                ref = grp[file_key][run_idx, 0]
                signals[canonical] = np.array(f[ref]).flatten()

            table = _make_table(routine, fault_mode, run_idx, signals)

            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += len(table)

            if (run_idx + 1) % 50 == 0:
                print(f"  {stem}: {run_idx + 1}/{n_runs} runs ({total_rows:,} rows written)")

    if writer:
        writer.close()

    return total_rows


def main():
    parser = argparse.ArgumentParser(description="Convert .mat files to Parquet (raw signals, streaming)")
    parser.add_argument("--data_dir", default="../data", help="Folder containing .mat files")
    parser.add_argument("--out_dir",  default="./parquet_output", help="Output folder")
    parser.add_argument("--file",     default=None, help="Convert a single file only (filename, not path)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        mat_files = [data_dir / args.file]
    else:
        mat_files = sorted(data_dir.glob("Segmented_*.mat"))

    print(f"Converting {len(mat_files)} file(s) ...")

    for mat_path in mat_files:
        out_path = out_dir / f"{mat_path.stem}.parquet"
        print(f"Processing {mat_path.name} ...")
        total = convert_file(mat_path, out_path)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  -> {out_path}  ({total:,} rows, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
