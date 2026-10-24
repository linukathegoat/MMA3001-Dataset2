"""Interpolation benchmarking on a building environmental temperature sensor.

Data source: Monash MMA3001 Dataset 2 environmental sensor CSV (inside ZIP).
Only the sensor's measurement Unix timestamp (in JSON), not the CSV createdate,
is used to determine when a reading was taken.

Engineering scope: impute short, internal temperature gaps at one location.
Do not use for safety-critical environmental control or extrapolation.
"""

from __future__ import annotations

import csv
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

SENSOR_ID = "6012002000869"
SENSOR_ROOM = "Monash Building 60, G.38, Compactus_upper (label EMS107)"
MONTH_START_UTC = "2024-05-01"
MONTH_END_EXCLUSIVE_UTC = "2024-06-01"
METHODS = ("last_known", "linear", "pchip", "cubic_spline")


def load_temperature(zip_path: str | Path, sensor_id: str = SENSOR_ID,
                     date_start: str = MONTH_START_UTC,
                     date_end: str = MONTH_END_EXCLUSIVE_UTC) -> tuple[pd.DataFrame, dict]:
    """Extract and audit one sensor's real readings from the supplied ZIP.

    Parameters
    ----------
    zip_path: path to original 5EnvSensor CSV ZIP (not the occupancy dataset).
    sensor_id: sensor identifier matching the location spreadsheet (string).
    date_start, date_end: UTC ISO date bounds, half-open [start, end).

    Returns
    -------
    (dataframe, audit): UTC timestamp and temperature_c, sorted and deduplicated;
    diagnostic counts including repeated messages and discarded values.

    Raises
    ------
    ValueError for empty/invalid study ranges; FileNotFoundError if ZIP missing.
    """
    path = Path(zip_path)
    if not path.exists():
        raise FileNotFoundError(f"Environmental data ZIP not found: {path}")
    start_s = int(pd.Timestamp(date_start, tz="UTC").timestamp())
    end_s = int(pd.Timestamp(date_end, tz="UTC").timestamp())
    if start_s >= end_s:
        raise ValueError("Date start must precede date end")

    records = []
    audit = {"sensor_id": sensor_id, "study_start_utc": date_start,
             "study_end_exclusive_utc": date_end, "sensor_rows": 0,
             "json_errors": 0, "temperature_values_in_range": 0,
             "invalid_temperature_values": 0, "conflicting_duplicates": 0}
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("ZIP contains no CSV")
        with archive.open(csv_names[0]) as input_file:
            reader = csv.DictReader(io.TextIOWrapper(input_file, encoding="utf-8-sig", newline=""))
            required = {"sensorid", "jsondata"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("Expected CSV columns sensorid and jsondata")
            for row in reader:
                if str(row["sensorid"]) != sensor_id:
                    continue
                audit["sensor_rows"] += 1
                try:
                    payload = json.loads(row["jsondata"])
                except (ValueError, TypeError):
                    audit["json_errors"] += 1
                    continue
                for item in payload:
                    variable = item.get("variable", {})
                    if variable.get("name") != "Temperature":
                        continue
                    if variable.get("unit") != "°C":
                        raise ValueError("Unexpected temperature unit: " + str(variable.get("unit")))
                    for point in item.get("values", []):
                        try:
                            ts = int(point["time"])
                            value = float(point["value"])
                        except (KeyError, TypeError, ValueError, OverflowError):
                            audit["invalid_temperature_values"] += 1
                            continue
                        if not (start_s <= ts < end_s):
                            continue
                        audit["temperature_values_in_range"] += 1
                        if not np.isfinite(value) or not (-30.0 <= value <= 65.0):
                            audit["invalid_temperature_values"] += 1
                            continue
                        records.append((ts, value))
    if not records:
        raise ValueError("No valid temperature readings in chosen date range")
    # Resolve duplicate messages: one measurement can have several database records.
    df = pd.DataFrame(records, columns=["unix_seconds", "temperature_c"])
    conflicts = df.groupby("unix_seconds")["temperature_c"].nunique()
    audit["conflicting_duplicates"] = int((conflicts > 1).sum())
    # If conflicting values exist at the same instant, use the median and flag it.
    df = df.groupby("unix_seconds", as_index=False)["temperature_c"].median()
    df = df.sort_values("unix_seconds").reset_index(drop=True)
    df["timestamp_utc"] = pd.to_datetime(df["unix_seconds"], unit="s", utc=True)
    df = df[["timestamp_utc", "unix_seconds", "temperature_c"]]
    audit["unique_measurements"] = int(len(df))
    audit["repeated_messages_removed"] = (audit["temperature_values_in_range"]
                                          -audit["invalid_temperature_values"]-len(df))
    gaps_min = np.diff(df["unix_seconds"].to_numpy(dtype=float)) / 60
    audit["median_observed_interval_minutes"] = float(np.median(gaps_min)) if len(gaps_min) else None
    audit["observed_intervals_over_30_minutes"] = int(np.sum(gaps_min > 30))
    audit["minimum_temperature_c"] = float(df["temperature_c"].min())
    audit["maximum_temperature_c"] = float(df["temperature_c"].max())
    audit["actual_first_timestamp_utc"] = df["timestamp_utc"].iloc[0].isoformat()
    audit["actual_last_timestamp_utc"] = df["timestamp_utc"].iloc[-1].isoformat()
    return df, audit


def predict(method: str, observed_times: np.ndarray,
            observed_values: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Predict values only INSIDE observed time domain (no extrapolation).

    ``last_known`` uses previous observed value, ``linear`` uses straight lines,
    ``pchip`` is shape-preserving cubic Hermite, and ``cubic_spline`` is natural
    cubic spline. All timestamps are seconds; temperature values are °C.
    """
    x = np.asarray(observed_times, dtype=float)
    y = np.asarray(observed_values, dtype=float)
    q = np.asarray(query_times, dtype=float)
    if len(x) < 3 or x.shape != y.shape or np.any(np.diff(x) <= 0):
        raise ValueError("Expected >=3 sorted, unique observed times and equal-length values")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(q))):
        raise ValueError("Inputs must be finite")
    if np.any(q < x[0]) or np.any(q > x[-1]):
        raise ValueError("Query outside observed time range: extrapolation disallowed")
    # Relative coordinate improves conditioning when Unix times are ~1.7e9.
    x_rel, q_rel = x - x[0], q - x[0]
    if method == "last_known":
        indices = np.searchsorted(x_rel, q_rel, side="right") - 1
        return y[indices]
    if method == "linear":
        return np.interp(q_rel, x_rel, y)
    if method == "pchip":
        return PchipInterpolator(x_rel, y)(q_rel)
    if method == "cubic_spline":
        return CubicSpline(x_rel, y, bc_type="natural")(q_rel)
    raise ValueError("Unsupported interpolation method: " + method)


def select_masked_blocks(times: np.ndarray, n_per_block: int,
                         n_blocks: int = 15, seed: int = 2026,
                         max_observed_step_minutes: float = 30.0) -> list[np.ndarray]:
    """Select disjoint, internally continuous test blocks with observed neighbours.

    ``n_per_block`` consecutive readings are hidden for each block. Candidates
    are accepted only when all surrounding actual sampling steps are <=30 min.
    This rejects gaps already present in the measured data.
    """
    times = np.asarray(times, dtype=float)
    if n_per_block < 1 or n_blocks < 1 or len(times) < n_per_block + 4:
        raise ValueError("Insufficient observations or invalid masking parameters")
    rng = np.random.default_rng(seed)
    intervals = np.diff(times) / 60
    valid_starts = []
    for start in range(2, len(times) - n_per_block - 2):
        # Covers the original samples before, within and after the masked block.
        steps = intervals[start - 1:start + n_per_block]
        if len(steps) == n_per_block + 1 and np.all(steps <= max_observed_step_minutes):
            valid_starts.append(start)
    shuffled = rng.permutation(valid_starts)
    used = np.zeros(len(times), dtype=bool)
    blocks = []
    for start in shuffled:
        positions = np.arange(start, start + n_per_block)
        surround = slice(start - 2, start + n_per_block + 2)
        if np.any(used[surround]):
            continue
        used[surround] = True
        blocks.append(positions)
        if len(blocks) == n_blocks:
            break
    if len(blocks) < n_blocks:
        raise ValueError(f"Could only sample {len(blocks)} non-overlapping gap blocks")
    return sorted(blocks, key=lambda a: a[0])


def evaluate(series: pd.DataFrame, gap_lengths: tuple[int,...] = (3, 6, 12),
             n_blocks: int = 15, seed: int = 2026,
             split_name: str = "validation") -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    """Mask genuine observed measurements; compare predictions to hidden truth.

    Returns per-method summary, individual predictions (for plots), and
    example containing one gap plus method predictions. No observed target in
    any gap is included in the interpolant's training values.
    """
    x = series["unix_seconds"].to_numpy(dtype=float)
    y = series["temperature_c"].to_numpy(dtype=float)
    summary, predicted_rows = [], []
    example = {}
    for gap_length in gap_lengths:
        blocks = select_masked_blocks(x, gap_length, n_blocks,
                                      seed=seed + gap_length)
        hidden = np.sort(np.concatenate(blocks))
        train_mask = np.ones(len(x), dtype=bool)
        train_mask[hidden] = False
        known_x, known_y = x[train_mask], y[train_mask]
        test_x, truth = x[hidden], y[hidden]
        # Seconds between surrounding known endpoints for actual masked gaps.
        gap_widths = []
        for block in blocks:
            gap_widths.append((x[block[-1] + 1] - x[block[0] - 1]) / 60)
        for method in METHODS:
            # Run 5 times to reduce single-call timing noise; timing excludes I/O.
            runtime_ms = []
            for _ in range(5):
                st = time.perf_counter()
                pred = predict(method, known_x, known_y, test_x)
                runtime_ms.append((time.perf_counter() - st) * 1000)
            err = pred - truth
            summary.append({"split": split_name, "gap_readings": gap_length,
                            "method": method, "n_gap_blocks":len(blocks),
                            "n_hidden_readings":int(len(hidden)),
                            "mean_gap_duration_minutes":float(np.mean(gap_widths)),
                            "MAE_C":float(np.mean(abs(err))),
                            "RMSE_C":float(np.sqrt(np.mean(err**2))),
                            "MaxAE_C":float(np.max(abs(err))),
                            "within_0_5_C_pct":float(np.mean(abs(err) <= 0.5)*100),
                            "median_prediction_runtime_ms":float(np.median(runtime_ms))})
            predicted_rows.extend({"split":split_name, "gap_readings": gap_length,
                                   "method":method, "unix_seconds":float(t),
                                   "true_temperature_c":float(actual),
                                   "predicted_temperature_c":float(estimate),
                                   "error_c":float(estimate-actual)}
                                  for t,actual,estimate in zip(test_x,truth,pred))
        if gap_length == 6:
            block = blocks[len(blocks)//2]
            lo,hi = max(0,block[0]-8), min(len(x),block[-1]+9)
            example={"times":x[lo:hi],"actual":y[lo:hi],
                     "hidden_times":x[block], "hidden_values":y[block],
                     "predictions": {method:predict(method,known_x,known_y,x[block])
                                     for method in METHODS}}
    return pd.DataFrame(summary),pd.DataFrame(predicted_rows),example


def choose_by_validation_mae(validation_summary: pd.DataFrame) -> str:
    """Choose before inspecting holdout results; equal weight per gap length."""
    m = validation_summary.groupby("method")["MAE_C"].mean().sort_values()
    return str(m.index[0])
