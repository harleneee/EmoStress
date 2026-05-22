import argparse
import copy
import json
import random
import re
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import welch
from scipy.stats import skew, kurtosis

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

warnings.filterwarnings("ignore")


# ============================================================
# EmoStress ECSMP E4 HR + IBI Training
# Models: Random Forest / ML candidates + 1D-CNN + BiLSTM
# Task: Six-emotion recognition
# Labels: anger, disgust, fear, happy, neutral, sad
# ============================================================

RANDOM_STATE = 42

DATA_EMOTION_ORDER = ["neutral", "fear", "sad", "happy", "anger", "disgust"]
REPORT_LABELS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]

EMOTION_TO_STRESS = {
    "happy": "Low stress",
    "neutral": "Low / Baseline stress",
    "sad": "Medium stress",
    "fear": "High stress",
    "anger": "High stress",
    "disgust": "Medium to High stress",
}

MIN_HR = 35
MAX_HR = 190
MIN_IBI = 0.30
MAX_IBI = 2.00

META_COLS = {
    "emotion", "stress_level", "subject", "subject_folder", "segment_method",
    "segment_start", "segment_end", "window_start", "window_end",
    "window_seconds", "dataset_source"
}


# ============================================================
# Utility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def finite_values(x):
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def safe_zscore_fit_transform(train_array, test_array):
    mean = np.nanmean(train_array, axis=0, keepdims=True)
    std = np.nanstd(train_array, axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    train_scaled = (train_array - mean) / std
    test_scaled = (test_array - mean) / std
    train_scaled = np.nan_to_num(train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    test_scaled = np.nan_to_num(test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return train_scaled, test_scaled


# ============================================================
# Folder detection
# ============================================================

def find_e4_root(data_dir: Path) -> Path:
    data_dir = Path(data_dir)

    if data_dir.name.lower() == "e4":
        return data_dir

    for p in data_dir.rglob("*"):
        if p.is_dir() and p.name.lower() == "e4":
            return p

    raise FileNotFoundError(
        "Cannot find E4 folder. Use --data_dir pointing to the ECSMP dataset root or the E4 folder."
    )


def find_subject_folders(e4_root: Path):
    folders = []

    for p in e4_root.rglob("*"):
        if not p.is_dir():
            continue

        files = {f.name.lower(): f for f in p.iterdir() if f.is_file()}

        if "hr.csv" in files and "ibi.csv" in files:
            folders.append(p)

    return sorted(set(folders), key=lambda x: str(x))


def get_subject_id(folder: Path) -> str:
    match = re.search(r"(\d+)", folder.name)
    if match:
        return f"S{int(match.group(1)):03d}"
    return folder.name


# ============================================================
# Empatica E4 readers
# ============================================================

def read_empatica_hr(hr_path: Path):
    df = pd.read_csv(hr_path, header=None)
    values = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values.astype(float)

    if len(values) < 3:
        raise ValueError(f"HR file too short: {hr_path}")

    start_ts = float(values[0])
    fs = float(values[1])

    if fs <= 0 or fs > 10:
        fs = 1.0

    hr = values[2:]
    t = np.arange(len(hr), dtype=float) / fs

    return t, hr, start_ts, fs


def read_empatica_ibi(ibi_path: Path, hr_start_ts):
    df = pd.read_csv(ibi_path, header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    if df.shape[1] < 2 or len(df) == 0:
        return np.array([]), np.array([])

    t = pd.to_numeric(df.iloc[:, 0], errors="coerce").values.astype(float)
    ibi = pd.to_numeric(df.iloc[:, 1], errors="coerce").values.astype(float)

    mask = np.isfinite(t) & np.isfinite(ibi)
    t = t[mask]
    ibi = ibi[mask]

    if len(t) > 0 and np.nanmedian(t) > 1e8 and hr_start_ts is not None:
        t = t - hr_start_ts

    return t, ibi


def read_empatica_tags(tags_path: Path, hr_start_ts, signal_end):
    if not tags_path.exists():
        return np.array([])

    try:
        df = pd.read_csv(tags_path, header=None)
    except Exception:
        return np.array([])

    vals = pd.to_numeric(df.values.ravel(), errors="coerce")
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return np.array([])

    if np.nanmedian(vals) > 1e8 and hr_start_ts is not None:
        vals = vals - hr_start_ts

    vals = vals[(vals >= 0) & (vals <= signal_end + 600)]
    vals = np.unique(np.sort(vals))

    return vals


# ============================================================
# Emotion segmentation
# ============================================================

def build_emotion_segments(tags, signal_end, tag_offset):
    usable_tags = tags[tag_offset:] if len(tags) > tag_offset else np.array([])
    segments = []

    if len(usable_tags) >= 6:
        starts = usable_tags[:6]

        if len(usable_tags) >= 7:
            ends = usable_tags[1:7]
        else:
            diffs = np.diff(starts)
            estimated_duration = float(np.median(diffs)) if len(diffs) > 0 else signal_end / 6
            ends = []

            for i, start in enumerate(starts):
                if i < 5:
                    ends.append(starts[i + 1])
                else:
                    ends.append(min(signal_end, start + estimated_duration))

            ends = np.array(ends)

        for emotion, start, end in zip(DATA_EMOTION_ORDER, starts, ends):
            start = float(max(0, start))
            end = float(min(signal_end, end))

            if end - start >= 20:
                segments.append({
                    "emotion": emotion,
                    "start": start,
                    "end": end,
                    "method": "tags"
                })

    if len(segments) == 6:
        return segments

    # fallback only if tags are missing/unusable
    segments = []
    edges = np.linspace(0, signal_end, 7)

    for i, emotion in enumerate(DATA_EMOTION_ORDER):
        segments.append({
            "emotion": emotion,
            "start": float(edges[i]),
            "end": float(edges[i + 1]),
            "method": "equal_split"
        })

    return segments


# ============================================================
# Feature extraction for RandomForest / tabular models
# ============================================================

def add_stat_features(x, prefix):
    x = finite_values(x)
    feats = {}

    keys = [
        "count", "mean", "std", "min", "max", "median",
        "q05", "q10", "q25", "q75", "q90", "q95",
        "iqr", "range", "mad", "cv", "skew", "kurtosis",
        "slope", "first_half_mean", "second_half_mean", "change"
    ]

    if len(x) == 0:
        for k in keys:
            feats[f"{prefix}_{k}"] = np.nan
        return feats

    feats[f"{prefix}_count"] = float(len(x))
    feats[f"{prefix}_mean"] = float(np.mean(x))
    feats[f"{prefix}_std"] = float(np.std(x))
    feats[f"{prefix}_min"] = float(np.min(x))
    feats[f"{prefix}_max"] = float(np.max(x))
    feats[f"{prefix}_median"] = float(np.median(x))
    feats[f"{prefix}_q05"] = float(np.percentile(x, 5))
    feats[f"{prefix}_q10"] = float(np.percentile(x, 10))
    feats[f"{prefix}_q25"] = float(np.percentile(x, 25))
    feats[f"{prefix}_q75"] = float(np.percentile(x, 75))
    feats[f"{prefix}_q90"] = float(np.percentile(x, 90))
    feats[f"{prefix}_q95"] = float(np.percentile(x, 95))
    feats[f"{prefix}_iqr"] = feats[f"{prefix}_q75"] - feats[f"{prefix}_q25"]
    feats[f"{prefix}_range"] = feats[f"{prefix}_max"] - feats[f"{prefix}_min"]
    feats[f"{prefix}_mad"] = float(np.mean(np.abs(x - np.mean(x))))
    feats[f"{prefix}_cv"] = float(np.std(x) / (np.mean(x) + 1e-8))

    if len(x) >= 3:
        feats[f"{prefix}_skew"] = float(skew(x, bias=False))
        feats[f"{prefix}_kurtosis"] = float(kurtosis(x, bias=False))
    else:
        feats[f"{prefix}_skew"] = np.nan
        feats[f"{prefix}_kurtosis"] = np.nan

    if len(x) >= 2:
        feats[f"{prefix}_slope"] = float(np.polyfit(np.arange(len(x)), x, 1)[0])
        mid = len(x) // 2
        first = x[:mid]
        second = x[mid:]
        feats[f"{prefix}_first_half_mean"] = float(np.mean(first)) if len(first) else np.nan
        feats[f"{prefix}_second_half_mean"] = float(np.mean(second)) if len(second) else np.nan
        feats[f"{prefix}_change"] = feats[f"{prefix}_second_half_mean"] - feats[f"{prefix}_first_half_mean"]
    else:
        feats[f"{prefix}_slope"] = np.nan
        feats[f"{prefix}_first_half_mean"] = np.nan
        feats[f"{prefix}_second_half_mean"] = np.nan
        feats[f"{prefix}_change"] = np.nan

    return feats


def add_diff_features(x, prefix):
    x = finite_values(x)
    feats = {}

    keys = [
        "diff_mean", "diff_std", "diff_abs_mean",
        "diff_abs_median", "diff_abs_max", "up_ratio", "down_ratio"
    ]

    if len(x) < 2:
        for k in keys:
            feats[f"{prefix}_{k}"] = np.nan
        return feats

    d = np.diff(x)
    abs_d = np.abs(d)

    feats[f"{prefix}_diff_mean"] = float(np.mean(d))
    feats[f"{prefix}_diff_std"] = float(np.std(d))
    feats[f"{prefix}_diff_abs_mean"] = float(np.mean(abs_d))
    feats[f"{prefix}_diff_abs_median"] = float(np.median(abs_d))
    feats[f"{prefix}_diff_abs_max"] = float(np.max(abs_d))
    feats[f"{prefix}_up_ratio"] = float(np.mean(d > 0))
    feats[f"{prefix}_down_ratio"] = float(np.mean(d < 0))

    return feats


def hrv_time_features(ibi):
    ibi = finite_values(ibi)
    ibi = ibi[(ibi >= MIN_IBI) & (ibi <= MAX_IBI)]

    feats = {
        "sdnn": np.nan, "sdsd": np.nan, "rmssd": np.nan,
        "pnn20": np.nan, "pnn50": np.nan, "nn20": np.nan, "nn50": np.nan,
        "cvnn": np.nan, "cvrmssd": np.nan,
        "sd1": np.nan, "sd2": np.nan, "sd1_sd2_ratio": np.nan,
    }

    if len(ibi) < 2:
        return feats

    diff = np.diff(ibi)
    abs_diff = np.abs(diff)

    feats["sdnn"] = float(np.std(ibi))
    feats["sdsd"] = float(np.std(diff))
    feats["rmssd"] = float(np.sqrt(np.mean(diff ** 2)))
    feats["nn20"] = float(np.sum(abs_diff > 0.020))
    feats["nn50"] = float(np.sum(abs_diff > 0.050))
    feats["pnn20"] = float(np.mean(abs_diff > 0.020))
    feats["pnn50"] = float(np.mean(abs_diff > 0.050))
    feats["cvnn"] = float(np.std(ibi) / (np.mean(ibi) + 1e-8))
    feats["cvrmssd"] = float(feats["rmssd"] / (np.mean(ibi) + 1e-8))

    sd1 = np.sqrt(0.5) * np.std(diff)
    sd2 = np.sqrt(max(0, 2 * np.std(ibi) ** 2 - 0.5 * np.std(diff) ** 2))

    feats["sd1"] = float(sd1)
    feats["sd2"] = float(sd2)
    feats["sd1_sd2_ratio"] = float(sd1 / (sd2 + 1e-8))

    return feats


def hrv_frequency_features(ibi):
    ibi = finite_values(ibi)
    ibi = ibi[(ibi >= MIN_IBI) & (ibi <= MAX_IBI)]

    feats = {
        "hrv_vlf": np.nan,
        "hrv_lf": np.nan,
        "hrv_hf": np.nan,
        "hrv_total_power": np.nan,
        "hrv_lf_hf_ratio": np.nan,
        "hrv_lf_norm": np.nan,
        "hrv_hf_norm": np.nan,
    }

    if len(ibi) < 10:
        return feats

    try:
        t = np.cumsum(ibi)
        fs_interp = 4.0
        t_interp = np.arange(t[0], t[-1], 1 / fs_interp)

        if len(t_interp) < 16:
            return feats

        inst_hr = 60.0 / ibi
        hr_interp = np.interp(t_interp, t, inst_hr)
        hr_interp = hr_interp - np.mean(hr_interp)

        freqs, psd = welch(hr_interp, fs=fs_interp, nperseg=min(256, len(hr_interp)))

        def band_power(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            if not np.any(mask):
                return 0.0
            return float(np.trapz(psd[mask], freqs[mask]))

        vlf = band_power(0.0033, 0.04)
        lf = band_power(0.04, 0.15)
        hf = band_power(0.15, 0.40)
        total = vlf + lf + hf

        feats["hrv_vlf"] = vlf
        feats["hrv_lf"] = lf
        feats["hrv_hf"] = hf
        feats["hrv_total_power"] = total
        feats["hrv_lf_hf_ratio"] = float(lf / (hf + 1e-8))
        feats["hrv_lf_norm"] = float(lf / (lf + hf + 1e-8))
        feats["hrv_hf_norm"] = float(hf / (lf + hf + 1e-8))

    except Exception:
        pass

    return feats


def extract_tabular_features(hr_values, ibi_values):
    hr = finite_values(hr_values)
    ibi = finite_values(ibi_values)

    hr = hr[(hr >= MIN_HR) & (hr <= MAX_HR)]
    ibi = ibi[(ibi >= MIN_IBI) & (ibi <= MAX_IBI)]

    feats = {}
    feats.update(add_stat_features(hr, "hr"))
    feats.update(add_diff_features(hr, "hr"))
    feats.update(add_stat_features(ibi, "ibi"))
    feats.update(add_diff_features(ibi, "ibi"))
    feats.update(hrv_time_features(ibi))
    feats.update(hrv_frequency_features(ibi))

    if len(hr) > 0:
        feats["hr_gt_60_ratio"] = float(np.mean(hr > 60))
        feats["hr_gt_70_ratio"] = float(np.mean(hr > 70))
        feats["hr_gt_80_ratio"] = float(np.mean(hr > 80))
        feats["hr_gt_90_ratio"] = float(np.mean(hr > 90))
        feats["hr_gt_100_ratio"] = float(np.mean(hr > 100))
        feats["hr_lt_60_ratio"] = float(np.mean(hr < 60))
    else:
        for k in ["hr_gt_60_ratio", "hr_gt_70_ratio", "hr_gt_80_ratio", "hr_gt_90_ratio", "hr_gt_100_ratio", "hr_lt_60_ratio"]:
            feats[k] = np.nan

    feats["ibi_inverse_mean"] = 1.0 / (feats.get("ibi_mean", np.nan) + 1e-8)
    feats["hr_mean_x_rmssd"] = feats.get("hr_mean", np.nan) * feats.get("rmssd", np.nan)
    feats["hr_mean_x_sdnn"] = feats.get("hr_mean", np.nan) * feats.get("sdnn", np.nan)
    feats["hr_std_over_rmssd"] = feats.get("hr_std", np.nan) / (feats.get("rmssd", np.nan) + 1e-8)
    feats["hr_ibi_ratio"] = feats.get("hr_mean", np.nan) / (feats.get("ibi_mean", np.nan) + 1e-8)

    return feats


# ============================================================
# Sequence extraction for 1D-CNN and BiLSTM
# ============================================================

def make_sequence_features(hr_t, hr, ibi_t, ibi, ws, we, seq_len):
    grid = np.linspace(ws, we, seq_len)

    hr_clean = np.asarray(hr, dtype=float)
    hr_t_clean = np.asarray(hr_t, dtype=float)
    hr_mask = np.isfinite(hr_t_clean) & np.isfinite(hr_clean) & (hr_clean >= MIN_HR) & (hr_clean <= MAX_HR)
    hr_t_clean = hr_t_clean[hr_mask]
    hr_clean = hr_clean[hr_mask]

    ibi_clean = np.asarray(ibi, dtype=float)
    ibi_t_clean = np.asarray(ibi_t, dtype=float)
    ibi_mask = np.isfinite(ibi_t_clean) & np.isfinite(ibi_clean) & (ibi_clean >= MIN_IBI) & (ibi_clean <= MAX_IBI)
    ibi_t_clean = ibi_t_clean[ibi_mask]
    ibi_clean = ibi_clean[ibi_mask]

    if len(hr_clean) >= 2:
        hr_resampled = np.interp(grid, hr_t_clean, hr_clean)
    elif len(hr_clean) == 1:
        hr_resampled = np.full(seq_len, hr_clean[0])
    else:
        hr_resampled = np.full(seq_len, np.nan)

    if len(ibi_clean) >= 2:
        ibi_resampled = np.interp(grid, ibi_t_clean, ibi_clean)
        inst_hr = 60.0 / ibi_clean
        inst_hr_resampled = np.interp(grid, ibi_t_clean, inst_hr)
    elif len(ibi_clean) == 1:
        ibi_resampled = np.full(seq_len, ibi_clean[0])
        inst_hr_resampled = np.full(seq_len, 60.0 / ibi_clean[0])
    else:
        ibi_resampled = np.full(seq_len, np.nan)
        inst_hr_resampled = np.full(seq_len, np.nan)

    # Channels: HR, IBI, instantaneous HR from IBI, HR first derivative, IBI first derivative
    hr_diff = np.gradient(hr_resampled)
    ibi_diff = np.gradient(ibi_resampled)

    seq = np.stack([hr_resampled, ibi_resampled, inst_hr_resampled, hr_diff, ibi_diff], axis=1)
    return seq.astype(np.float32)


# ============================================================
# Dataset loading
# ============================================================

def window_subject_folder(subject_folder, window_seconds, step_seconds, tag_offset, min_hr_points, min_ibi_points, seq_len):
    hr_path = subject_folder / "HR.csv"
    ibi_path = subject_folder / "IBI.csv"
    tags_path = subject_folder / "tags.csv"

    subject = get_subject_id(subject_folder)

    try:
        hr_t, hr, hr_start_ts, _ = read_empatica_hr(hr_path)
        ibi_t, ibi = read_empatica_ibi(ibi_path, hr_start_ts)
    except Exception as e:
        print(f"[SKIP] {subject_folder.name}: {e}")
        return pd.DataFrame(), []

    signal_end = max(float(hr_t[-1]) if len(hr_t) else 0, float(ibi_t[-1]) if len(ibi_t) else 0)

    if signal_end < 60:
        print(f"[SKIP] {subject}: recording too short")
        return pd.DataFrame(), []

    tags = read_empatica_tags(tags_path, hr_start_ts, signal_end)
    segments = build_emotion_segments(tags, signal_end, tag_offset)

    rows = []
    sequences = []

    for seg in segments:
        emotion = seg["emotion"]
        seg_start = seg["start"]
        seg_end = seg["end"]
        duration = seg_end - seg_start

        if duration <= 0:
            continue

        if duration < window_seconds:
            window_starts = [seg_start]
            actual_window = duration
        else:
            window_starts = np.arange(seg_start, seg_end - window_seconds + 1, step_seconds)
            actual_window = window_seconds

        for ws in window_starts:
            we = min(ws + actual_window, seg_end)

            hr_mask = (hr_t >= ws) & (hr_t < we)
            ibi_mask = (ibi_t >= ws) & (ibi_t < we)

            hr_win = hr[hr_mask]
            ibi_win = ibi[ibi_mask]

            if len(hr_win) < min_hr_points and len(ibi_win) < min_ibi_points:
                continue

            feats = extract_tabular_features(hr_win, ibi_win)

            feats["emotion"] = emotion
            feats["stress_level"] = EMOTION_TO_STRESS[emotion]
            feats["subject"] = subject
            feats["subject_folder"] = subject_folder.name
            feats["segment_method"] = seg["method"]
            feats["segment_start"] = seg_start
            feats["segment_end"] = seg_end
            feats["window_start"] = float(ws)
            feats["window_end"] = float(we)
            feats["window_seconds"] = float(we - ws)
            feats["dataset_source"] = "ECSMP_E4_HR_IBI"

            seq = make_sequence_features(hr_t, hr, ibi_t, ibi, ws, we, seq_len)

            rows.append(feats)
            sequences.append(seq)

    df = pd.DataFrame(rows)

    if not df.empty:
        print(
            f"[OK] {subject}: rows={len(df)} | tags={len(tags)} | "
            f"method={df['segment_method'].mode().iloc[0]}"
        )

    return df, sequences


def load_dataset(args):
    e4_root = find_e4_root(Path(args.data_dir))
    print(f"[INFO] E4 folder: {e4_root}")

    subject_folders = find_subject_folders(e4_root)

    if not subject_folders:
        raise FileNotFoundError("No subject folders with HR.csv and IBI.csv were found.")

    print(f"[INFO] Subject folders found: {len(subject_folders)}")

    frames = []
    all_sequences = []

    for folder in subject_folders:
        df, seqs = window_subject_folder(
            subject_folder=folder,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            tag_offset=args.tag_offset,
            min_hr_points=args.min_hr_points,
            min_ibi_points=args.min_ibi_points,
            seq_len=args.seq_len,
        )

        if not df.empty:
            frames.append(df)
            all_sequences.extend(seqs)

    if not frames:
        raise ValueError("No usable HR/IBI windows were created.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    sequences = np.stack(all_sequences, axis=0).astype(np.float32)

    print("\n========== Loaded ECSMP E4 HR/IBI Dataset ==========")
    print(f"Rows: {len(combined)}")
    print(f"Sequence shape: {sequences.shape}")
    print("Class counts:")
    print(combined["emotion"].value_counts())
    print("Segment method counts:")
    print(combined["segment_method"].value_counts())

    return combined, sequences


# ============================================================
# Training table preparation
# ============================================================

def add_subject_normalized_features(df, feature_cols):
    df = df.copy()
    new_cols = []

    for col in feature_cols:
        z_col = f"{col}_subject_z"
        rel_col = f"{col}_subject_rel"

        group = df.groupby("subject")[col]
        med = group.transform("median")
        std = group.transform("std").replace(0, np.nan)

        df[z_col] = (df[col] - med) / (std + 1e-8)
        df[rel_col] = df[col] - med

        new_cols.extend([z_col, rel_col])

    return df, feature_cols + new_cols


def prepare_training_table(df, baseline_calibration=True):
    clean_df = df.copy()
    clean_df = clean_df[clean_df["emotion"].isin(REPORT_LABELS)].copy()

    feature_cols = [c for c in clean_df.columns if c not in META_COLS]

    for col in feature_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df[feature_cols] = clean_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    empty_cols = [c for c in feature_cols if clean_df[c].isna().all()]
    clean_df = clean_df.drop(columns=empty_cols, errors="ignore")
    feature_cols = [c for c in feature_cols if c not in empty_cols]

    if baseline_calibration:
        clean_df, feature_cols = add_subject_normalized_features(clean_df, feature_cols)

    non_null_count = clean_df[feature_cols].notna().sum(axis=1)
    clean_df = clean_df[non_null_count >= 5].copy()

    return clean_df, feature_cols


def make_group_split(X, y, groups, test_size, random_state):
    all_classes = set(y.unique())

    for seed_add in range(300):
        seed = random_state + seed_add
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups))

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if set(y_train.unique()) == all_classes and set(y_test.unique()) == all_classes:
            return train_idx, test_idx, seed

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return train_idx, test_idx, random_state


def split_data(df, feature_cols, sequences, split_mode, test_size, random_state):
    X = df[feature_cols]
    y = df["emotion"]

    if split_mode == "subject":
        groups = df["subject"]
        train_idx, test_idx, used_seed = make_group_split(X, y, groups, test_size, random_state)
    else:
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
        used_seed = random_state

    return {
        "X_train": X.iloc[train_idx].copy(),
        "X_test": X.iloc[test_idx].copy(),
        "y_train": y.iloc[train_idx].copy(),
        "y_test": y.iloc[test_idx].copy(),
        "seq_train": sequences[train_idx],
        "seq_test": sequences[test_idx],
        "train_meta": df.iloc[train_idx].copy(),
        "test_meta": df.iloc[test_idx].copy(),
        "train_idx": train_idx,
        "test_idx": test_idx,
        "used_seed": used_seed,
    }


def oversample_training(X_train, y_train, seq_train=None):
    temp = X_train.copy()
    temp["__target__"] = y_train.values
    temp["__orig_index__"] = np.arange(len(X_train))

    max_count = temp["__target__"].value_counts().max()
    parts = []

    for label, group in temp.groupby("__target__"):
        if len(group) < max_count:
            group = group.sample(max_count, replace=True, random_state=RANDOM_STATE)
        parts.append(group)

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    orig_indices = out["__orig_index__"].values.astype(int)
    y_out = out.pop("__target__")
    out.pop("__orig_index__")
    X_out = out

    if seq_train is not None:
        seq_out = seq_train[orig_indices]
        return X_out, y_out, seq_out

    return X_out, y_out


# ============================================================
# Classical models
# ============================================================

def build_classical_models(random_state, include_svc=False):
    models = {
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=2000,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),

        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=1500,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),

        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.035,
                max_iter=900,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=random_state,
            )),
        ]),

        "mlp": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                alpha=0.0008,
                learning_rate_init=0.0008,
                max_iter=1200,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=random_state,
            )),
        ]),
    }

    # SVC_RBF is accurate sometimes, but it can become extremely slow on this
    # windowed HR/IBI dataset. It is disabled by default to prevent freezing.
    if include_svc:
        models["svc_rbf"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=15,
                gamma="scale",
                class_weight="balanced",
                probability=False,
                random_state=random_state,
                max_iter=5000,
            )),
        ])

    return models


def train_classical_models(X_fit, y_fit, X_test, y_test, random_state, include_svc=False):
    models = build_classical_models(random_state, include_svc=include_svc)
    results = []
    fitted = {}
    preds = {}

    print("\n========== Training Classical Models ==========")

    for name, model in models.items():
        try:
            model.fit(X_fit, y_fit)
            pred = model.predict(X_test)

            acc = accuracy_score(y_test, pred)
            bal = balanced_accuracy_score(y_test, pred)
            macro = f1_score(y_test, pred, average="macro", zero_division=0)
            weighted = f1_score(y_test, pred, average="weighted", zero_division=0)

            print(f"\n[{name}]")
            print(f"Accuracy          : {acc:.4f}")
            print(f"Balanced Accuracy : {bal:.4f}")
            print(f"Macro F1          : {macro:.4f}")
            print(f"Weighted F1       : {weighted:.4f}")

            results.append({
                "model": name,
                "accuracy": acc,
                "balanced_accuracy": bal,
                "macro_f1": macro,
                "weighted_f1": weighted,
            })
            fitted[name] = model
            preds[name] = pred

        except Exception as e:
            print(f"[WARN] {name} failed: {e}")

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise RuntimeError("All classical models failed.")

    results_df = results_df.sort_values(["macro_f1", "accuracy"], ascending=False)

    # Soft voting top 3 classical models
    top_names = results_df.head(3)["model"].tolist()

    try:
        estimators = [(name, clone(build_classical_models(random_state, include_svc=include_svc)[name])) for name in top_names]
        voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
        voting.fit(X_fit, y_fit)
        pred = voting.predict(X_test)

        acc = accuracy_score(y_test, pred)
        bal = balanced_accuracy_score(y_test, pred)
        macro = f1_score(y_test, pred, average="macro", zero_division=0)
        weighted = f1_score(y_test, pred, average="weighted", zero_division=0)

        print("\n[soft_voting_top3]")
        print(f"Accuracy          : {acc:.4f}")
        print(f"Balanced Accuracy : {bal:.4f}")
        print(f"Macro F1          : {macro:.4f}")
        print(f"Weighted F1       : {weighted:.4f}")

        results_df = pd.concat([
            results_df,
            pd.DataFrame([{
                "model": "soft_voting_top3",
                "accuracy": acc,
                "balanced_accuracy": bal,
                "macro_f1": macro,
                "weighted_f1": weighted,
            }])
        ], ignore_index=True)

        fitted["soft_voting_top3"] = voting
        preds["soft_voting_top3"] = pred

    except Exception as e:
        print(f"[WARN] soft voting skipped: {e}")

    results_df = results_df.sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
    return results_df, fitted, preds


# ============================================================
# Deep learning models: 1D-CNN and BiLSTM
# ============================================================

if TORCH_AVAILABLE:
    class CNN1DClassifier(nn.Module):
        def __init__(self, input_channels, n_classes, dropout=0.30):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),

                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),

                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),

                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            # x: batch, time, channels
            x = x.permute(0, 2, 1)
            return self.net(x)


    class BiLSTMClassifier(nn.Module):
        def __init__(self, input_channels, n_classes, hidden_size=96, num_layers=2, dropout=0.30):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size * 2),
                nn.Linear(hidden_size * 2, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            pooled = out.mean(dim=1)
            return self.head(pooled)


def encode_labels(y_train, y_test):
    label_to_id = {label: i for i, label in enumerate(REPORT_LABELS)}
    y_train_id = np.array([label_to_id[v] for v in y_train], dtype=np.int64)
    y_test_id = np.array([label_to_id[v] for v in y_test], dtype=np.int64)
    return y_train_id, y_test_id, label_to_id


def scale_sequences(seq_train, seq_test):
    # Fit channel mean/std on training data only.
    mean = np.nanmean(seq_train, axis=(0, 1), keepdims=True)
    std = np.nanstd(seq_train, axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1.0, std)

    train_scaled = (seq_train - mean) / std
    test_scaled = (seq_test - mean) / std

    train_scaled = np.nan_to_num(train_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    test_scaled = np.nan_to_num(test_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    scaler = {"mean": mean, "std": std}
    return train_scaled, test_scaled, scaler


def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate_deep(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item() * xb.size(0)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_true.append(yb.cpu().numpy())

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "probs": np.vstack(all_probs),
        "preds": np.concatenate(all_preds),
        "true": np.concatenate(all_true),
    }


def train_one_deep_model(model_name, model, X_train, y_train, X_test, y_test, args, class_weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device)

    # Split train into train/validation for early stopping.
    idx = np.arange(len(X_train))
    tr_idx, val_idx = train_test_split(
        idx,
        test_size=args.deep_val_size,
        random_state=args.random_state,
        stratify=y_train,
    )

    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    train_loader = make_loader(X_tr, y_tr, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, args.batch_size, shuffle=False)

    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n========== Training {model_name} ==========")
    print(f"Device: {device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_eval = evaluate_deep(model, val_loader, criterion, device)
        val_loss = val_eval["loss"]
        val_acc = val_eval["accuracy"]
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    model.load_state_dict(best_state)
    test_eval = evaluate_deep(model, test_loader, criterion, device)

    pred_ids = test_eval["preds"]
    pred_labels = np.array([REPORT_LABELS[i] for i in pred_ids])

    true_labels = np.array([REPORT_LABELS[i] for i in y_test])

    acc = accuracy_score(true_labels, pred_labels)
    bal = balanced_accuracy_score(true_labels, pred_labels)
    macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    weighted = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    print(f"\n[{model_name}]")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Balanced Accuracy : {bal:.4f}")
    print(f"Macro F1          : {macro:.4f}")
    print(f"Weighted F1       : {weighted:.4f}")
    print(f"Best epoch        : {best_epoch}")

    result = {
        "model": model_name,
        "accuracy": acc,
        "balanced_accuracy": bal,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }

    return result, model, pred_labels, test_eval["probs"], history


def plot_deep_history(history, output_path_prefix):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_path_prefix) + "_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_path_prefix) + "_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Reports
# ============================================================

def save_confusion_matrix(y_true, y_pred, labels, output_path, normalize=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, values_format=".2f" if normalize else "d", xticks_rotation=45, colorbar=True)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_report_txt(output_path, model_name, y_true, y_pred, settings, feature_count, row_count, train_rows, test_rows, extra=None):
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        labels=REPORT_LABELS,
        digits=4,
        zero_division=0,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Balanced Accuracy: {bal:.4f}\n")
        f.write(f"Macro F1: {macro:.4f}\n")
        f.write(f"Weighted F1: {weighted:.4f}\n")
        f.write("\n")
        f.write("Training Settings:\n")
        f.write("Dataset: ECSMP\n")
        f.write("Feature scope: HR + IBI only\n")
        f.write("Task: Six-emotion recognition\n")
        f.write("Emotion labels: anger, disgust, fear, happy, neutral, sad\n")
        f.write("Stress mapping: happy->low, neutral->baseline, sad/disgust->medium, fear/anger->high\n")
        f.write(f"Model type: {model_name}\n")
        f.write(f"Split: {settings['split']}\n")
        f.write(f"Balance train: {settings['balance_train']}\n")
        f.write(f"Baseline calibration: {settings['baseline_calibration']}\n")
        f.write(f"Window size: {settings['window_seconds']}\n")
        f.write(f"Step size: {settings['step_seconds']}\n")
        f.write(f"Tag offset: {settings['tag_offset']}\n")
        f.write(f"Sequence length: {settings['seq_len']}\n")
        f.write(f"Test size: {settings['test_size']}\n")
        f.write(f"Random state: {settings['random_state']}\n")
        f.write(f"Rows: {row_count}\n")
        f.write(f"Train rows: {train_rows}\n")
        f.write(f"Test rows: {test_rows}\n")
        f.write(f"Feature count: {feature_count}\n")

        if extra:
            f.write("\nExtra Settings:\n")
            for k, v in extra.items():
                f.write(f"{k}: {v}\n")

    return {
        "model": model_name,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
    }


# ============================================================
# Main train
# ============================================================

def train(args):
    set_seed(args.random_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, sequences = load_dataset(args)
    clean_df, feature_cols = prepare_training_table(df, baseline_calibration=not args.no_baseline_calibration)

    # keep sequences aligned after cleaning rows
    clean_indices = clean_df.index.values
    sequences = sequences[clean_indices]
    clean_df = clean_df.reset_index(drop=True)

    missing = sorted(set(REPORT_LABELS) - set(clean_df["emotion"].unique()))
    if missing:
        raise ValueError(f"Missing emotion classes after loading: {missing}")

    print("\n========== Final Training Table ==========")
    print(f"Rows: {len(clean_df)}")
    print(f"Tabular features: {len(feature_cols)}")
    print(f"Sequence shape: {sequences.shape}")
    print("Class counts:")
    print(clean_df["emotion"].value_counts())
    print(f"Split mode: {args.split}")

    clean_df.to_csv(output_dir / "ecsmp_hr_ibi_training_table.csv", index=False)
    summary = clean_df.groupby(["subject", "emotion"]).size().reset_index(name="rows")
    summary.to_csv(output_dir / "segment_summary.csv", index=False)

    split = split_data(
        df=clean_df,
        feature_cols=feature_cols,
        sequences=sequences,
        split_mode=args.split,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    X_train = split["X_train"]
    X_test = split["X_test"]
    y_train = split["y_train"]
    y_test = split["y_test"]
    seq_train = split["seq_train"]
    seq_test = split["seq_test"]
    used_seed = split["used_seed"]

    print("\nTrain class counts before balancing:")
    print(y_train.value_counts())
    print("\nTest class counts:")
    print(y_test.value_counts())

    if not args.no_balance_train:
        X_fit, y_fit, seq_fit = oversample_training(X_train, y_train, seq_train)
        print("\nTrain class counts after oversampling:")
        print(y_fit.value_counts())
    else:
        X_fit, y_fit, seq_fit = X_train, y_train, seq_train

    settings = {
        "split": args.split,
        "balance_train": not args.no_balance_train,
        "baseline_calibration": not args.no_baseline_calibration,
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "tag_offset": args.tag_offset,
        "seq_len": args.seq_len,
        "test_size": args.test_size,
        "random_state": used_seed,
    }

    all_results = []
    all_predictions = {}
    saved_models = {}

    # ------------------------------
    # Classical ML models
    # ------------------------------
    classical_results, classical_models, classical_preds = train_classical_models(
        X_fit=X_fit,
        y_fit=y_fit,
        X_test=X_test,
        y_test=y_test,
        random_state=used_seed,
        include_svc=args.include_svc,
    )

    classical_results.to_csv(output_dir / "candidate_classical_model_results.csv", index=False)

    for _, row in classical_results.iterrows():
        name = row["model"]
        pred = classical_preds[name]
        metrics = write_report_txt(
            output_path=output_dir / f"classification_report_{name}.txt",
            model_name=name,
            y_true=y_test,
            y_pred=pred,
            settings=settings,
            feature_count=len(feature_cols),
            row_count=len(clean_df),
            train_rows=len(X_train),
            test_rows=len(X_test),
        )
        all_results.append(metrics)
        all_predictions[name] = pred
        saved_models[name] = classical_models[name]

    # ------------------------------
    # Deep learning models
    # ------------------------------
    if args.train_deep:
        if not TORCH_AVAILABLE:
            print("\n[WARN] PyTorch is not installed. Skipping 1D-CNN and BiLSTM.")
        else:
            y_fit_id, y_test_id, label_to_id = encode_labels(y_fit.values, y_test.values)

            seq_fit_scaled, seq_test_scaled, seq_scaler = scale_sequences(seq_fit, seq_test)

            counts = pd.Series(y_fit.values).value_counts()
            class_weights = []
            for label in REPORT_LABELS:
                class_weights.append(len(y_fit) / (len(REPORT_LABELS) * counts.get(label, 1)))
            class_weights = np.array(class_weights, dtype=np.float32)

            input_channels = seq_fit_scaled.shape[2]
            n_classes = len(REPORT_LABELS)

            deep_jobs = []

            if args.deep_model in ["cnn", "both"]:
                deep_jobs.append((
                    "1dcnn_raw_sequence",
                    CNN1DClassifier(input_channels=input_channels, n_classes=n_classes, dropout=args.dropout)
                ))

            if args.deep_model in ["bilstm", "both"]:
                deep_jobs.append((
                    "bilstm_raw_sequence",
                    BiLSTMClassifier(input_channels=input_channels, n_classes=n_classes, hidden_size=args.hidden_size, num_layers=args.lstm_layers, dropout=args.dropout)
                ))

            for model_name, model in deep_jobs:
                result, trained_model, pred_labels, probs, history = train_one_deep_model(
                    model_name=model_name,
                    model=model,
                    X_train=seq_fit_scaled,
                    y_train=y_fit_id,
                    X_test=seq_test_scaled,
                    y_test=y_test_id,
                    args=args,
                    class_weights=class_weights,
                )

                write_report_txt(
                    output_path=output_dir / f"classification_report_{model_name}.txt",
                    model_name=model_name,
                    y_true=y_test,
                    y_pred=pred_labels,
                    settings=settings,
                    feature_count=len(feature_cols),
                    row_count=len(clean_df),
                    train_rows=len(X_train),
                    test_rows=len(X_test),
                    extra={
                        "deep_epochs_requested": args.epochs,
                        "best_epoch": result.get("best_epoch"),
                        "batch_size": args.batch_size,
                        "learning_rate": args.lr,
                        "dropout": args.dropout,
                        "input_channels": input_channels,
                    },
                )

                save_confusion_matrix(
                    y_true=y_test,
                    y_pred=pred_labels,
                    labels=REPORT_LABELS,
                    output_path=output_dir / f"confusion_matrix_{model_name}.png",
                    normalize=None,
                    title=f"{model_name} Confusion Matrix",
                )

                save_confusion_matrix(
                    y_true=y_test,
                    y_pred=pred_labels,
                    labels=REPORT_LABELS,
                    output_path=output_dir / f"confusion_matrix_{model_name}_normalized.png",
                    normalize="true",
                    title=f"{model_name} Normalized Confusion Matrix",
                )

                plot_deep_history(history, output_dir / f"training_curve_{model_name}")

                torch.save({
                    "model_state_dict": trained_model.state_dict(),
                    "model_name": model_name,
                    "labels": REPORT_LABELS,
                    "input_channels": input_channels,
                    "seq_len": args.seq_len,
                    "sequence_scaler": seq_scaler,
                    "settings": settings,
                }, output_dir / f"{model_name}.pt")

                all_results.append({
                    "model": model_name,
                    "accuracy": float(result["accuracy"]),
                    "balanced_accuracy": float(result["balanced_accuracy"]),
                    "macro_f1": float(result["macro_f1"]),
                    "weighted_f1": float(result["weighted_f1"]),
                })
                all_predictions[model_name] = pred_labels
                saved_models[model_name] = trained_model

    # ------------------------------
    # Final selection by macro F1, then accuracy
    # ------------------------------
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
    results_df.to_csv(output_dir / "all_model_results.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_pred = all_predictions[best_name]

    print("\n========== Final Best Model ==========")
    print(results_df)
    print(f"\n[BEST] {best_name}")
    print(classification_report(y_test, best_pred, labels=REPORT_LABELS, digits=4, zero_division=0))

    final_metrics = write_report_txt(
        output_path=output_dir / "final_best_classification_report.txt",
        model_name=best_name,
        y_true=y_test,
        y_pred=best_pred,
        settings=settings,
        feature_count=len(feature_cols),
        row_count=len(clean_df),
        train_rows=len(X_train),
        test_rows=len(X_test),
    )

    save_confusion_matrix(
        y_true=y_test,
        y_pred=best_pred,
        labels=REPORT_LABELS,
        output_path=output_dir / "final_best_confusion_matrix.png",
        normalize=None,
        title=f"Best Model Confusion Matrix: {best_name}",
    )

    save_confusion_matrix(
        y_true=y_test,
        y_pred=best_pred,
        labels=REPORT_LABELS,
        output_path=output_dir / "final_best_confusion_matrix_normalized.png",
        normalize="true",
        title=f"Best Model Normalized Confusion Matrix: {best_name}",
    )

    predictions_df = split["test_meta"][[
        "subject", "subject_folder", "emotion", "stress_level",
        "segment_method", "window_start", "window_end"
    ]].copy()
    predictions_df = predictions_df.rename(columns={"emotion": "true_label"})
    predictions_df["predicted_label"] = best_pred
    predictions_df["predicted_stress"] = predictions_df["predicted_label"].map(EMOTION_TO_STRESS)
    predictions_df.to_csv(output_dir / "final_best_test_predictions.csv", index=False)

    joblib.dump({
        "best_model_name": best_name,
        "model": saved_models[best_name] if best_name in saved_models and not str(best_name).startswith(("1dcnn", "bilstm")) else None,
        "feature_columns": feature_cols,
        "labels": REPORT_LABELS,
        "emotion_to_stress": EMOTION_TO_STRESS,
        "settings": settings,
        "metrics": final_metrics,
        "note": "For deep models, load the .pt file instead of this joblib model field.",
    }, output_dir / "emostress_best_model_bundle.joblib")

    with open(output_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_name,
            "metrics": final_metrics,
            "all_results": results_df.to_dict(orient="records"),
            "settings": settings,
            "emotion_to_stress": EMOTION_TO_STRESS,
        }, f, indent=4)

    with open(output_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=4)

    print("\n========== Saved Outputs ==========")
    print(f"All results: {output_dir / 'all_model_results.csv'}")
    print(f"Best TXT report: {output_dir / 'final_best_classification_report.txt'}")
    print(f"Best confusion matrix: {output_dir / 'final_best_confusion_matrix.png'}")
    print(f"Best normalized matrix: {output_dir / 'final_best_confusion_matrix_normalized.png'}")
    print(f"Predictions: {output_dir / 'final_best_test_predictions.csv'}")
    print(f"Metrics: {output_dir / 'final_metrics.json'}")

    if final_metrics["accuracy"] >= 0.80:
        print("\n[SUCCESS] Best model reached 80%+ accuracy.")
    else:
        print("\n[NOTE] Best model is below 80% accuracy.")
        print("This script does not fake or override metrics. Try sample split, tag_offset 0/1, and 120s/180s windows.")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train ECSMP E4 HR + IBI six-emotion models: classical ML + 1D-CNN + BiLSTM."
    )

    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="models_ecsmp_hr_ibi_rf_cnn_bilstm")

    parser.add_argument("--split", type=str, default="sample", choices=["sample", "subject"])
    parser.add_argument("--test_size", type=float, default=0.20)

    parser.add_argument("--window_seconds", type=int, default=120)
    parser.add_argument("--step_seconds", type=int, default=15)
    parser.add_argument("--tag_offset", type=int, default=1)

    parser.add_argument("--min_hr_points", type=int, default=20)
    parser.add_argument("--min_ibi_points", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=120)

    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--no_balance_train", action="store_true")
    parser.add_argument("--no_baseline_calibration", action="store_true")

    parser.add_argument("--include_svc", action="store_true", help="Also train SVC_RBF. This is disabled by default because it can be very slow.")
    parser.add_argument("--train_deep", action="store_true", help="Train 1D-CNN and/or BiLSTM using PyTorch.")
    parser.add_argument("--deep_model", type=str, default="both", choices=["cnn", "bilstm", "both"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--hidden_size", type=int, default=96)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--deep_val_size", type=float, default=0.15)
    parser.add_argument("--print_every", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
