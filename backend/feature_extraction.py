import pandas as pd
import numpy as np
from scipy import stats

# ─────────────────────────────────────────────────────────────
# Core statistical feature extractor for any numeric series
# Matches the feature set expected by the WESAD stress model
# ─────────────────────────────────────────────────────────────
def extract_hr_ibi_features(series, prefix):
    """Extract the full 25-feature set per signal (HR or IBI)."""
    base = {f"{prefix}_{k}": 0 for k in [
        'count','mean','std','min','max','median','q25','q75','iqr','range',
        'rms','energy','cv','skew','kurtosis',
        'diff_mean','diff_std','diff_abs_mean','diff_abs_max','slope',
        'sdnn','rmssd','sdsd','nn20','nn50','pnn20','pnn50','ln_rmssd',
        'estimated_hr_count','estimated_hr_mean','estimated_hr_std',
        'estimated_hr_min','estimated_hr_max','estimated_hr_median',
        'estimated_hr_q25','estimated_hr_q75','estimated_hr_iqr',
        'estimated_hr_range','estimated_hr_rms','estimated_hr_energy',
        'estimated_hr_cv','estimated_hr_skew','estimated_hr_kurtosis',
        'estimated_hr_diff_mean','estimated_hr_diff_std',
        'estimated_hr_diff_abs_mean','estimated_hr_diff_abs_max',
    ]}
    # HR-specific threshold ratios only apply to HR prefix
    if prefix == 'hr':
        for thresh in [70, 80, 90, 100]:
            base[f"{prefix}_above_{thresh}_ratio"] = 0
        base[f"{prefix}_below_60_ratio"] = 0
        base[f"{prefix}_slope"] = 0

    if series is None or len(series) == 0:
        return base

    s = pd.to_numeric(series, errors='coerce').dropna().values
    if len(s) == 0:
        return base

    diff = np.diff(s)
    q25 = np.percentile(s, 25)
    q75 = np.percentile(s, 75)
    mean_s = np.mean(s)
    std_s = np.std(s, ddof=1) if len(s) > 1 else 0
    rmssd = float(np.sqrt(np.mean(diff**2))) if len(diff) > 0 else 0
    sdnn = std_s
    sdsd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0
    nn20 = int(np.sum(np.abs(diff) > 0.020)) if len(diff) > 0 else 0
    nn50 = int(np.sum(np.abs(diff) > 0.050)) if len(diff) > 0 else 0

    feats = {
        f"{prefix}_count": len(s),
        f"{prefix}_mean": mean_s,
        f"{prefix}_std": std_s,
        f"{prefix}_min": float(np.min(s)),
        f"{prefix}_max": float(np.max(s)),
        f"{prefix}_median": float(np.median(s)),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q75": float(q75),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_range": float(np.max(s) - np.min(s)),
        f"{prefix}_rms": float(np.sqrt(np.mean(s**2))),
        f"{prefix}_energy": float(np.sum(s**2)),
        f"{prefix}_cv": float(std_s / mean_s) if mean_s != 0 else 0,
        f"{prefix}_skew": float(stats.skew(s)) if len(s) > 2 else 0,
        f"{prefix}_kurtosis": float(stats.kurtosis(s)) if len(s) > 3 else 0,
        f"{prefix}_diff_mean": float(np.mean(diff)) if len(diff) > 0 else 0,
        f"{prefix}_diff_std": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0,
        f"{prefix}_diff_abs_mean": float(np.mean(np.abs(diff))) if len(diff) > 0 else 0,
        f"{prefix}_diff_abs_max": float(np.max(np.abs(diff))) if len(diff) > 0 else 0,
        f"{prefix}_slope": float(np.polyfit(np.arange(len(s)), s, 1)[0]) if len(s) > 1 else 0,
        f"{prefix}_sdnn": sdnn,
        f"{prefix}_rmssd": rmssd,
        f"{prefix}_sdsd": sdsd,
        f"{prefix}_nn20": nn20,
        f"{prefix}_nn50": nn50,
        f"{prefix}_pnn20": float(nn20 / len(diff)) if len(diff) > 0 else 0,
        f"{prefix}_pnn50": float(nn50 / len(diff)) if len(diff) > 0 else 0,
        f"{prefix}_ln_rmssd": float(np.log(rmssd)) if rmssd > 0 else 0,
    }

    # Estimated HR from IBI (60 / ibi_s)
    if prefix == 'ibi' and mean_s > 0:
        est_hr = 60.0 / s
        est_hr = est_hr[np.isfinite(est_hr)]
        if len(est_hr) > 0:
            e_diff = np.diff(est_hr)
            feats.update({
                f"{prefix}_estimated_hr_count": len(est_hr),
                f"{prefix}_estimated_hr_mean": float(np.mean(est_hr)),
                f"{prefix}_estimated_hr_std": float(np.std(est_hr, ddof=1)) if len(est_hr) > 1 else 0,
                f"{prefix}_estimated_hr_min": float(np.min(est_hr)),
                f"{prefix}_estimated_hr_max": float(np.max(est_hr)),
                f"{prefix}_estimated_hr_median": float(np.median(est_hr)),
                f"{prefix}_estimated_hr_q25": float(np.percentile(est_hr, 25)),
                f"{prefix}_estimated_hr_q75": float(np.percentile(est_hr, 75)),
                f"{prefix}_estimated_hr_iqr": float(np.percentile(est_hr, 75) - np.percentile(est_hr, 25)),
                f"{prefix}_estimated_hr_range": float(np.max(est_hr) - np.min(est_hr)),
                f"{prefix}_estimated_hr_rms": float(np.sqrt(np.mean(est_hr**2))),
                f"{prefix}_estimated_hr_energy": float(np.sum(est_hr**2)),
                f"{prefix}_estimated_hr_cv": float(np.std(est_hr, ddof=1) / np.mean(est_hr)) if np.mean(est_hr) != 0 else 0,
                f"{prefix}_estimated_hr_skew": float(stats.skew(est_hr)) if len(est_hr) > 2 else 0,
                f"{prefix}_estimated_hr_kurtosis": float(stats.kurtosis(est_hr)) if len(est_hr) > 3 else 0,
                f"{prefix}_estimated_hr_diff_mean": float(np.mean(e_diff)) if len(e_diff) > 0 else 0,
                f"{prefix}_estimated_hr_diff_std": float(np.std(e_diff, ddof=1)) if len(e_diff) > 1 else 0,
                f"{prefix}_estimated_hr_diff_abs_mean": float(np.mean(np.abs(e_diff))) if len(e_diff) > 0 else 0,
                f"{prefix}_estimated_hr_diff_abs_max": float(np.max(np.abs(e_diff))) if len(e_diff) > 0 else 0,
            })

    # HR threshold ratios
    if prefix == 'hr':
        feats[f"hr_above_70_ratio"] = float(np.mean(s > 70))
        feats[f"hr_above_80_ratio"] = float(np.mean(s > 80))
        feats[f"hr_above_90_ratio"] = float(np.mean(s > 90))
        feats[f"hr_above_100_ratio"] = float(np.mean(s > 100))
        feats[f"hr_below_60_ratio"] = float(np.mean(s < 60))

    return feats


def _add_baseline_delta_features(feats, baseline_feats, prefix, features_list):
    """Add delta/ratio/z-score features relative to a baseline window."""
    for key in list(feats.keys()):
        short_key = key  # e.g. hr_mean
        base_key = short_key  # same key in baseline
        if base_key in baseline_feats:
            base_val = baseline_feats[base_key]
            cur_val = feats[key]
            delta_key = f"{key}_delta_base"
            ratio_key = f"{key}_ratio_base"
            z_key = f"{key}_z_base"
            if delta_key in features_list:
                feats[delta_key] = cur_val - base_val
            if ratio_key in features_list:
                feats[ratio_key] = cur_val / base_val if base_val != 0 else 1.0
            if z_key in features_list:
                feats[z_key] = (cur_val - base_val) / (abs(base_val) + 1e-9)
    return feats


def parse_hr_series(df):
    """Parse HR CSV into a clean numeric series."""
    if df is None or df.empty:
        return None
    # Empatica E4: first row = start timestamp, second = sample rate, rest = values
    try:
        if len(df.columns) == 1:
            series = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
            if len(series) > 2:
                return series.iloc[2:]  # skip timestamp + rate
            return series
        else:
            # Multi-column: find the HR column
            numeric_df = df.apply(pd.to_numeric, errors='coerce')
            return numeric_df.iloc[:, 0].dropna()
    except Exception:
        return None


def parse_ibi_series(df):
    """Parse IBI CSV: col 0 = time offset, col 1 = IBI in seconds."""
    if df is None or df.empty:
        return None
    try:
        if len(df.columns) >= 2:
            series = pd.to_numeric(df.iloc[1:, 1], errors='coerce').dropna()
        else:
            series = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
        return series if len(series) > 0 else None
    except Exception:
        return None


def combine_all_features(hr_df, ibi_df, ecg_exp_bin=None, ecg_sleep_bin=None, stress_features_list=None):
    """
    Extract the full feature vector compatible with the WESAD stress model.
    stress_features_list: the exact list of 286 feature column names expected by the model.
    """
    hr_series = parse_hr_series(hr_df)
    ibi_series = parse_ibi_series(ibi_df)

    hr_feats = extract_hr_ibi_features(hr_series, 'hr')
    ibi_feats = extract_hr_ibi_features(ibi_series, 'ibi')

    features = {}
    features.update(hr_feats)
    features.update(ibi_feats)

    # If we have a stress features list, add baseline delta features (zeroed baseline)
    if stress_features_list:
        # Use a zeroed baseline since we don't have a separate resting window
        baseline = {k: 0 for k in features}
        features = _add_baseline_delta_features(features, baseline, 'hr', stress_features_list)
        features = _add_baseline_delta_features(features, baseline, 'ibi', stress_features_list)

    return features
