import pandas as pd
import numpy as np
from scipy import stats

def extract_base_features(series, prefix):
    if series is None or len(series) == 0:
        # Default 0 if missing
        return {f"{prefix}_{feat}": 0 for feat in [
            'mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 
            'iqr', 'range', 'skew', 'kurtosis', 'rms', 'diff_mean', 'diff_std'
        ]}
    
    series = series.dropna()
    if len(series) == 0:
        return extract_base_features(None, prefix)

    diff = np.diff(series)
    
    q25 = np.percentile(series, 25)
    q75 = np.percentile(series, 75)
    
    return {
        f"{prefix}_mean": np.mean(series),
        f"{prefix}_std": np.std(series, ddof=1) if len(series) > 1 else 0,
        f"{prefix}_min": np.min(series),
        f"{prefix}_max": np.max(series),
        f"{prefix}_median": np.median(series),
        f"{prefix}_q25": q25,
        f"{prefix}_q75": q75,
        f"{prefix}_iqr": q75 - q25,
        f"{prefix}_range": np.max(series) - np.min(series),
        f"{prefix}_skew": stats.skew(series) if len(series) > 2 else 0,
        f"{prefix}_kurtosis": stats.kurtosis(series) if len(series) > 3 else 0,
        f"{prefix}_rms": np.sqrt(np.mean(series**2)),
        f"{prefix}_diff_mean": np.mean(diff) if len(diff) > 0 else 0,
        f"{prefix}_diff_std": np.std(diff, ddof=1) if len(diff) > 1 else 0
    }

def extract_hr_features(df):
    if df is None or df.empty:
        return extract_base_features(None, 'hr')
    # Empatica E4 HR.csv usually has timestamp on first row, rate on second, then HR values.
    # Assuming standard format or single column.
    hr_series = df.iloc[:, 0] if len(df.columns) == 1 else df.iloc[2:, 0]
    hr_series = pd.to_numeric(hr_series, errors='coerce').dropna()
    return extract_base_features(hr_series, 'hr')

def extract_ibi_features(df):
    base_feats = extract_base_features(None, 'ibi')
    base_feats.update({
        'ibi_rmssd': 0, 'ibi_sdnn': 0, 'ibi_pnn50': 0, 'ibi_mean_hr_est': 0
    })

    if df is None or df.empty:
        return base_feats
    
    # E4 IBI.csv: col 0 is time from start, col 1 is IBI duration in seconds
    if len(df.columns) > 1:
        ibi_series = pd.to_numeric(df.iloc[1:, 1], errors='coerce').dropna()
    else:
        ibi_series = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()

    if len(ibi_series) == 0:
        return base_feats

    feats = extract_base_features(ibi_series, 'ibi')
    
    diff = np.diff(ibi_series)
    feats['ibi_rmssd'] = np.sqrt(np.mean(diff**2)) if len(diff) > 0 else 0
    feats['ibi_sdnn'] = np.std(ibi_series, ddof=1) if len(ibi_series) > 1 else 0
    feats['ibi_pnn50'] = (np.sum(np.abs(diff) > 0.05) / len(diff)) if len(diff) > 0 else 0
    feats['ibi_mean_hr_est'] = 60.0 / np.mean(ibi_series) if np.mean(ibi_series) > 0 else 0
    
    return feats

def extract_ecg_features(binary_content, prefix):
    base_feats = extract_base_features(None, prefix)
    base_feats.update({f'{prefix}_peak_count': 0, f'{prefix}_peak_rate_proxy': 0})
    
    if binary_content is None or len(binary_content) == 0:
        return base_feats
    
    # Mocking binary parsing for now, assuming 16-bit integers
    try:
        ecg_data = np.frombuffer(binary_content, dtype=np.int16)
    except Exception:
        return base_feats

    if len(ecg_data) == 0:
        return base_feats

    feats = extract_base_features(ecg_data, prefix)
    
    # Simple threshold proxy for peak detection
    threshold = np.mean(ecg_data) + 1.5 * np.std(ecg_data)
    peaks = np.where(ecg_data > threshold)[0]
    
    feats[f'{prefix}_peak_count'] = len(peaks)
    # proxy rate assuming arbitrary sampling rate if timestamp isn't parsed
    feats[f'{prefix}_peak_rate_proxy'] = len(peaks) / (len(ecg_data) / 256.0) if len(ecg_data) > 0 else 0
    
    return feats

def combine_all_features(hr_df, ibi_df, ecg_exp_bin, ecg_sleep_bin):
    features = {}
    features.update(extract_hr_features(hr_df))
    features.update(extract_ibi_features(ibi_df))
    features.update(extract_ecg_features(ecg_exp_bin, 'ecg_exp'))
    features.update(extract_ecg_features(ecg_sleep_bin, 'ecg_sleep'))
    return features
