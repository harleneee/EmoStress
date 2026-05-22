import argparse
import io
import json
import re
import warnings
import zipfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

try:
    import h5py
except Exception:
    h5py = None

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# EmoStress Emotion Training Script
# Dataset: ECG and GSR Data for Emotion Recognition
# Scope: ECG extracted features + GSR extracted features
# Target: emotion label from Self-Annotation Labels
#
# Recommended first run:
# python train_emotion_ecg_gsr.py --data_dir "." --inspect_only
#
# Training run:
# python train_emotion_ecg_gsr.py --data_dir "." --emotion_set six --split group --model_type extra_trees --yes
# ============================================================


TEXT_RATING_MAP = {
    "verylow": 1,
    "very low": 1,
    "low": 2,
    "moderate": 3,
    "medium": 3,
    "high": 4,
    "veryhigh": 5,
    "very high": 5,
}

EMOTION_COLUMNS = [
    "happy",
    "sad",
    "fear",
    "anger",
    "neutral",
    "disgust",
    "surprised",
]

SIX_EMOTIONS = ["happy", "sad", "fear", "anger", "neutral", "disgust"]
SEVEN_EMOTIONS = ["happy", "sad", "fear", "anger", "neutral", "disgust", "surprised"]


# =========================
# General helpers
# =========================

def normalize_name(value):
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def safe_col_name(value):
    value = str(value).strip()
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value)
    value = value.strip("_")
    if value == "":
        value = "feature"
    return value.lower()


def parse_spv_from_name(filename):
    """
    Parses names like:
    FV_ECGdata_s1p1v1
    FV_GSRdata_s2p5v7.csv

    Returns:
    (session_id, participant_id, video_id)
    """
    name = Path(filename).stem.lower()
    match = re.search(r"s(\d+)p(\d+)v(\d+)", name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def looks_like_excel(path):
    try:
        with open(path, "rb") as f:
            signature = f.read(8)

        # xlsx starts with 50 4B; old xls starts with D0 CF 11 E0.
        xlsx_signature = bytes.fromhex("50 4B")
        xls_signature = bytes.fromhex("D0 CF 11 E0")

        return signature.startswith(xlsx_signature) or signature.startswith(xls_signature)

    except Exception:
        return False


def collect_numeric_arrays(obj, arrays):
    """Recursively collect numeric arrays from MATLAB objects."""
    if obj is None:
        return

    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.number):
            values = obj.astype(float).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size > 0:
                arrays.append(values)
            return

        if obj.dtype == object:
            for item in obj.flat:
                collect_numeric_arrays(item, arrays)
            return

    if hasattr(obj, "__dict__"):
        for value in obj.__dict__.values():
            collect_numeric_arrays(value, arrays)


def collect_string_values(obj, strings):
    """Recursively collect strings from MATLAB objects."""
    if obj is None:
        return

    if isinstance(obj, str):
        text = obj.strip()
        if text:
            strings.append(text)
        return

    if isinstance(obj, bytes):
        text = obj.decode("utf-8", errors="ignore").strip()
        if text:
            strings.append(text)
        return

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ["U", "S"]:
            for item in obj.reshape(-1):
                collect_string_values(item, strings)
            return

        if obj.dtype == object:
            for item in obj.flat:
                collect_string_values(item, strings)
            return

    if hasattr(obj, "__dict__"):
        for value in obj.__dict__.values():
            collect_string_values(value, strings)


def read_mat_numeric_vector(path):
    """Reads a .mat file and returns the largest numeric vector found."""
    path = Path(path)
    arrays = []

    try:
        mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            collect_numeric_arrays(value, arrays)
    except NotImplementedError:
        if h5py is None:
            raise RuntimeError("This .mat file needs h5py. Install it with: pip install h5py")

        with h5py.File(path, "r") as f:
            def visit(_, obj):
                if isinstance(obj, h5py.Dataset):
                    try:
                        data = obj[()]
                        collect_numeric_arrays(np.asarray(data), arrays)
                    except Exception:
                        pass
            f.visititems(visit)

    if not arrays:
        raise ValueError(f"No numeric arrays found inside MATLAB file: {path}")

    arrays = sorted(arrays, key=lambda x: x.size, reverse=True)
    return arrays[0].astype(float).reshape(-1)


def read_mat_string_list(path):
    path = Path(path)
    strings = []

    try:
        mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            collect_string_values(value, strings)
    except Exception:
        return []

    cleaned = []
    for value in strings:
        value = str(value).strip()
        if value and value.lower() != "nan":
            cleaned.append(value)
    return cleaned


def read_any_table(path, header="infer"):
    """
    Robust reader for CSV, TSV, TXT, XLSX, XLS, or files without visible extensions.
    """
    path = Path(path)

    errors = []

    # Try Excel first if signature looks like Excel/Office file
    if looks_like_excel(path) or path.suffix.lower() in [".xlsx", ".xls", ".xlsm"]:
        try:
            return pd.read_excel(path, header=header)
        except Exception as e:
            errors.append(f"read_excel failed: {e}")

    # Try common delimited formats
    for sep in [None, ",", "\t", ";", " "]:
        try:
            if sep is None:
                return pd.read_csv(path, header=header, sep=None, engine="python", encoding="utf-8")
            return pd.read_csv(path, header=header, sep=sep, engine="python", encoding="utf-8")
        except Exception as e:
            errors.append(f"read_csv sep={sep} utf-8 failed: {e}")

    # Try latin1 encoding
    for sep in [None, ",", "\t", ";", " "]:
        try:
            if sep is None:
                return pd.read_csv(path, header=header, sep=None, engine="python", encoding="latin1")
            return pd.read_csv(path, header=header, sep=sep, engine="python", encoding="latin1")
        except Exception as e:
            errors.append(f"read_csv sep={sep} latin1 failed: {e}")

    raise ValueError(f"Could not read table file: {path}\n" + "\n".join(errors[-5:]))


# =========================
# Dataset path detection
# =========================

def find_extracted_features_dir(data_dir):
    data_dir = Path(data_dir)
    candidates = []

    for p in data_dir.rglob("*"):
        if p.is_dir() and "extracted" in normalize_name(p.name) and "feature" in normalize_name(p.name):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            "Could not find 'Extracted Features Supplementary Data' folder. "
            "Point --data_dir to the dataset root or to the folder that contains it."
        )

    return candidates[0]


def find_self_annotation_dir(data_dir):
    data_dir = Path(data_dir)
    candidates = []

    for p in data_dir.rglob("*"):
        if p.is_dir() and "self" in normalize_name(p.name) and "annotation" in normalize_name(p.name):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError("Could not find 'Self-Annotation Labels' folder.")

    return candidates[0]


def find_stimulus_file(data_dir):
    data_dir = Path(data_dir)

    candidates = []
    for p in data_dir.rglob("*"):
        if p.is_file() and "stimulus" in normalize_name(p.stem):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError("Could not find Stimulus_Description file.")

    return candidates[0]


def find_feature_files(extracted_dir, modality):
    """
    Finds ECG or GSR feature vector files.
    It prefers Single Modal feature files when available, because this project
    trains ECG + GSR as separate modalities before combining them.
    """
    extracted_dir = Path(extracted_dir)
    modality = modality.lower()

    if modality == "ecg":
        pattern = "fv_ecgdata"
    elif modality == "gsr":
        pattern = "fv_gsrdata"
    else:
        raise ValueError("modality must be ecg or gsr")

    all_files = []

    for p in extracted_dir.rglob("*"):
        if p.is_file() and pattern in p.stem.lower():
            if parse_spv_from_name(p.name) is not None:
                all_files.append(p)

    single_modal_files = [p for p in all_files if "single modal" in normalize_name(str(p))]

    files = single_modal_files if single_modal_files else all_files
    files = sorted(files, key=lambda p: parse_spv_from_name(p.name))
    return files


def find_feature_names_file(extracted_dir, modality):
    extracted_dir = Path(extracted_dir)
    modality = modality.lower()

    candidates = []

    for p in extracted_dir.rglob("*"):
        if not p.is_file():
            continue
        name = p.stem.lower()
        if modality == "ecg" and "ecg" in name and "feat" in name and "name" in name:
            candidates.append(p)
        if modality == "gsr" and "gsr" in name and "feat" in name and "name" in name:
            candidates.append(p)

    return candidates[0] if candidates else None


def find_annotation_file(annotation_dir, preferred="multimodal"):
    annotation_dir = Path(annotation_dir)
    preferred = preferred.lower()

    all_files = [p for p in annotation_dir.rglob("*") if p.is_file()]

    preferred_files = [p for p in all_files if preferred in p.stem.lower()]
    if preferred_files:
        return preferred_files[0]

    if all_files:
        return all_files[0]

    raise FileNotFoundError(f"No annotation files found inside {annotation_dir}")


# =========================
# Feature reading
# =========================

def read_feature_vector(path):
    """
    Reads one FV_ECGdata_* or FV_GSRdata_* file and returns numeric 1D vector.
    Supports MATLAB .mat files plus CSV/Excel-like files.
    """
    path = Path(path)

    if path.suffix.lower() == ".mat":
        values = read_mat_numeric_vector(path)
        if len(values) == 0:
            raise ValueError(f"No numeric feature values found in {path}")
        return values

    # Try header=None first because these feature files are usually numeric matrices/vectors.
    try:
        df = read_any_table(path, header=None)
    except Exception:
        df = read_any_table(path, header="infer")

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    values = numeric_df.to_numpy().astype(float).reshape(-1)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError(f"No numeric feature values found in {path}")

    return values


def read_feature_names(path, prefix):
    if path is None:
        return []

    path = Path(path)

    if path.suffix.lower() == ".mat":
        raw_names = read_mat_string_list(path)
        return [f"{prefix}_{safe_col_name(value)}" for value in raw_names]

    try:
        df = read_any_table(path, header=None)
    except Exception:
        return []

    raw = df.astype(str).to_numpy().reshape(-1)
    names = []

    for value in raw:
        value = str(value).strip()
        if value == "" or value.lower() == "nan":
            continue
        names.append(f"{prefix}_{safe_col_name(value)}")

    return names


def make_feature_dict(values, names, prefix):
    features = {}

    if names and len(names) == len(values):
        final_names = names
    else:
        final_names = [f"{prefix}_feature_{i + 1}" for i in range(len(values))]

    for name, value in zip(final_names, values):
        features[name] = float(value)

    return features


def build_feature_index(feature_files):
    index = {}
    duplicates = []

    for path in feature_files:
        key = parse_spv_from_name(path.name)
        if key is None:
            continue
        if key in index:
            duplicates.append((key, index[key], path))
        index[key] = path

    return index, duplicates


# =========================
# Label reading
# =========================

def normalize_annotation_columns(df):
    original_columns = list(df.columns)
    normalized_map = {}

    for col in original_columns:
        n = normalize_name(col)
        n = re.sub(r"\s+", " ", n)
        normalized_map[col] = n

    df = df.rename(columns=normalized_map)
    return df


def find_column(df, possible_names):
    columns = list(df.columns)
    normalized_columns = {normalize_name(c): c for c in columns}

    for name in possible_names:
        n = normalize_name(name)
        if n in normalized_columns:
            return normalized_columns[n]

    # partial fallback
    for col in columns:
        col_norm = normalize_name(col)
        for name in possible_names:
            if normalize_name(name) in col_norm:
                return col

    return None


def convert_rating(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return safe_float(value)

    text = str(value).strip()
    text_norm = normalize_name(text).replace(" ", "")

    if text_norm in TEXT_RATING_MAP:
        return float(TEXT_RATING_MAP[text_norm])

    text_norm_space = normalize_name(text)
    if text_norm_space in TEXT_RATING_MAP:
        return float(TEXT_RATING_MAP[text_norm_space])

    try:
        return float(text)
    except Exception:
        return np.nan


def derive_emotion_label(row, emotion_cols, emotion_set="six", min_top_score=2.0, min_margin=0.0):
    selected = SIX_EMOTIONS if emotion_set == "six" else SEVEN_EMOTIONS

    scores = {}
    for emotion, col in emotion_cols.items():
        scores[emotion] = convert_rating(row[col])

    # Determine the real top emotion across all available emotion columns.
    valid_scores = {k: v for k, v in scores.items() if np.isfinite(v)}

    if not valid_scores:
        return None, None, None

    max_score = max(valid_scores.values())
    top_emotions = [k for k, v in valid_scores.items() if v == max_score]

    # Skip ambiguous ties.
    if len(top_emotions) != 1:
        return None, max_score, "tie"

    top_emotion = top_emotions[0]

    # If using six classes, skip rows where surprised is the strongest label.
    if top_emotion not in selected:
        return None, max_score, "excluded_emotion"

    if max_score < min_top_score:
        return None, max_score, "low_confidence"

    # Margin rule: require top to be higher than second best.
    other_scores = [v for k, v in valid_scores.items() if k != top_emotion]
    second_score = max(other_scores) if other_scores else -np.inf

    if min_margin > 0 and (max_score - second_score) < min_margin:
        return None, max_score, "low_margin"

    return top_emotion, max_score, "ok"


def load_labels(annotation_file, emotion_set="six", min_top_score=2.0, min_margin=0.0):
    df = read_any_table(annotation_file, header=0)
    df = normalize_annotation_columns(df)

    participant_col = find_column(df, ["participant id", "participant", "participantid", "participant no"])
    session_col = find_column(df, ["session id", "session", "sessionid"])
    video_col = find_column(df, ["video id", "video", "videoid", "stimulus id"])

    if participant_col is None or session_col is None or video_col is None:
        raise ValueError(
            "Could not find Participant ID, Session ID, or Video ID columns in annotation file."
            f"Detected columns: {list(df.columns)}"
        )

    emotion_cols = {}
    for emotion in EMOTION_COLUMNS:
        col = find_column(df, [emotion, emotion.capitalize()])
        if col is not None:
            emotion_cols[emotion] = col

    required = set(SEVEN_EMOTIONS if emotion_set == "seven" else SIX_EMOTIONS)
    available = set(emotion_cols.keys())

    missing = required - available
    if missing:
        raise ValueError(
            f"Missing required emotion columns: {missing}. Available emotion columns: {available}"
        )

    rows = []
    skip_reasons = []

    for _, row in df.iterrows():
        try:
            session_id = int(float(row[session_col]))
            participant_id = int(float(row[participant_col]))
            video_id = int(float(row[video_col]))
        except Exception:
            continue

        label, top_score, reason = derive_emotion_label(
            row=row,
            emotion_cols=emotion_cols,
            emotion_set=emotion_set,
            min_top_score=min_top_score,
            min_margin=min_margin,
        )

        if label is None:
            skip_reasons.append(reason)
            continue

        rows.append(
            {
                "session_id": session_id,
                "participant_id": participant_id,
                "video_id": video_id,
                "emotion": label,
                "top_emotion_score": top_score,
            }
        )

    label_df = pd.DataFrame(rows)

    if label_df.empty:
        raise ValueError(
            "No usable emotion labels were created. Try lowering --min_top_score or --min_margin."
        )

    print("Label derivation summary:")
    print(f"  Annotation rows read: {len(df)}")
    print(f"  Usable labeled rows: {len(label_df)}")
    if skip_reasons:
        print("  Skipped rows:")
        print(pd.Series(skip_reasons).value_counts())

    return label_df, list(df.columns), emotion_cols


def load_stimulus_labels(stimulus_file, emotion_set="six"):
    df = read_any_table(stimulus_file, header=0)
    df = normalize_annotation_columns(df)

    session_col = find_column(df, ["session id", "session", "sessionid"])
    video_col = find_column(df, ["video id", "video", "videoid"])
    target_col = find_column(df, ["target emotion", "emotion", "target", "label"])

    if session_col is None or video_col is None or target_col is None:
        raise ValueError(
            "Could not find Session ID, Video ID, or Target Emotion columns in stimulus file."
            f"Detected columns: {list(df.columns)}"
        )

    selected = SIX_EMOTIONS if emotion_set == "six" else SEVEN_EMOTIONS

    rows = []
    skipped = []

    for _, row in df.iterrows():
        try:
            session_id = int(float(row[session_col]))
            video_id = int(float(row[video_col]))
        except Exception:
            continue

        emotion = normalize_name(row[target_col]).replace(" ", "")
        if emotion == "surprise":
            emotion = "surprised"

        if emotion not in selected:
            skipped.append(emotion)
            continue

        rows.append(
            {
                "session_id": session_id,
                "video_id": video_id,
                "emotion": emotion,
            }
        )

    label_df = pd.DataFrame(rows)

    if label_df.empty:
        raise ValueError("No usable stimulus labels were created.")

    print("Stimulus label summary:")
    print(f"  Stimulus rows read: {len(df)}")
    print(f"  Usable stimulus labels: {len(label_df)}")
    if skipped:
        print("  Skipped target emotions:")
        print(pd.Series(skipped).value_counts())

    return label_df, list(df.columns), {"target_emotion": target_col}


# =========================
# Dataset building
# =========================

def build_dataset(
    data_dir,
    emotion_set="six",
    min_top_score=2.0,
    min_margin=0.0,
    annotation_preference="multimodal",
    label_source="self_annotation",
):
    data_dir = Path(data_dir)

    extracted_dir = find_extracted_features_dir(data_dir)
    annotation_dir = find_self_annotation_dir(data_dir)
    annotation_file = find_annotation_file(annotation_dir, preferred=annotation_preference)
    stimulus_file = find_stimulus_file(data_dir)

    ecg_files = find_feature_files(extracted_dir, "ecg")
    gsr_files = find_feature_files(extracted_dir, "gsr")

    if not ecg_files:
        raise FileNotFoundError("No ECG feature files found. Expected names like FV_ECGdata_s1p1v1.")

    if not gsr_files:
        raise FileNotFoundError("No GSR feature files found. Expected names like FV_GSRdata_s1p1v1.")

    ecg_featnames_file = find_feature_names_file(extracted_dir, "ecg")
    gsr_featnames_file = find_feature_names_file(extracted_dir, "gsr")

    ecg_index, ecg_duplicates = build_feature_index(ecg_files)
    gsr_index, gsr_duplicates = build_feature_index(gsr_files)

    common_keys = sorted(set(ecg_index.keys()) & set(gsr_index.keys()))

    if not common_keys:
        raise ValueError("No matching ECG/GSR feature files were found by session-participant-video key.")

    if label_source == "self_annotation":
        label_df, annotation_columns, emotion_cols = load_labels(
            annotation_file=annotation_file,
            emotion_set=emotion_set,
            min_top_score=min_top_score,
            min_margin=min_margin,
        )

        label_map = {}
        for _, row in label_df.iterrows():
            key = (int(row["session_id"]), int(row["participant_id"]), int(row["video_id"]))
            label_map[key] = row["emotion"]

    elif label_source == "stimulus":
        label_df, annotation_columns, emotion_cols = load_stimulus_labels(
            stimulus_file=stimulus_file,
            emotion_set=emotion_set,
        )

        label_map = {}
        for _, row in label_df.iterrows():
            key = (int(row["session_id"]), int(row["video_id"]))
            label_map[key] = row["emotion"]

    else:
        raise ValueError("label_source must be self_annotation or stimulus")

    ecg_names = read_feature_names(ecg_featnames_file, "ecg")
    gsr_names = read_feature_names(gsr_featnames_file, "gsr")

    rows = []
    skipped_no_label = 0
    skipped_bad_feature = 0

    first_ecg_len = None
    first_gsr_len = None

    for key in common_keys:
        label_key = key if label_source == "self_annotation" else (key[0], key[2])

        if label_key not in label_map:
            skipped_no_label += 1
            continue

        try:
            ecg_values = read_feature_vector(ecg_index[key])
            gsr_values = read_feature_vector(gsr_index[key])

            if first_ecg_len is None:
                first_ecg_len = len(ecg_values)
            if first_gsr_len is None:
                first_gsr_len = len(gsr_values)

            features = {}
            features.update(make_feature_dict(ecg_values, ecg_names, "ecg"))
            features.update(make_feature_dict(gsr_values, gsr_names, "gsr"))

            row = {
                "session_id": key[0],
                "participant_id": key[1],
                "video_id": key[2],
                "emotion": label_map[label_key],
            }
            row.update(features)
            rows.append(row)

        except Exception as e:
            skipped_bad_feature += 1
            print(f"  Skipped {key} due to feature read error: {e}")

    dataset_df = pd.DataFrame(rows)

    info = {
        "data_dir": str(data_dir),
        "extracted_dir": str(extracted_dir),
        "annotation_dir": str(annotation_dir),
        "annotation_file": str(annotation_file),
        "stimulus_file": str(stimulus_file),
        "label_source": label_source,
        "ecg_feature_files": len(ecg_files),
        "gsr_feature_files": len(gsr_files),
        "matching_ecg_gsr_pairs": len(common_keys),
        "usable_rows": len(dataset_df),
        "skipped_no_label": skipped_no_label,
        "skipped_bad_feature": skipped_bad_feature,
        "ecg_featnames_file": str(ecg_featnames_file) if ecg_featnames_file else None,
        "gsr_featnames_file": str(gsr_featnames_file) if gsr_featnames_file else None,
        "ecg_feature_name_count": len(ecg_names),
        "gsr_feature_name_count": len(gsr_names),
        "first_ecg_feature_length": first_ecg_len,
        "first_gsr_feature_length": first_gsr_len,
        "ecg_duplicates": len(ecg_duplicates),
        "gsr_duplicates": len(gsr_duplicates),
        "annotation_columns": annotation_columns,
        "emotion_columns_detected": emotion_cols,
    }

    if dataset_df.empty:
        raise ValueError("No training rows were created after matching features and labels.")

    return dataset_df, info


# =========================
# Balancing, splitting, plots
# =========================

def oversample_training_data(X_train, y_train, random_state=42):
    train_df = X_train.copy()
    train_df["__target__"] = y_train

    counts = train_df["__target__"].value_counts()
    max_count = counts.max()

    balanced_parts = []

    for class_id, count in counts.items():
        class_df = train_df[train_df["__target__"] == class_id]

        if count < max_count:
            class_df = class_df.sample(n=max_count, replace=True, random_state=random_state)

        balanced_parts.append(class_df)

    balanced_df = pd.concat(balanced_parts, axis=0)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    y_balanced = balanced_df["__target__"].values
    X_balanced = balanced_df.drop(columns=["__target__"])

    return X_balanced, y_balanced


def make_group_split(X, y, groups, test_size, random_state):
    unique_classes = set(np.unique(y))

    for seed_offset in range(200):
        seed = random_state + seed_offset
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        train_classes = set(np.unique(y[train_idx]))
        test_classes = set(np.unique(y[test_idx]))

        if train_classes == unique_classes and test_classes == unique_classes:
            return train_idx, test_idx, seed

    print("\nWarning: Could not find a group split containing all classes in both train and test.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return train_idx, test_idx, random_state


def plot_confusion_matrix_image(cm, labels, output_path, normalize=False, title="Confusion Matrix"):
    if normalize:
        cm_to_plot = cm.astype(float)
        row_sums = cm_to_plot.sum(axis=1, keepdims=True)
        cm_to_plot = np.divide(
            cm_to_plot,
            row_sums,
            out=np.zeros_like(cm_to_plot),
            where=row_sums != 0,
        )
        fmt = ".2f"
    else:
        cm_to_plot = cm.astype(int)
        fmt = "d"

    plt.figure(figsize=(9, 7))
    plt.imshow(cm_to_plot, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            value = format(cm_to_plot[i, j], fmt)
            plt.text(j, i, value, ha="center", va="center")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feature_columns, output_path, top_n=25):
    if not hasattr(model, "feature_importances_"):
        print("Model does not support feature_importances_. Skipping feature importance plot.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_columns[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(11, 8))
    plt.barh(range(len(top_features)), top_importances)
    plt.yticks(range(len(top_features)), top_features)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Emotion Model Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_model(model_type, random_state):
    if model_type == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError("model_type must be extra_trees or random_forest")


# =========================
# Inspection and training
# =========================

def inspect_dataset(dataset_df, info):
    print("\n" + "=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)

    print("\nDetected paths:")
    print(f"  Extracted features folder: {info['extracted_dir']}")
    print(f"  Annotation folder: {info['annotation_dir']}")
    print(f"  Annotation file: {info['annotation_file']}")
    print(f"  ECG feature names file: {info['ecg_featnames_file']}")
    print(f"  GSR feature names file: {info['gsr_featnames_file']}")

    print("\nFeature files detected:")
    print(f"  ECG feature files: {info['ecg_feature_files']}")
    print(f"  GSR feature files: {info['gsr_feature_files']}")
    print(f"  Matching ECG/GSR pairs: {info['matching_ecg_gsr_pairs']}")
    print(f"  Usable rows after label matching: {info['usable_rows']}")
    print(f"  Skipped no label: {info['skipped_no_label']}")
    print(f"  Skipped bad feature: {info['skipped_bad_feature']}")

    print("\nFeature dimensions:")
    print(f"  ECG feature vector length: {info['first_ecg_feature_length']}")
    print(f"  GSR feature vector length: {info['first_gsr_feature_length']}")
    print(f"  ECG feature name count: {info['ecg_feature_name_count']}")
    print(f"  GSR feature name count: {info['gsr_feature_name_count']}")

    print("\nEmotion distribution:")
    print(dataset_df["emotion"].value_counts())

    print("\nParticipant distribution:")
    print(dataset_df["participant_id"].value_counts().sort_index())

    print("\nFirst 5 rows:")
    preview_cols = ["session_id", "participant_id", "video_id", "emotion"]
    print(dataset_df[preview_cols].head())

    print("\nTotal feature count:")
    feature_cols = [c for c in dataset_df.columns if c not in ["session_id", "participant_id", "video_id", "emotion"]]
    print(f"  {len(feature_cols)} features")

    print("=" * 70)


def train_emotion_model(
    data_dir,
    model_dir,
    eval_dir,
    emotion_set="six",
    min_top_score=2.0,
    min_margin=0.0,
    annotation_preference="multimodal",
    label_source="self_annotation",
    model_type="extra_trees",
    split="group",
    balance_train="oversample",
    test_size=0.20,
    random_state=42,
    inspect_only=False,
    yes=False,
):
    model_dir = Path(model_dir)
    eval_dir = Path(eval_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("\nBuilding ECG+GSR emotion dataset...")
    dataset_df, info = build_dataset(
        data_dir=data_dir,
        emotion_set=emotion_set,
        min_top_score=min_top_score,
        min_margin=min_margin,
        annotation_preference=annotation_preference,
        label_source=label_source,
    )

    inspect_dataset(dataset_df, info)

    inspection_path = eval_dir / "emotion_dataset_inspection.json"
    with open(inspection_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, default=str)

    dataset_df.to_csv(eval_dir / "emotion_ecg_gsr_extracted_features.csv", index=False)
    print(f"\nSaved inspection metadata to: {inspection_path}")
    print(f"Saved extracted training table to: {eval_dir / 'emotion_ecg_gsr_extracted_features.csv'}")

    if inspect_only:
        print("\nInspect-only mode enabled. Training stopped before model fitting.")
        print("If the inspection looks correct, rerun the command without --inspect_only and add --yes.")
        return None

    if not yes:
        answer = input("\nProceed with training? Type YES to continue: ").strip()
        if answer != "YES":
            print("Training cancelled.")
            return None

    metadata_cols = ["session_id", "participant_id", "video_id", "emotion"]
    feature_cols = [c for c in dataset_df.columns if c not in metadata_cols]

    X = dataset_df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")

    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        print(f"\nDropping {len(all_nan_cols)} all-NaN feature columns.")
        X = X.drop(columns=all_nan_cols)

    feature_cols = list(X.columns)

    y_text = dataset_df["emotion"].astype(str).values
    groups = dataset_df["participant_id"].astype(str).values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    print("\nEncoded emotion labels:")
    for class_id, class_name in enumerate(label_encoder.classes_):
        print(f"  {class_id}: {class_name}")

    indices = np.arange(len(X))

    if split == "random":
        print("\nUsing RANDOM split.")
        print("Warning: random split can overestimate performance.")
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        actual_random_state = random_state

    elif split == "group":
        print("\nUsing PARTICIPANT-BASED group split.")
        print("This is more honest because test participants are unseen during training.")
        train_idx, test_idx, actual_random_state = make_group_split(
            X=X,
            y=y,
            groups=groups,
            test_size=test_size,
            random_state=random_state,
        )

    else:
        raise ValueError("split must be group or random")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"\nTrain rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    print("\nTrain class distribution before balancing:")
    print(pd.Series(label_encoder.inverse_transform(y_train)).value_counts())

    print("\nTest class distribution:")
    print(pd.Series(label_encoder.inverse_transform(y_test)).value_counts())

    if balance_train == "oversample":
        X_train, y_train = oversample_training_data(X_train, y_train, random_state=random_state)
        print("\nTrain class distribution after oversampling:")
        print(pd.Series(label_encoder.inverse_transform(y_train)).value_counts())
    elif balance_train == "none":
        print("\nNo oversampling applied.")
    else:
        raise ValueError("balance_train must be oversample or none")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", create_model(model_type=model_type, random_state=random_state)),
        ]
    )

    print(f"\nTraining {model_type} emotion model using ECG+GSR extracted features...")
    pipeline.fit(X_train, y_train)

    print("\nEvaluating emotion model...")
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    class_names = list(label_encoder.classes_)

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(report)
    print(f"Accuracy: {accuracy:.4f}")

    report_path = eval_dir / "emotion_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\nAccuracy: {accuracy:.4f}\n")
        f.write("\nTraining Settings:\n")
        f.write("Dataset: ECG and GSR Data for Emotion Recognition\n")
        f.write("Feature scope: ECG extracted features + GSR extracted features\n")
        if label_source == "stimulus":
            f.write("Label source: Stimulus_Description target emotion: {label}\n")
        else:
            f.write("Label source: Self-Annotation Labels")
        f.write(f"Emotion set: {emotion_set}\n")
        f.write(f"Minimum top emotion score: {min_top_score}\n")
        f.write(f"Minimum margin: {min_margin}\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Balance train: {balance_train}\n")
        f.write(f"Test size: {test_size}\n")
        f.write(f"Random state: {actual_random_state}\n")
        f.write(f"Feature count: {len(feature_cols)}\n")

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))

    plot_confusion_matrix_image(
        cm=cm,
        labels=class_names,
        output_path=eval_dir / "emotion_confusion_matrix.png",
        normalize=False,
        title="Emotion Confusion Matrix",
    )

    plot_confusion_matrix_image(
        cm=cm,
        labels=class_names,
        output_path=eval_dir / "emotion_confusion_matrix_normalized.png",
        normalize=True,
        title="Emotion Confusion Matrix Normalized",
    )

    trained_model = pipeline.named_steps["model"]
    plot_feature_importance(
        model=trained_model,
        feature_columns=feature_cols,
        output_path=eval_dir / "emotion_feature_importance.png",
        top_n=25,
    )

    joblib.dump(pipeline, model_dir / "emotion_random_forest.joblib")
    joblib.dump(label_encoder, model_dir / "emotion_label_encoder.joblib")
    joblib.dump(feature_cols, model_dir / "emotion_feature_columns.joblib")

    metadata = {
        "model_name": model_type,
        "dataset": "ECG and GSR Data for Emotion Recognition",
        "task": "emotion_classification",
        "feature_scope": "ECG extracted features + GSR extracted features",
        "label_source": "Stimulus_Description target emotion" if label_source == "stimulus" else "Self-Annotation Labels",
        "emotion_set": emotion_set,
        "classes": class_names,
        "accuracy": float(accuracy),
        "min_top_score": min_top_score,
        "min_margin": min_margin,
        "split": split,
        "balance_train": balance_train,
        "test_size": test_size,
        "random_state": actual_random_state,
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "inspection": info,
    }

    with open(model_dir / "emotion_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, default=str)

    predictions_df = pd.DataFrame(
        {
            "session_id": dataset_df.iloc[test_idx]["session_id"].values,
            "participant_id": dataset_df.iloc[test_idx]["participant_id"].values,
            "video_id": dataset_df.iloc[test_idx]["video_id"].values,
            "true_label": label_encoder.inverse_transform(y_test),
            "predicted_label": label_encoder.inverse_transform(y_pred),
        }
    )

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_test)
        for i, class_name in enumerate(class_names):
            predictions_df[f"probability_{class_name}"] = probs[:, i]

    predictions_df.to_csv(eval_dir / "emotion_test_predictions.csv", index=False)

    print("\nTraining complete.")
    print(f"Saved model to: {model_dir / 'emotion_random_forest.joblib'}")
    print(f"Saved label encoder to: {model_dir / 'emotion_label_encoder.joblib'}")
    print(f"Saved feature columns to: {model_dir / 'emotion_feature_columns.joblib'}")
    print(f"Saved metadata to: {model_dir / 'emotion_model_metadata.json'}")
    print(f"Saved report to: {report_path}")
    print(f"Saved confusion matrix to: {eval_dir / 'emotion_confusion_matrix.png'}")
    print(f"Saved normalized confusion matrix to: {eval_dir / 'emotion_confusion_matrix_normalized.png'}")
    print(f"Saved feature importance to: {eval_dir / 'emotion_feature_importance.png'}")
    print(f"Saved test predictions to: {eval_dir / 'emotion_test_predictions.csv'}")

    return pipeline, label_encoder, feature_cols, accuracy


# =========================
# Main
# =========================

def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Train emotion model using ECG and GSR extracted features."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Dataset root folder. Can be '.' if the extracted dataset folders are in the current folder.",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(script_dir / "models"),
        help="Directory where trained models will be saved.",
    )

    parser.add_argument(
        "--eval_dir",
        type=str,
        default=str(script_dir / "evaluation"),
        help="Directory where evaluation files will be saved.",
    )

    parser.add_argument(
        "--emotion_set",
        type=str,
        default="six",
        choices=["six", "seven"],
        help="six excludes surprised; seven includes surprised.",
    )

    parser.add_argument(
        "--min_top_score",
        type=float,
        default=2.0,
        help="Minimum top emotion rating required. Text ratings map: VeryLow=1, Low=2, Moderate=3, High=4, VeryHigh=5.",
    )

    parser.add_argument(
        "--min_margin",
        type=float,
        default=0.0,
        help="Minimum difference between top emotion and second emotion. Use 1.0 for cleaner labels.",
    )

    parser.add_argument(
        "--annotation_preference",
        type=str,
        default="multimodal",
        choices=["multimodal", "single"],
        help="Which self-annotation file to prefer when --label_source self_annotation is used.",
    )

    parser.add_argument(
        "--label_source",
        type=str,
        default="self_annotation",
        choices=["self_annotation", "stimulus"],
        help="self_annotation uses participant ratings; stimulus uses Stimulus_Description target emotion.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="extra_trees",
        choices=["extra_trees", "random_forest"],
    )

    parser.add_argument(
        "--split",
        type=str,
        default="group",
        choices=["group", "random"],
        help="group = participant-based split; random = ordinary random split.",
    )

    parser.add_argument(
        "--balance_train",
        type=str,
        default="oversample",
        choices=["oversample", "none"],
    )

    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--inspect_only", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt and train immediately.")

    args = parser.parse_args()

    train_emotion_model(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        eval_dir=args.eval_dir,
        emotion_set=args.emotion_set,
        min_top_score=args.min_top_score,
        min_margin=args.min_margin,
        annotation_preference=args.annotation_preference,
        label_source=args.label_source,
        model_type=args.model_type,
        split=args.split,
        balance_train=args.balance_train,
        test_size=args.test_size,
        random_state=args.random_state,
        inspect_only=args.inspect_only,
        yes=args.yes,
    )


if __name__ == "__main__":
    main()
