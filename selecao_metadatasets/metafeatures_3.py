import os
import time
import traceback
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from pymfe.mfe import MFE
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

OUTPUT_FILE = "meta_features.csv"
FAILED_FILE = "failed_datasets.csv"
FAILED_MF_FILE = "failed_metafeatures.csv"

N_WORKERS = min(8, cpu_count())

# ============================================================
# SUMMARIES
# ============================================================

SUMMARYS = [
    "mean",
    "sd",
    "median",
    "count",
    "histogram",
    "iq_range",
    "kurtosis",
    "max",
    "min",
    "quantiles",
    "range",
    "skewness"
]

# ============================================================
# LOAD ALREADY PROCESSED DATASETS
# ============================================================

processed_datasets = set()

if os.path.exists(OUTPUT_FILE):
    try:
        existing_df = pd.read_csv(OUTPUT_FILE)

        if "dataset" in existing_df.columns:
            processed_datasets = set(
                existing_df["dataset"].astype(str)
            )

        print(f"Already processed: {len(processed_datasets)}")

    except Exception as e:
        print(f"Could not read existing file: {e}")

# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(X, y):

    if y is None:
        raise ValueError("Target y is None")

    # convert y
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    # dataframe
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # categorical -> string
    for col in X.columns:

        if pd.api.types.is_categorical_dtype(X[col]):
            X[col] = X[col].astype(str)

    # object columns
    obj_cols = X.select_dtypes(include=["object"]).columns

    for col in obj_cols:

        X[col] = X[col].fillna("missing")

        le = LabelEncoder()

        X[col] = le.fit_transform(X[col].astype(str))

    # categorical columns
    cat_cols = X.select_dtypes(include=["category"]).columns

    for col in cat_cols:

        X[col] = X[col].astype(str)

        le = LabelEncoder()

        X[col] = le.fit_transform(X[col])

    # numeric columns
    num_cols = X.select_dtypes(include=[np.number]).columns

    X[num_cols] = X[num_cols].replace(
        [np.inf, -np.inf],
        np.nan
    )

    X[num_cols] = X[num_cols].fillna(0)

    # y encoding
    if y.dtype == object or str(y.dtype).startswith("category"):

        y = y.astype(str)

        le = LabelEncoder()

        y = le.fit_transform(y)

    return X.values.astype(np.float32), np.array(y)

# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(dataset_name):

    if dataset_name in processed_datasets:
        print(f"[SKIPPED] {dataset_name}")
        return None

    start = time.time()

    try:

        print(f"[PROCESSING] {dataset_name}")

        dataset = fetch_openml(
            name=dataset_name,
            version=1,
            as_frame=True
        )

        X = dataset.data
        y = dataset.target

        # skip regression datasets
        if y is None:
            raise ValueError("Dataset has no target")

        # classification only
        unique_classes = np.unique(y)

        if len(unique_classes) < 2:
            raise ValueError("Not a classification dataset")

        X, y = prepare_data(X, y)

        mfe = MFE(
            groups="all",
            summary=SUMMARYS
        )

        mfe.fit(X, y)

        ft_names, ft_values = mfe.extract()

        result = {
            name: value
            for name, value in zip(ft_names, ft_values)
        }

        result["dataset"] = dataset_name
        result["time_seconds"] = round(
            time.time() - start,
            2
        )

        result_df = pd.DataFrame([result])

        # append incrementally
        result_df.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=not os.path.exists(OUTPUT_FILE),
            index=False
        )

        print(f"[OK] {dataset_name}")

        return result

    except Exception as e:

        error_text = traceback.format_exc()

        fail_df = pd.DataFrame([{
            "dataset": dataset_name,
            "error": error_text
        }])

        fail_df.to_csv(
            FAILED_FILE,
            mode="a",
            header=not os.path.exists(FAILED_FILE),
            index=False
        )

        print(f"[FAILED] {dataset_name}")

        return None

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    datasets = openml.datasets.list_datasets(
        output_format="dataframe"
    )

    dataset_names = datasets["name"].unique().tolist()

    # remove already processed
    dataset_names = [
        d for d in dataset_names
        if d not in processed_datasets
    ]

    print(f"Found {len(dataset_names)} remaining datasets")
    print(f"Using {N_WORKERS} workers")

    with Pool(N_WORKERS) as pool:

        results = pool.map(
            process_dataset,
            dataset_names
        )

    successful = sum(r is not None for r in results)
    failed = sum(r is None for r in results)

    print("\nDONE")
    print(f"Successful datasets: {successful}")
    print(f"Failed datasets: {failed}")