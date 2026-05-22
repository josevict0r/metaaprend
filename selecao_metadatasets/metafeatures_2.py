import time
import traceback
import warnings
import multiprocessing as mp

import os
import numpy as np
import pandas as pd

from pymfe.mfe import MFE
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
    TimeoutError
)

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================

DATASET_CSV = "datasets_filtered_cleaned.csv"

OUTPUT_METAFEATURES = "meta_features.csv"
FAILED_DATASETS_LOG = "failed_datasets.csv"
FAILED_METAFEATURES_LOG = "failed_metafeatures.csv"

TIMEOUT_SECONDS = 1500
MAX_WORKERS = 8


# datasets known to fail on openml
BAD_DATASETS = {
    "CostaMadre1",
    "iriiiiiis",
    "jEdit_4.2_4.3",
    "jEdit_4.0_4.2"
}


# =========================================================
# DATA PREP
# =========================================================


def prepare_data(X, y):

    if y is None:
        raise ValueError("Target y is None")

    X = pd.DataFrame(X).copy()

    # categorical target
    y = pd.Series(y)

    if y.dtype == object or str(y.dtype) == "category":

        y = y.astype(str)

        y = LabelEncoder().fit_transform(y)

    # categorical features
    for col in X.columns:

        if (
            X[col].dtype == object
            or str(X[col].dtype) == "category"
        ):

            X[col] = (
                X[col]
                .astype(str)
                .fillna("missing")
            )

            X[col] = LabelEncoder().fit_transform(
                X[col]
            )

        else:

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            )

            X[col] = X[col].replace(
                [np.inf, -np.inf],
                np.nan
            )

            X[col] = X[col].fillna(0)

    X = X.astype(np.float32)

    y = np.asarray(y)

    return X, y


# =========================================================
# SINGLE FEATURE EXTRACTION
# =========================================================


def try_extract_feature(group_name, feature_name, X, y):

    try:

        mfe = MFE(
            groups=[group_name],
            features=[feature_name],
            summary=[
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
        )

        mfe.fit(X.values, y)

        names, values = mfe.extract()

        processed = {}

        for name, value in zip(names, values):

            # scalar values
            if np.isscalar(value):

                processed[name] = value

            # vector outputs
            elif isinstance(
                value,
                (list, tuple, np.ndarray)
            ):

                arr = np.asarray(value).ravel()

                for i, v in enumerate(arr):

                    processed[f"{name}.{i}"] = v

            else:

                processed[name] = str(value)

        return processed, None

    except Exception as e:

        return None, {
            "group": group_name,
            "feature": feature_name,
            "error": str(e)
        }

# =========================================================
# DATASET PROCESSING
# =========================================================


def process_dataset(dataset_name):

    start_total = time.time()

    failed_features = []

    try:

        if dataset_name in BAD_DATASETS:

            return {
                "success": False,
                "dataset_name": dataset_name,
                "error": "Known bad dataset",
                "failed_features": []
            }

        # =============================================
        # DOWNLOAD DATASET
        # =============================================

        dataset = fetch_openml(
            name=dataset_name,
            version=1,
            as_frame=True
        )

        X = dataset.data
        y = dataset.target

        X, y = prepare_data(X, y)

        # =============================================
        # GET ALL GROUPS
        # =============================================

        mfe_all = MFE(groups="all")

        all_groups = mfe_all.valid_groups()

        dataset_features = {
            "dataset": dataset_name
        }

        # =============================================
        # EXTRACT FEATURE BY FEATURE
        # =============================================

        for group in all_groups:

            try:

                feature_names = MFE(
                    groups=[group]
                ).valid_metafeatures(
                    groups=[group]
                )

            except Exception as e:

                failed_features.append({
                    "dataset": dataset_name,
                    "group": group,
                    "feature": "GROUP_INIT",
                    "error": str(e)
                })

                continue

            for feature in feature_names:

                result, error = try_extract_feature(
                    group,
                    feature,
                    X,
                    y
                )

                if error is not None:

                    error["dataset"] = dataset_name

                    failed_features.append(error)

                else:

                    dataset_features.update(result)

        dataset_features["time_seconds"] = (
            time.time() - start_total
        )

        return {
            "success": True,
            "dataset": dataset_features,
            "failed_features": failed_features
        }

    except Exception:

        return {
            "success": False,
            "dataset_name": dataset_name,
            "error": traceback.format_exc(),
            "failed_features": failed_features
        }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    dataset_df = pd.read_csv(DATASET_CSV)

    dataset_names = (
        dataset_df.iloc[:, 0]
        .dropna()
        .astype(str)
        .tolist()
    )

    print(f"Found {len(dataset_names)} datasets")
    print(f"Using {MAX_WORKERS} workers")

    all_results = []
    failed_datasets = []
    all_failed_features = []

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        # =====================================================
        # LOAD PREVIOUS PROGRESS
        # =====================================================

        already_processed = set()

        if os.path.exists(OUTPUT_METAFEATURES):

            previous_df = pd.read_csv(OUTPUT_METAFEATURES)

            if "dataset" in previous_df.columns:

                already_processed = set(
                    previous_df["dataset"].astype(str)
                )

                print(
                    f"Skipping {len(already_processed)} "
                    f"already processed datasets"
                )

        dataset_names = [
            d for d in dataset_names
            if d not in already_processed
        ]

        future_map = {
            executor.submit(
                process_dataset,
                dataset_name
            ): dataset_name
            for dataset_name in dataset_names
        }

        for future in as_completed(future_map):

            dataset_name = future_map[future]

            try:

                result = future.result(
                    timeout=TIMEOUT_SECONDS
                )

                if result["success"]:

                    all_results.append(result["dataset"])

                    # =====================================================
                    # SAVE PROGRESS IMMEDIATELY
                    # =====================================================

                    pd.DataFrame(all_results).to_csv(
                        OUTPUT_METAFEATURES,
                        index=False
                    )

                    pd.DataFrame(failed_datasets).to_csv(
                        FAILED_DATASETS_LOG,
                        index=False
                    )

                    all_failed_features.extend(
                        result["failed_features"]
                    )

                    pd.DataFrame(all_failed_features).to_csv(
                        FAILED_METAFEATURES_LOG,
                        index=False
                    )

                    print(f"[OK] {dataset_name}")

                else:

                    failed_datasets.append({
                        "dataset": result["dataset_name"],
                        "error": result["error"]
                    })

                    print(f"[FAILED] {dataset_name}")

            except TimeoutError:

                failed_datasets.append({
                    "dataset": dataset_name,
                    "error": f"Timeout > {TIMEOUT_SECONDS}s"
                })

                print(f"[TIMEOUT] {dataset_name}")

            except Exception as e:

                failed_datasets.append({
                    "dataset": dataset_name,
                    "error": str(e)
                })

                print(f"[CRASH] {dataset_name}")

    # =====================================================
    # SAVE OUTPUTS
    # =====================================================

    if len(all_results) > 0:

        pd.DataFrame(all_results).to_csv(
            OUTPUT_METAFEATURES,
            index=False
        )

    if len(failed_datasets) > 0:

        pd.DataFrame(failed_datasets).to_csv(
            FAILED_DATASETS_LOG,
            index=False
        )

    if len(all_failed_features) > 0:

        pd.DataFrame(all_failed_features).to_csv(
            FAILED_METAFEATURES_LOG,
            index=False
        )

    print("\nDONE")
    print(f"Successful datasets: {len(all_results)}")
    print(f"Failed datasets: {len(failed_datasets)}")
    print(f"Failed meta-features: {len(all_failed_features)}")