import os
import time
import traceback
import warnings

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

file_path = os.path.join("..", "selecao_metadatasets", "datasets_filtered.csv")

INPUT_CSV = file_path          # must contain a column named "name"
OUTPUT_CSV = "classifier_results.csv"
FAILED_CSV = "failed_datasets.csv"

N_SPLITS = 5
RANDOM_STATE = 42

# ============================================================
# CLASSIFIERS
# ============================================================

classifiers = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),

    'SVM': SVC(random_state=42),

    'KNN': KNeighborsClassifier(),

    'LogisticRegression': LogisticRegression(
        random_state=42,
        max_iter=1000
    ),

    'Perceptron': Perceptron(
        random_state=42,
        max_iter=1000
    ),

    'MLP': MLPClassifier(
        random_state=42,
        max_iter=1000
    )
}

# ============================================================
# LOAD COMPLETED DATASETS
# ============================================================

completed_datasets = set()

if os.path.exists(OUTPUT_CSV):

    try:
        existing_df = pd.read_csv(OUTPUT_CSV)

        if "dataset" in existing_df.columns:

            completed_datasets = set(
                existing_df["dataset"].astype(str)
            )

        print(f"Already completed: {len(completed_datasets)}")

    except Exception as e:
        print(f"Could not read output CSV: {e}")

# ============================================================
# LOAD DATASET LIST
# ============================================================

datasets_df = pd.read_csv(INPUT_CSV)

dataset_names = datasets_df["name"].astype(str).tolist()

# remove already processed datasets
dataset_names = [
    d for d in dataset_names
    if d not in completed_datasets
]

print(f"Remaining datasets: {len(dataset_names)}")

# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(X, y):

    # dataframe
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # categorical -> string
    for col in X.columns:

        if pd.api.types.is_categorical_dtype(X[col]):
            X[col] = X[col].astype(str)

    # encode categorical columns
    obj_cols = X.select_dtypes(include=["object"]).columns

    for col in obj_cols:

        X[col] = X[col].fillna("missing")

        le = LabelEncoder()

        X[col] = le.fit_transform(
            X[col].astype(str)
        )

    # numeric cleanup
    X = X.replace([np.inf, -np.inf], np.nan)

    # encode target
    y = np.array(y)

    if y.dtype == object or str(y.dtype).startswith("category"):

        le = LabelEncoder()

        y = le.fit_transform(
            y.astype(str)
        )

    return X, y

# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(dataset_name):

    start_time = time.time()

    print(f"\n[PROCESSING] {dataset_name}")

    try:

        # ----------------------------------------------------
        # LOAD DATASET
        # ----------------------------------------------------

        dataset = fetch_openml(
            name=dataset_name,
            version=1,
            as_frame=True
        )

        X = dataset.data
        y = dataset.target

        if y is None:
            raise ValueError("Dataset has no target")

        # classification only
        unique_classes = np.unique(y)

        if len(unique_classes) < 2:
            raise ValueError("Not a classification dataset")

        X, y = prepare_data(X, y)

        # ----------------------------------------------------
        # CROSS VALIDATION
        # ----------------------------------------------------

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        result_row = {
            "dataset": dataset_name
        }

        for clf_name, clf in classifiers.items():

            print(f"  -> {clf_name}")

            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", clf)
            ])

            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring="accuracy",
                n_jobs=1
            )

            result_row[clf_name] = scores.mean()

        result_row["time_seconds"] = round(
            time.time() - start_time,
            2
        )

        # ----------------------------------------------------
        # SAVE IMMEDIATELY
        # ----------------------------------------------------

        result_df = pd.DataFrame([result_row])

        result_df.to_csv(
            OUTPUT_CSV,
            mode="a",
            header=not os.path.exists(OUTPUT_CSV),
            index=False
        )

        print(f"[DONE] {dataset_name}")

    except Exception as e:

        error_text = traceback.format_exc()

        fail_df = pd.DataFrame([{
            "dataset": dataset_name,
            "error": error_text
        }])

        fail_df.to_csv(
            FAILED_CSV,
            mode="a",
            header=not os.path.exists(FAILED_CSV),
            index=False
        )

        print(f"[FAILED] {dataset_name}")
        print(str(e))

# ============================================================
# MAIN LOOP
# ============================================================

for dataset_name in dataset_names:

    process_dataset(dataset_name)

print("\nALL DONE")