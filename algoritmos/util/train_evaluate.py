import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, accuracy_score
from sklearn.impute import KNNImputer

meta_dataset = pd.read_csv('..\meta_dataset_bestcol.csv')

classifier_cols = [c for c in meta_dataset.columns if c in ['DecisionTree', 'SVM', 'KNN',
                                                           'LogisticRegression', 'Perceptron', 'MLP']]
meta_feature_cols = [c for c in meta_dataset.columns if c not in ['dataset', 'Best'] + classifier_cols]


def train_and_evaluate_meta_model(meta_dataset):
    # Create a dictionary to store the reuslts:
    summary_of_predictions = {'Dataset':[], 'Best clf (true)':[], 'Perf of best clf (true)':[],
                           'Best clf (pred)':[], 'Perf of best clf (pred)':[]}
    loo = LeaveOneOut()
    feat_import_perf_fold = []
    y_true = meta_dataset['Best'].values
    y_pred = []
    feat_import_per_fold = []

    for train_index, test_index in loo.split(meta_dataset):
        # Split the data into training and test sets
        X = meta_dataset.drop(columns=['dataset', 'Best'] + classifier_cols) # Drop everything except meta-features
        y = meta_dataset['Best']
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # --- Handle infinite and missing values in X_train and X_test ---
        # Replace infinity with NaN
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)

        # Identify columns that are all NaN in X_train after replacement
        all_nan_cols = X_train.columns[X_train.isnull().all()].tolist()

        # Drop these columns from X_train and X_test
        X_train_filtered = X_train.drop(columns=all_nan_cols)
        X_test_filtered = X_test.drop(columns=all_nan_cols) # Apply same filtering to test set

        # Impute missing values using KNNImputer on the filtered data
        imputer = KNNImputer(n_neighbors=3)
        X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_filtered), columns=X_train_filtered.columns, index=X_train_filtered.index)
        X_test_imputed = pd.DataFrame(imputer.transform(X_test_filtered), columns=X_test_filtered.columns, index=X_test_filtered.index)
        # --- End handling ---

        # Train a simple classifier (e.g., Decision Tree) on the training set
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train_imputed, y_train) # Use imputed data
        feat_import_per_fold.append(clf.feature_importances_)   # Get feature importances for this fold

        # Predict the best classifier for the test dataset
        y_pred.append(clf.predict(X_test_imputed)[0]) # Use imputed data

        # Store results in the summary dictionary
        summary_of_predictions['Dataset'].append(meta_dataset['dataset'].iloc[test_index].values[0])
        summary_of_predictions['Best clf (true)'].append(y_test.values[0])
        summary_of_predictions['Perf of best clf (true)'].append(meta_dataset.loc[test_index, y_test.values[0]].values[0])
        summary_of_predictions['Best clf (pred)'].append(y_pred[-1])
        summary_of_predictions['Perf of best clf (pred)'].append(meta_dataset.loc[test_index, y_pred[-1]].values[0])


    # Create a DataFrame from the summary of predictions
    summary_df = pd.DataFrame(summary_of_predictions)

    # Calculate meta-model accuracy and F1-score
    meta_model_accuracy = accuracy_score(y_true, y_pred)
    meta_model_f1 = f1_score(y_true, y_pred, average='weighted')
    return summary_df, meta_model_accuracy, meta_model_f1, feat_import_per_fold

summary_df, meta_model_accuracy, meta_model_f1, feature_importances = train_and_evaluate_meta_model(meta_dataset)
print(f'Meta-model Accuracy: {meta_model_accuracy:.2f}')
print(f'Meta-model F1-score: {meta_model_f1:.2f}')