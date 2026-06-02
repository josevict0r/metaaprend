

def evaluate_feature_subset(feature_subset):
    X = meta_dataset[feature_subset]
    y = meta_dataset['Best'].values
    loo = LeaveOneOut()
    y_pred = []
    for train_idx, test_idx in loo.split(X):
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X.iloc[train_idx], y[train_idx])
        y_pred.append(clf.predict(X.iloc[test_idx])[0])
    return f1_score(y, y_pred, average='weighted')

