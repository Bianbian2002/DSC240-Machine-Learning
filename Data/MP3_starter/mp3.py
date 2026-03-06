# Starter code for DSC 240 MP3
import math
import numpy as np
import pandas as pd
from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

np.random.seed(0)


def _f1_from_labels(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2 * tp / denom)


def _best_threshold_from_probs(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_f1 = -1.0
    best_th = 0.5
    for th in np.linspace(0.10, 0.70, 25):
        pred = (probs >= th).astype(int)
        f1 = _f1_from_labels(y_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
    return best_th, best_f1

def compute_metric(labels, expected):
    tp = np.sum(labels[expected == 1])
    fp = np.sum(labels[expected == 0])
    tn = np.sum(1-labels[expected == 0])
    fn = np.sum(1-labels[expected == 1])
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    error_rate = (fp+fn)/(tp+fp+tn+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)

    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
    }


def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: 
        testing_data: the same as training_data with "target" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1, 2], k=len(testing_data))
    """

    training_data = training_data.copy()
    testing_data = testing_data.copy()

    if "target" not in training_data.columns:
        raise ValueError("training_data must include a 'target' column.")

    y = training_data["target"].astype(int)
    X_train = training_data.drop(columns=["target"])
    X_test = testing_data.drop(columns=["target"]) if "target" in testing_data.columns else testing_data

    binary_cols = [
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "FLAG_MOBIL",
        "FLAG_WORK_PHONE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
    ]
    categorical_cols = [
        "CNT_CHILDREN",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "CNT_FAM_MEMBERS",
        "QUANTIZED_INC",
        "QUANTIZED_AGE",
        "QUANTIZED_WORK_YEAR",
    ]
    continuous_cols = [
        "AMT_INCOME_TOTAL",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
    ]

    binary_cols = [c for c in binary_cols if c in X_train.columns]
    categorical_cols = [c for c in categorical_cols if c in X_train.columns]
    continuous_cols = [c for c in continuous_cols if c in X_train.columns]

    categorical_features = binary_cols + categorical_cols

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                continuous_cols,
            ),
        ],
        remainder="drop",
    )

    candidate_models = [
        RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=0,
            n_jobs=1,
        ),
        RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=0,
            n_jobs=1,
        ),
        ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=0,
            n_jobs=1,
        ),
    ]

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    y_np = y.values
    oof_probs_per_model: List[np.ndarray] = []

    for model in candidate_models:
        oof_probs = np.zeros(len(X_train), dtype=float)
        for train_idx, val_idx in skf.split(X_train, y_np):
            X_tr = X_train.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_va = X_train.iloc[val_idx]

            clf = Pipeline(steps=[("preprocess", preprocessor), ("model", clone(model))])
            clf.fit(X_tr, y_tr)
            oof_probs[val_idx] = clf.predict_proba(X_va)[:, 1]
        oof_probs_per_model.append(oof_probs)

    # Evaluate single models + a few robust weighted blends.
    candidates = []
    for i, probs in enumerate(oof_probs_per_model):
        candidates.append((f"model_{i}", probs, [i], [1.0]))

    # Blends are often more stable on private test data.
    blend_specs = [
        ([0, 1], [0.5, 0.5]),
        ([0, 2], [0.5, 0.5]),
        ([1, 2], [0.5, 0.5]),
        ([0, 1, 2], [0.34, 0.33, 0.33]),
        ([0, 2], [0.4, 0.6]),
        ([0, 2], [0.6, 0.4]),
    ]
    for indices, weights in blend_specs:
        if max(indices) >= len(oof_probs_per_model):
            continue
        blend = np.zeros(len(X_train), dtype=float)
        for idx, w in zip(indices, weights):
            blend += w * oof_probs_per_model[idx]
        candidates.append(("blend", blend, indices, weights))

    best_f1 = -1.0
    best_threshold = 0.5
    best_indices = [0]
    best_weights = [1.0]
    for _name, probs, indices, weights in candidates:
        th, f1 = _best_threshold_from_probs(y_np, probs)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
            best_indices = indices
            best_weights = weights

    # Refit selected model(s) on full training data and predict test probabilities.
    test_probs = np.zeros(len(X_test), dtype=float)
    for idx, w in zip(best_indices, best_weights):
        model = candidate_models[idx]
        clf = Pipeline(steps=[("preprocess", preprocessor), ("model", clone(model))])
        clf.fit(X_train, y)
        test_probs += w * clf.predict_proba(X_test)[:, 1]

    predict = (test_probs >= best_threshold).astype(int)
    return np.asarray(predict, dtype=int)


if __name__ == '__main__':

    training = pd.read_csv('./data/train.csv')
    development = pd.read_csv('./data/dev.csv')

    target_label = development['target']
    development.drop('target', axis=1, inplace=True)
    prediction = np.asarray(run_train_test(training, development))
    target_label = target_label.values
    status = compute_metric(prediction, target_label)
    print(status)

    


    


