# Starter code for DSC 240 MP2
import math
import random
import numpy as np
import pandas as pd

from typing import List, Tuple

def _standardize(
    train_x: np.ndarray, test_x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    # Standardize based on training stats only.
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def _add_bias(x: np.ndarray) -> np.ndarray:
    # Append a bias term as the last feature.
    bias = np.ones((x.shape[0], 1), dtype=x.dtype)
    return np.hstack([x, bias])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid.
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _train_logreg_sgd(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 80,
    lr: float = 0.15,
    batch_size: int = 32,
    l2: float = 1e-4,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_samples, n_features = x.shape
    w = np.zeros(n_features, dtype=float)
    for epoch in range(epochs):
        indices = rng.permutation(n_samples)
        lr_t = lr / (1.0 + 0.02 * epoch)
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            xb = x[batch_idx]
            yb = y[batch_idx]
            probs = _sigmoid(xb @ w)
            grad = (xb.T @ (probs - yb)) / len(yb) + l2 * w
            w -= lr_t * grad
    return w


def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: pd.DataFrame
        testing_data: the same as training_data with "label" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1], k=len(testing_data))
    """
    train_x = training_data.drop('target', axis=1).to_numpy(dtype=float)
    train_y = training_data['target'].to_numpy(dtype=float)
    test_x = testing_data.to_numpy(dtype=float)

    train_x, test_x = _standardize(train_x, test_x)
    train_x = _add_bias(train_x)
    test_x = _add_bias(test_x)

    weights = _train_logreg_sgd(train_x, train_y)
    test_probs = _sigmoid(test_x @ weights)
    test_pred = (test_probs >= 0.5).astype(int)
    return test_pred.tolist()

if __name__ == '__main__':
    # load data
    training = pd.read_csv('data/train.csv')
    testing = pd.read_csv('data/dev.csv')
    target_label = testing['target']
    testing.drop('target', axis=1, inplace=True)

    # run training and testing
    prediction = np.array(run_train_test(training, testing), dtype=int)

    # Example metric 1: check accuracy 
    target_label = target_label.values
    print("Dev Accuracy: ", np.sum(prediction == target_label) / len(target_label))
    
    # Metric 2: F1 score
    # Please implement F1 score metric to test your predictions. We do not evlaute your F1 score function
    # nor do you need to provide it, but you should implement it for your understading. 
    # Please note: Autograder will evaluate your predictions on hidden test data using F1 scoring.
    tp = np.sum((prediction == 1) & (target_label == 1))
    fp = np.sum((prediction == 1) & (target_label == 0))
    fn = np.sum((prediction == 0) & (target_label == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    print("Dev F1: ", f1)


    


