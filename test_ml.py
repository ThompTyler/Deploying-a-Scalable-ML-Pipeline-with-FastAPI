import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

from ml.model import train_model, inference, compute_model_metrics

def test_train_model_returns_model():
    """
    Test that train_model returns a trained LogisticRegression model.
    """
    X = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
    y = np.array([0, 1, 1, 0])

    model = train_model(X, y)

    assert isinstance(model, LogisticRegression)


def test_inference_output_shape():
    """
    Test that inference returns predictions with correct shape.
    """
    X = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
    y = np.array([0, 1, 1, 0])

    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == y.shape


def test_compute_model_metrics_range():
    """
    Test that compute_model_metrics returns valid precision, recall, and fbeta scores.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
