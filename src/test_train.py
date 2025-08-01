import warnings
import pytest
from sklearn.linear_model import LinearRegression
import joblib
from train import train_model

def test_model_training():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        model = train_model()
    assert isinstance(model, LinearRegression), "Model is not LinearRegression"
    assert hasattr(model, "coef_"), "Model does not have coefficients"

def test_model_saved():
    model = joblib.load("model.joblib")
    assert model is not None, "Saved model not found"
