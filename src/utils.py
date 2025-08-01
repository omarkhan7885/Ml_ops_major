import os
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

#  Load the California housing dataset and split into train/test
def load_dataset():
    data = fetch_california_housing()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

#  Create and return a new Linear Regression model
def create_model():
    return LinearRegression()

# Save model to file using joblib
def save_model(model, filepath):
    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)
    joblib.dump(model, filepath)

#  Load model from file
def load_model(filepath):
    return joblib.load(filepath)

#  Evaluate predictions using R² and MSE
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse

#  Quantize weights to uint8 (shared scaling for the entire array)
def quantize_to_uint8(values, scale_factor=None):
    if np.all(values == 0):
        return np.zeros_like(values, dtype=np.uint8), 0.0, 0.0, 1.0

    if scale_factor is None:
        max_val = np.abs(values).max()
        scale_factor = 200.0 / max_val if max_val > 0 else 1.0

    scaled = values * scale_factor
    min_val, max_val = scaled.min(), scaled.max()

    if min_val == max_val:
        quantized = np.full_like(values, 127, dtype=np.uint8)
        return quantized, min_val, max_val, scale_factor

    normalized = ((scaled - min_val) / (max_val - min_val)) * 255
    clipped = np.clip(normalized, 0, 255).astype(np.uint8)

    return clipped, min_val, max_val, scale_factor

# Quantize each value individually (for finer control)
def quantize_to_uint8_individual(values):
    quantized = np.zeros_like(values, dtype=np.uint8)
    metadata = []

    for i, val in enumerate(values):
        if val == 0:
            quantized[i] = 127
            metadata.append({'min_val': 0.0, 'max_val': 0.0, 'scale': 1.0})
            continue

        abs_val = abs(val)
        scale = 127.0 / abs_val
        q_val = 127 - abs_val * scale if val < 0 else 128 + abs_val * scale

        quantized[i] = int(np.clip(q_val, 0, 255))
        metadata.append({
            'min_val': val,
            'max_val': val,
            'scale': scale,
            'original': val
        })

    return quantized, metadata

#  Reconstruct original values from individual quantized values
def dequantize_from_uint8_individual(quantized_values, metadata):
    dequantized = np.zeros_like(quantized_values, dtype=np.float32)

    for i, (q_val, meta) in enumerate(zip(quantized_values, metadata)):
        if meta['scale'] == 1.0:
            dequantized[i] = 0.0
        elif q_val <= 127:
            dequantized[i] = -(127 - q_val) / meta['scale']
        else:
            dequantized[i] = (q_val - 128) / meta['scale']

    return dequantized

#  Reconstruct original values using shared quantization metadata
def dequantize_from_uint8(quantized_values, min_val, max_val, scale_factor):
    if min_val == max_val:
        return np.full_like(quantized_values, min_val / scale_factor, dtype=np.float32)

    value_range = max_val - min_val
    denormalized = (quantized_values.astype(np.float32) / 255.0) * value_range + min_val
    return denormalized / scale_factor
