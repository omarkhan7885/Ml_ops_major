# src/quantize.py
import joblib
import numpy as np
from utils import load_model

def quantize_parameters(coefs, intercept):
    min_val = np.min(coefs)
    max_val = np.max(coefs)
    scale = 255 / (max_val - min_val)
    quantized = np.round((coefs - min_val) * scale).astype(np.uint8)
    dequantized = quantized / scale + min_val
    return quantized, dequantized, intercept

def run_quantization():
    model = load_model("model.joblib")
    coefs = model.coef_
    intercept = model.intercept_

    joblib.dump({'coef': coefs, 'intercept': intercept}, "unquant_params.joblib")

    q_params, deq_params, bias = quantize_parameters(coefs, intercept)
    joblib.dump({'coef': q_params, 'intercept': bias}, "quant_params.joblib")

    # Test inference with dequantized values
    sample_input = np.random.rand(1, len(deq_params))
    output = np.dot(sample_input, deq_params) + bias
    print("Sample inference output:", output[0])

if __name__ == "__main__":
    run_quantization()
