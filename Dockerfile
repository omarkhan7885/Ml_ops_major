# Use a slim Python image
FROM python:3.11-slim

# Set metadata
LABEL maintainer="Anmol" \
      description="MLOps pipeline for California Housing prediction"

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

# Copy application source code
COPY src/ src/

# Copy models if available
COPY model.joblib . 
COPY quant_params.joblib . 
COPY unquant_params.joblib . 

# Default command to run prediction
CMD ["python", "src/predict.py"]
