import numpy as np
from utils import load_model, load_dataset, calculate_metrics

def main():
    print(" Loading the trained Linear Regression model...")
    model = load_model("model.joblib")

    print(" Fetching the test data...")
    _, X_test, _, y_test = load_dataset()

    print(" Generating predictions on test data...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2_score, mse = calculate_metrics(y_test, y_pred)

    print("\n Model Evaluation Metrics:")
    print(f" - R² Score            : {r2_score:.4f}")
    print(f" - Mean Squared Error  : {mse:.4f}")

    print("\n Sample Predictions:")
    for i in range(min(10, len(y_test))):
        actual = y_test[i]
        predicted = y_pred[i]
        error = abs(actual - predicted)
        print(f"   ▶ True: {actual:.2f} | Predicted: {predicted:.2f} | Error: {error:.2f}")

    print("\n Prediction process completed successfully.")
    return True

if __name__ == "__main__":
    main()
