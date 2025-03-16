import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model_training import RegressionNN, load_data

def load_model(model_path, input_dim):
    """Load the trained model."""
    model = RegressionNN(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and compute metrics."""
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    predictions = model(X_test_tensor).detach().numpy().flatten()
    
    # Check for NaN values in predictions
    if np.isnan(predictions).any():
        raise ValueError("Model predictions contain NaN values. Check preprocessing and model training.")
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return predictions

def plot_results(y_test, predictions):
    """Plot actual vs. predicted values."""
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.xlabel("Actual DON Concentration")
    plt.ylabel("Predicted DON Concentration")
    plt.title("Actual vs. Predicted Values")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    input_file = "/Users/apple/DON_Concentration/data/processed_data.csv"
    model_path = "../models/trained_model.pth"
    X_train, X_test, y_train, y_test = load_data(input_file)
    
    # Check for NaN values in test set
    if np.isnan(X_test).any() or np.isnan(y_test).any():
        raise ValueError("Test set contains NaN values. Check preprocessing steps.")
    
    model = load_model(model_path, input_dim=X_train.shape[1])
    predictions = evaluate_model(model, X_test, y_test)
    plot_results(y_test, predictions)
