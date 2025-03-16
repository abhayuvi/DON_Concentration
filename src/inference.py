import torch
import pandas as pd
import numpy as np
from model_training import RegressionNN

def load_model(model_path, input_dim):
    """Load trained model for inference."""
    model = RegressionNN(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, input_data):
    """Make predictions for new data."""
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    predictions = model(input_tensor).detach().numpy().flatten()
    return predictions

if __name__ == "__main__":
    model_path = "../models/trained_model.pth"
    input_file = "../data/new_samples.csv"  # Assume new spectral data
    df = pd.read_csv(input_file)
    
    model = load_model(model_path, input_dim=df.shape[1])
    predictions = predict(model, df.values)
    
    df["Predicted_DON"] = predictions
    df.to_csv("../data/predictions.csv", index=False)
    print("Predictions saved.")
