from fastapi import FastAPI
import torch
import numpy as np
import pandas as pd
from pydantic import BaseModel
from src.model_training import RegressionNN

app = FastAPI()

import torch
import pandas as pd


# Load dataset to determine correct input size
df = pd.read_csv("/Users/apple/DON_Concentration/data/processed_data.csv")
INPUT_DIM = df.shape[1] - 1  # All columns except target

# Load model with the correct input size
MODEL_PATH = "../models/trained_model.pth"
model = RegressionNN(INPUT_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    """Predict DON concentration from input spectral features."""
    input_tensor = torch.tensor([data.features], dtype=torch.float32)
    prediction = model(input_tensor).detach().numpy().flatten()[0]
    return {"predicted_don": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
