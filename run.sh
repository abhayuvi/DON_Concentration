#!/bin/bash

# Run data preprocessing
python src/data_preprocessing.py

echo "Data Preprocessing Completed."

# Run visualization
python src/visualization.py

echo "Data Visualization Completed."

# Train model
python src/model_training.py

echo "Model Training Completed."

# Evaluate model
python src/model_evaluation.py

echo "Model Evaluation Completed."

# Start API service
uvicorn deployment:app --host 0.0.0.0 --port 8000
