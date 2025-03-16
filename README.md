# DON_Concentration

## Overview
This project focuses on predicting DON concentration in corn samples using hyperspectral imaging data. It involves data preprocessing, model training, evaluation, and API deployment for real-time predictions.

## Repository Structure
```
MLE-Assignment/
│── data/
│   ├── MLE-Assignment.csv  # Raw dataset
│   ├── processed_data.csv  # Preprocessed dataset
│── src/
│   ├── data_preprocessing.py  # Data preprocessing pipeline
│   ├── model_training.py  # Neural network model training
│   ├── model_evaluation.py  # Model evaluation and validation
│   ├── inference.py  # Making predictions on new data
│── deployment/
│   ├── app.py  # FastAPI implementation for real-time prediction
│   ├── Dockerfile  # Docker container setup
│── tests/
│   ├── test_model.py  # Unit tests for preprocessing and model validation
│── models/
│   ├── trained_model.pth  # Saved model weights
│── requirements.txt  # Dependencies
│── config.yaml  # Configurations for preprocessing and training
│── train.py  # Script to run full training pipeline
│── run.sh  # Shell script to automate execution
│── README.md  # Project documentation
│── report.pdf  # Final project report
```

## Setup Instructions
### 1. Clone the Repository
```bash
git clone <repository_url>
cd MLE-Assignment
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Preprocessing Pipeline
```bash
python src/data_preprocessing.py
```

### 5. Train the Model
```bash
python src/model_training.py
```

### 6. Evaluate the Model
```bash
python src/model_evaluation.py
```

### 7. Run the API
```bash
uvicorn deployment.app:app --host 0.0.0.0 --port 8000
```

## API Usage
### Test the API using cURL
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"features": [0.5, 1.2, -0.3, 0.8, ...]}'
```

### Test with Python Requests
```python
import requests
url = "http://127.0.0.1:8000/predict"
data = {"features": [0.5, 1.2, -0.3, 0.8, ...]}
response = requests.post(url, json=data)
print(response.json())
```

## Running Tests
```bash
pytest tests/
```

## Docker Setup
### 1. Build the Docker Image
```bash
docker build -t mle-assignment .
```

### 2. Run the Container
```bash
docker run -p 8000:8000 mle-assignment
```
## [Report](https://docs.google.com/document/d/1VhziDAZ_123gH--wI0egCKVxi_sl8BxTzmhp96YTKkI/edit?tab=t.0)

