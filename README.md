# DON_Concentration

## Objective
Develop a machine learning pipeline to predict DON concentration in corn samples using hyperspectral imaging data.

## Project Structure
```
MLE-Assignment/
│── data/
│   ├── MLE-Assignment.csv  # Provided dataset
│── src/
│   ├── data_preprocessing.py  # Data preprocessing pipeline
│   ├── visualization.py  # Data visualization scripts
│   ├── model_training.py  # Train regression model (Neural Network)
│   ├── model_evaluation.py  # Evaluate model performance
│   ├── inference.py  # Load model and make predictions
│── deployment/
│   ├── app.py  # FastAPI for real-time prediction
│   ├── Dockerfile  # Deployment containerization
│── tests/
│   ├── test_model.py  # Unit tests for preprocessing and model
│── docs/
│   ├── report.pdf  # Summary of findings, model insights
│── requirements.txt  # Dependencies
│── config.yaml  # Configurations for preprocessing and training
│── train.py  # Main training script
│── run.sh  # Bash script to automate execution
```

## Installation
```bash
pip install -r requirements.txt
```

## Running the Pipeline
```bash
python src/data_preprocessing.py
python src/visualization.py
python src/model_training.py
python src/model_evaluation.py
```

## Running API
```bash
uvicorn deployment.app:app --host 0.0.0.0 --port 8000
```

## Running Tests
```bash
python -m unittest tests/
```

## Deployment
To build and run the Docker container:
```bash
docker build -t mle-assignment .
docker run -p 8000:8000 mle-assignment
```

