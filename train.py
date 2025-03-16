import pandas as pd
from model_training import train_model, load_data

if __name__ == "__main__":
    input_file = "../data/processed_data.csv"
    X_train, X_test, y_train, y_test = load_data(input_file)
    
    print("Starting model training...")
    model = train_model(X_train, y_train, input_dim=X_train.shape[1])
    print("Training complete. Model saved.")
