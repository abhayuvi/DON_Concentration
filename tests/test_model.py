import unittest
import pandas as pd
import torch
from model_training import RegressionNN, load_data, train_model
from data_preprocessing import preprocess_data

class TestModel(unittest.TestCase):
    def setUp(self):
        self.input_file = "../data/processed_data.csv"
        self.X_train, self.X_test, self.y_train, self.y_test = load_data(self.input_file)
        self.model = train_model(self.X_train, self.y_train, input_dim=self.X_train.shape[1], epochs=1)

    def test_model_output_shape(self):
        """Ensure model outputs correct shape."""
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        output = self.model(X_test_tensor)
        self.assertEqual(output.shape[0], self.X_test.shape[0])

    def test_preprocessing(self):
        """Check if preprocessing runs without errors."""
        df, _ = preprocess_data(self.input_file)
        self.assertFalse(df.isnull().values.any())

if __name__ == "__main__":
    unittest.main()
