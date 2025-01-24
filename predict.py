import torch
import torch.nn as nn
from models.base_model import BaseModel  # Import your model
from utils import load_config
import numpy as np

def predict(model, data, device):
    model.eval()  # Set the model to evaluation mode

    # Convert data to tensor if it's not already
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    # Add a batch dimension if the input is a single sample
    if data.dim() == 1:
        data = data.unsqueeze(0)

    data = data.to(device)

    with torch.no_grad():  # Disable gradient calculation
        output = model(data)
        # If it's a classification problem:
        # _, predicted = torch.max(output, 1)
        # return predicted.cpu().numpy()
        # For regression:
        return output.cpu().numpy()

def main():
    # Load configuration
    config = load_config("config/default_config.yaml")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and config['device'] == 'cuda' else "cpu")

    # Create model instance
    model = BaseModel().to(device)  # Replace with your model

    # Load the trained model
    model_path = config['model_path']
    model.load_state_dict(torch.load(model_path))

    # Example: Prepare your input data for prediction
    # This could involve loading from a file, preprocessing, etc.
    # Make sure the input data format matches what your model expects
    input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)  # Example input

    # Make a prediction
    prediction = predict(model, input_data, device)

    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
