import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.base_model import BaseModel  # Import your model
from data_preprocessing import create_data_loaders
from utils import load_config

def evaluate(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    targets = []

    with torch.no_grad():  # Disable gradient calculation
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)  # Get the index of the max log-probability (for classification)
            predictions.extend(predicted.cpu().numpy())
            targets.extend(target.cpu().numpy())

    # Calculate metrics (example for classification)
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    return accuracy, precision, recall, f1

def main():
    # Load configuration
    config = load_config("config/default_config.yaml")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and config['device'] == 'cuda' else "cpu")

    # Create model instance
    model = BaseModel().to(device)  # Replace with your model and make sure it matches the trained model

    # Load the trained model
    model_path = config['model_path']  # Path to the saved model
    model.load_state_dict(torch.load(model_path))

    # Load your data
    # Assuming you have a test_loader (similar to train_loader in train.py)
    # Example:
    from data_preprocessing import load_data, preprocess_data, create_data_loaders
    data_path = config['data_path']
    data = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    _, test_loader = create_data_loaders(X_train, X_test, y_train, y_test, batch_size=config['batch_size'])  # Use test data

    # Evaluate the model
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()
