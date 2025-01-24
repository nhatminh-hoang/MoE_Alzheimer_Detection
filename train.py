import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from models.base_model import *  # Import your model
from data_preprocessing import *  # Import your data loading function
from scripts.utils import load_config

MODEL = {
    'MLP': MLP,
}

def accuracy(output, label):
    _, pred = torch.max(output, dim=1)
    return torch.sum(pred == label).item() / len(label)

def training(dt_loader, model:nn.Module, optimizer:torch.optim.Optimizer, criterion:nn.Module, flatten=False, device='cpu'):
    model.train()
    model.training = True
    losses = 0
    acc = 0

    for waveform, label in dt_loader:
        waveform, label = waveform.to(device), label.to(device)
        optimizer.zero_grad()

        if flatten:
            waveform = waveform.view(waveform.size(0), -1)

        output = model(waveform)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc += accuracy(output, label)
            losses += loss.cpu().detach().numpy()

    losses /= len(dt_loader)
    acc /= len(dt_loader)
    return losses, acc

def testing(dt_loader, model, criterion:nn.Module, flatten=False, device='cpu'):
    model.eval()
    model.training = False
    losses = 0
    acc = 0

    with torch.no_grad():
        for waveform, label in dt_loader:
            waveform, label = waveform.to(device), label.to(device)
            if flatten:
                waveform = waveform.view(waveform.size(0), -1)
            output = model(waveform)
            losses += criterion(output, label).cpu().detach().numpy()
            acc += accuracy(output, label)

    losses /= len(dt_loader)
    acc /= len(dt_loader)
    return losses, acc

def evaluate(model, test_loader, criterion, flatten=False, device='cpu', name_ex='ADReSS2020_MLP_waveform'):
    model.eval()
    model.training = False
    test_loss, test_acc = testing(test_loader, model, criterion, flatten, device=device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Metrics
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for waveform, label in test_loader:
            waveform, label = waveform.to(device), label.to(device)
            if flatten:
                waveform = waveform.view(waveform.size(0), -1)
            output = model(waveform)
            _, pred = torch.max(output, dim=1)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    save_evaluation_metrics(y_true, y_pred, name_ex)

def fit(name_ex, train_loader, val_loader, model, epochs, optimizer, criterion, 
        input_size, flatten=False, device='cpu', early_stop=5):
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_loss = float('inf')
    early_stop_counter = 0
    create_training_log(name_ex)
    save_model_summary(model, input_size=input_size, log_name=name_ex)

    for epoch in range(epochs):
        print('-'*50)
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_acc = training(train_loader, model, optimizer, criterion, flatten, device=device)
        val_loss, val_acc = testing(val_loader, model, criterion, flatten, device=device)
        with torch.no_grad():
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            log_training(name_ex, epoch, train_loss, train_acc, val_loss, val_acc)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), SAVED_PATH + f'{name_ex}.pth')
                early_stop_counter = 0
                best_epoch = epoch
            else:
                early_stop_counter += 1
                if early_stop_counter == early_stop:
                    print(f"Early stopping at epoch {epoch + 1} | Best epoch: {best_epoch + 1}")
                    break
    
    save_training_images(train_losses, train_accs, val_losses, val_accs, name_ex)
    print(f"Training completed. Save the model to {SAVED_PATH + name_ex}.pth")

    return train_losses, train_accs, val_losses, val_accs

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Getting arguments to create the new config 
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--data_name', type=str, help='Name of the dataset', default='ADReSS2020')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--flatten', type=bool, help='Flatten the input', default=False)
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--mfcc', type=bool, help='Use MFCC features', default=False)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('--input_size', type=int, default=14, help='Input feature dimension')
    parser.add_argument('--output_size', type=int, default=2, help='Number of output classes')
    parser.add_argument('--hidden_size', type=int, help='Hidden size', default=128)
    parser.add_argument('--dropout', type=float, help='Dropout rate', default=0.5)
    parser.add_argument('--early_stop', type=int, help='Early stopping', default=5)
    
    args = parser.parse_args()

    # Check if the model is valid
    if args.model not in MODEL:
        raise ValueError(f"Model {args.model} not found")
    
    # Check if the data_name is valid
    if args.data_name not in ['ADReSS2020']:
        raise ValueError(f"Dataset {args.data_name} not found")
    
    # Confirm the batch size when not using MFCC
    if not args.mfcc and args.batch_size > 32:
        # Calculate the GPU RAM usage
        gpu_ram = torch.cuda.get_device_properties(0).total_memory
        gpu_ram_usage = 0.00000095367431640625 * args.batch_size * 14 * 128

        while gpu_ram_usage > gpu_ram:

            input(f'GPU RAM usage is {gpu_ram_usage} and GPU RAM is {gpu_ram} consider reducing the batch size. Press Enter to continue or Ctrl+C to exit')
            args.batch_size = int(input('Enter a new batch size: '))
            # Calculate the GPU RAM usage=
            gpu_ram = torch.cuda.get_device_properties(0).total_memory
            gpu_ram_usage = 0.00000095367431640625 * args.batch_size * 14 * 128
    
    # Setting the input size of each data_name, using mFCC or waveform features and flatten or not
    if args.data_name == 'ADReSS2020':
        args.output_size = 2
        sr = 44100
        if args.flatten:
            if args.mfcc:
                args.input_size = 14
            else:
                args.input_size = 3 * sr
        else:
            args.input_size = 1
    
    print(args)
    # Create config file from the arguments
    mfcc_str = 'mfcc' if args.mfcc else 'waveform'
    name_ex = args.data_name + '_' + args.model + '_' + mfcc_str 
    name_config = name_ex + '.yaml'
    config_path = './config/experiment_configs/' + name_config
    if not os.path.exists(config_path):
        config = {}
        for arg in vars(args):
            config[arg] = getattr(args, arg)
        create_config(config_path, config)
    else:
        config = load_config(config_path)
    
    print(config)
    # Load data
    train_df, test_df = load_data(config['data_name'])
    train_loader, val_loader, test_loader = create_data_loaders(train_df, test_df, config['mfcc'], config['batch_size'], config['data_name'])

    # Load model
    model = MODEL[config['model']](input_size=config['input_size'], hidden_size=config['hidden_size'], output_size=config['output_size'], drop_out=config['dropout'])
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    # Train model
    train_losses, train_accs, val_losses, val_accs = fit(name_ex, train_loader, val_loader, 
                                                         model, config['epochs'], optimizer, criterion, 
                                                         config['input_size'], config['flatten'], 
                                                         early_stop=config['early_stop'], 
                                                         device=device)

    # Test model
    model.load_state_dict(torch.load(SAVED_PATH + f'{name_ex}.pth', weights_only=False))
    evaluate(model, test_loader, criterion, config['flatten'], device=device, name_ex=name_ex)

if __name__ == "__main__":
    main()
