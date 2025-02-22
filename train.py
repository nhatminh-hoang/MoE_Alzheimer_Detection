import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from models.base_model import *  # Import your model
from data_preprocessing import *  # Import your data loading function
from scripts.utils import load_config

MODEL = {
    'MLP': MLPModel,
    'CNN': CNNModel,
    'LSTM': LSTMModel,
    'BiLSTM': BiLSTMModel,
    'Transformer': TransformerModel,
}

def accuracy(output, label):
    if output.size(1) > 1:
        _, pred = torch.max(output, dim=1)
        return torch.sum(pred == torch.argmax(label, dim=1)).item() / len(label)
    else:
        pred = torch.round(output)
        return torch.sum(pred == label).item() / len(label)

def get_lr(it, warmup_steps, max_step, max_lr=1e-3, min_lr=1e-5):
    if it < warmup_steps:
        lr = (max_lr - min_lr) / warmup_steps * it + min_lr
        return lr
    if it > max_step:
        lr = min_lr
        return lr
    
    decay_ratio = (it - warmup_steps) / (max_step - warmup_steps)
    coeff = 0.5 * (1 + torch.cos(torch.tensor(decay_ratio) * np.pi))
    lr = min_lr + (max_lr - min_lr) * coeff
    return lr



def training(dt_loader, model:nn.Module, optimizer:torch.optim.Optimizer, criterion:nn.Module, flatten=False, device='cpu'):
    model.train()
    model.training = True
    losses = 0
    acc = 0

    for waveform, label in dt_loader:
        waveform, label = waveform.to(device), label.to(device)
        # One-hot encoding the label
        # label = torch.nn.functional.one_hot(label, num_classes=2).float()
        label = label.unsqueeze(-1).float()
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
            # One-hot encoding the label
            # label = torch.nn.functional.one_hot(label, num_classes=2).float()
            label = label.unsqueeze(-1).float()
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
            
            pred = torch.round(output)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    save_evaluation_metrics(y_true, y_pred, name_ex)

def fit(name_ex, train_loader, val_loader, model, epochs, optimizer, criterion, learning_rate,
        input_dummy, flatten, device='cpu', early_stop=5, USE_COMPILE=False):
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    lr_list = []
    best_val_loss = float('inf')
    early_stop_counter = 0
    create_training_log(name_ex)
    save_model_summary(model, input_data=input_dummy, log_name=name_ex)
    model = model.to(device)

    if USE_COMPILE:
        try:
            model = torch.compile(model)
            print("Model successfully compiled")
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")
            print("Falling back to eager mode")
    else:
        print("Running in eager mode (torch.compile disabled)")

    for epoch in range(epochs):
        print('-'*50)
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_acc = training(train_loader, model, optimizer, criterion, flatten, device=device)
        val_loss, val_acc = testing(val_loader, model, criterion, flatten, device=device)

        lr = get_lr(epoch, warmup_steps=epochs//100+5, max_step=epochs, max_lr=learning_rate, min_lr=1e-6)
        lr_list.append(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

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
    save_lr_plot(lr_list, name_ex)
    print('-'*50)
    print(f"Training completed. Save the model to {SAVED_PATH + name_ex}.pth")

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)
    if device.type == 'cuda':
        # Check CUDA capability
        if not torch.cuda.is_bf16_supported():
            print("Warning: BF16 not supported on this GPU. Using FP32 instead.")
            torch.set_float32_matmul_precision('high')
    
    # Suppress TorchDynamo errors to fall back to eager mode if compilation fails
    torch._dynamo.config.suppress_errors = True

    # Disable torch.compile if running on CPU or if CUDA capabilities are limited
    USE_COMPILE = device.type == 'cuda' and torch.cuda.get_device_capability()[0] >= 10

    # Getting arguments to create the new config 
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--data_name', type=str, help='Name of the dataset', default='ADReSS2020')
    parser.add_argument('--data_type', type=str, help='Type of data', default='audio')
    parser.add_argument('--wave_type', type=str, help='Type of waveform', default='full')
    parser.add_argument('--feature_type', type=str, help='Use features {MFCC, LogmelDelta}', default='None')

    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--flatten', type=bool, help='Flatten the input', default=False)
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-2)
    parser.add_argument('--output_size', type=int, default=2, help='Number of output classes')
    parser.add_argument('--hidden_size', type=int, help='Hidden size', default=128)
    parser.add_argument('--dropout', type=float, help='Dropout rate', default=0.5)
    parser.add_argument('--early_stop', help='Early stopping', default=5)
    
    args = parser.parse_args()

    # Check if the model is valid
    if args.model not in MODEL:
        raise ValueError(f"Model {args.model} not found")
    
    # Check if the data_name is valid
    if args.data_name not in ['ADReSS2020']:
        raise ValueError(f"Dataset {args.data_name} not found")
    
    # If the early stop is not a number, set it to epochs
    if not isinstance(args.early_stop, int):
        args.early_stop = args.epochs
    
    print(args)
    # Create config file from the arguments
    feature_type = 'mfcc' if args.feature_type == 'MFCC' else 'mel_delta_delta2' if args.feature_type == 'LogmelDelta' else 'waveform'
    text_name = 'text'
    audio_name = 'audio'+ '_' + args.wave_type + '_' + feature_type
    type_name = text_name if args.data_type == 'text' else audio_name
    name_ex = args.data_name + '_' + args.model + '_' + type_name + \
              '_' + str(args.epochs) + 'epochs' + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + \
              '_hs' + str(args.hidden_size) + '_do' + str(args.dropout)
    name_config = name_ex + '.yaml'
    config_path = './config/experiment_configs/' + name_config
    
    config = {}
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    create_config(config_path, config)
    
    print(config)
    # Load data
    train_loader, val_loader, test_loader = create_dataloader(data_type=config['data_type'],
                                                              feature_type=feature_type,
                                                              batch_size=config['batch_size'],
                                                              data_name=config['data_name'],
                                                              wave_type=config['wave_type'])
    input_dummy = next(iter(train_loader))[0]
    input_shape = input_dummy.shape
    if config['flatten']:
        input_size = input_shape[1]
    else: 
        input_size = input_shape[-1]

    print(f"Input shape: {input_shape}")
    print(f"Input size: {input_size}")
    # import sys; sys.exit()
    
    # Load model
    model = MODEL[config['model']](input_size=input_size, hidden_size=config['hidden_size'], output_size=config['output_size'], drop_out=config['dropout'])
    
    # Optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-3)
    criterion = nn.BCELoss()

    # Checking the input_shape to the model
    # model_summary = summary(model, input_data=input_dummy, device='cpu')

    # Train model
    fit(name_ex, train_loader, val_loader, model, config['epochs'], optimizer, criterion, 
        config['lr'], input_dummy, flatten=config['flatten'], device=device, early_stop=config['early_stop'],
        USE_COMPILE=USE_COMPILE)

    # Test model
    name_model = name_ex + '.pth'
    path_model = SAVED_PATH + name_model
    print(f"Load model from {path_model}")
    state_dict = torch.load(SAVED_PATH + f'{name_ex}.pth', weights_only=False)
    # Remove the '_orig_mod.' prefix from keys
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    evaluate(model, test_loader, criterion, flatten=config['flatten'], device=device, name_ex=name_ex)

if __name__ == "__main__":
    main()