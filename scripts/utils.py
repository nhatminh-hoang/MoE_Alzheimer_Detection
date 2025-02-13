import os
import yaml
from tqdm import tqdm

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

import torchaudio
import librosa

from torchinfo import summary

# Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
                            confusion_matrix, classification_report

# Folder paths
LOG_PATH = "./logs/"
SAVED_PATH = "./models/saved_models/"

ADReSS2020_DATAPATH = "./data/ADReSS-IS2020-data"
ADReSS2020_TRAIN_PATH = ADReSS2020_DATAPATH + "/train"
ADReSS2020_TEST_PATH = ADReSS2020_DATAPATH + "/test"

ADReSS2020_FULLWAVE = "/Full_wave_enhanced_audio"
ADReSS2020_CHUNKSWAVE = "/Normalised_audio-chunks"
ADReSS2020_TRANSCRIPTION = "/transcription"

AD_data_txt = "/cd_meta_data.txt"
NAD_data_txt = "/cc_meta_data.txt"

def create_config(config_path, config):
    with open(config_path, 'w') as f:
        yaml.dump(config, f)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_waveform(waveform, sample_rate, title="waveform"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    plt.figure(figsize=(10, 6))
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show(block=False)

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    plt.figure(figsize=(10, 6))
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show(block=False)

# Plot the distribution of time per audio file
def plot_time_distribution(df, split='train'):
  for label in [0, 1]:
      dic_path = ADReSS2020_TRAIN_PATH if split == 'train' else ADReSS2020_TEST_PATH
      label_df = df[df['Label '] == label]
      time_list = []
      for idx, row in label_df.iterrows():
          id = row['ID   '].strip()
          path = dic_path + ADReSS2020_CHUNKSWAVE
          if split == 'train':
              if label == 0:
                  path = os.path.join(path, 'cc')
              elif label == 1:
                  path = os.path.join(path, 'cd')
          audio_paths = os.listdir(path)
          for audio_path in audio_paths:
              if id in audio_path:
                file_path = os.path.join(path, audio_path)
                waveform, sample_rate = torchaudio.load(file_path)
                time = waveform.shape[-1] / sample_rate
                time_list.append(time)

  plt.figure(figsize=(10, 6))
  plt.hist(time_list, bins=100)
  plt.xlabel('Time (s)')
  plt.ylabel('Count')
  plt.title('Distribution of Time per Audio File')
  plt.show()

# Data Augmentation Functions
def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def pitch_shift(data, sampling_rate, pitch_factor=2):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def time_stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def pad_audio(waveform: torch.Tensor, sr, segment_length=25):
    segment_samples = sr * segment_length

    if waveform.shape[-1] < segment_samples:
        num_missing_samples = int(segment_samples - waveform.shape[-1])
        last_dim_padding = (0, num_missing_samples)
        waveform = F.pad(waveform, last_dim_padding)

    if waveform.shape[-1] > segment_samples:
        waveform = waveform[:, :segment_samples]

    return waveform
# Load and augment the audio data
def load_and_augment_audio(file_path, label, audio_data, audio_labels):
    data, sr = librosa.load(file_path, sr=None)
    augmented_data = [
        data,
        add_noise(data),
        pitch_shift(data, sr),
        time_stretch(data, 0.9),
        time_stretch(data, 1.1)
    ]
    for aug_data in augmented_data:
        audio_data.append(aug_data)
        audio_labels.append(label)

def process_batches(files, labels, batch_size):
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        batch_audio_data = []
        batch_audio_labels = []
        for file_path, label in tqdm(zip(batch_files, batch_labels), total=len(batch_files),
                                       desc=f"Processing batch {i//batch_size + 1}"):
            # Reference: [`scripts.utils.load_and_augment_audio`](scripts/utils.py#L130)
            load_and_augment_audio(file_path, label, batch_audio_data, batch_audio_labels)
        yield batch_audio_data, batch_audio_labels

# Segment the audio data into 25-second segments
def segment_audio(data, sr, segment_length=25):
    segment_samples = sr * segment_length
    segments = []
    for start in range(0, len(data), segment_samples):
        end = start + segment_samples
        if end <= len(data):
            segments.append(data[start:end])
    return segments

def calculate_silence_percentage(waveform, sr, silence_threshold=0.01):
    silent_samples = np.sum(np.abs(waveform) < silence_threshold)
    silence_percentage = silent_samples / len(waveform)
    return silence_percentage

# Feature extraction with customizable window size and hop length
def extract_features(data, sr, n_mfcc=13, window_size=2048, hop_length=512):
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, n_fft=window_size, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

def prepare_test_data(test_audio_files, test_labels):
    segmented_test_data = []
    segmented_test_labels = []
    for file_path, label in zip(test_audio_files, test_labels):
        data, sr = librosa.load(file_path, sr=None)
        segments = segment_audio(data, sr)
        segmented_test_data.extend(segments)
        segmented_test_labels.extend([label] * len(segments))
    return segmented_test_data, segmented_test_labels

def plot_time_distribution(df, split='train'):
  for label in [0, 1]:
      dic_path = ADReSS2020_TRAIN_PATH if split == 'train' else ADReSS2020_TEST_PATH
      label_df = df[df['Label '] == label]
      time_list = []
      for idx, row in label_df.iterrows():
          id = row['ID   '].strip()
          path = dic_path + ADReSS2020_CHUNKSWAVE
          if split == 'train':
              if label == 0:
                  path = os.path.join(path, 'cc')
              elif label == 1:
                  path = os.path.join(path, 'cd')
          audio_paths = os.listdir(path)
          for audio_path in audio_paths:
              if id in audio_path:
                file_path = os.path.join(path, audio_path)
                waveform, sample_rate = torchaudio.load(file_path)
                time = waveform.shape[-1] / sample_rate
                time_list.append(time)

  plt.figure(figsize=(10, 6))
  plt.hist(time_list, bins=100)
  plt.xlabel('Time (s)')
  plt.ylabel('Count')
  plt.title('Distribution of Time per Audio File')
  plt.show()

def create_training_log(log_name: str):
    '''Create a new log file with the given name'''
    if not os.path.exists(LOG_PATH + log_name):
        os.makedirs(LOG_PATH + log_name)
    with open(f'{LOG_PATH + log_name}/training_log.csv', 'w') as f:
        f.write('epoch,train_loss,train_acc,val_loss,val_acc\n')

def log_training(log_name: str, epoch, train_loss, train_acc, val_loss, val_acc):
    '''Log the training results to the log file'''
    with open(f'{LOG_PATH + log_name}/training_log.csv', 'a') as f:
        f.write(f'{epoch},{train_loss},{train_acc},{val_loss},{val_acc}\n')

def save_training_images(train_losses, train_accs, val_losses, val_accs, log_name: str):
    '''Save the training and validation loss and accuracy plots'''
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{LOG_PATH + log_name}/loss.png')

    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{LOG_PATH + log_name}/accuracy.png')

def save_evaluation_metrics(y_true, y_pred, log_name: str):
    '''Save the evaluation metrics to a file'''
    with open(f'{LOG_PATH + log_name}/evaluation_metrics.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy_score(y_true, y_pred)}\n')
        f.write(f'Precision: {precision_score(y_true, y_pred)}\n')
        f.write(f'Recall: {recall_score(y_true, y_pred)}\n')
        f.write(f'F1 Score: {f1_score(y_true, y_pred)}\n')
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'{LOG_PATH + log_name}/confusion_matrix.png')

    report = classification_report(y_true, y_pred)
    with open(f'{LOG_PATH + log_name}/classification_report.txt', 'w') as f:
        f.write(report)

def save_model_summary(model, input_shape: tuple, log_name: str):
    '''Save the model summary to a file'''
    # Save original device
    orig_device = next(model.parameters()).device
    # Move model to CPU for summary generation
    model_cpu = model.to('cpu')
    try:
        from torchinfo import summary
        model_summary = summary(model_cpu, input_size=input_shape, verbose=0)
        # Save or log the model_summary (e.g., write to a file)
        with open(f'logs/{log_name}_model_summary.txt', 'w') as f:
            f.write(str(model_summary))
    except Exception as e:
        print(f"Failed to generate model summary: {e}")
    # Move model back to its original device
    model.to(orig_device)

def save_lr_plot(lr_list, log_name: str):
    '''Save the learning rate plot to a file'''
    plt.figure(figsize=(10, 6))
    plt.plot(lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.savefig(f'{LOG_PATH + log_name}/learning_rate.png')