import os
import re
import yaml
import random
from tqdm import tqdm

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torchaudio
import librosa

import pylangacq

from torchinfo import summary

# Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
                            confusion_matrix, classification_report

# Folder paths
LOG_PATH = "./logs/"
SAVED_PATH = "./models/saved_models/"

ADReSS2020_DATAPATH = "./data/ADReSS-IS2020-data"
ADReSS2020_TRAINPATH = ADReSS2020_DATAPATH + "/train"
ADReSS2020_TESTPATH = ADReSS2020_DATAPATH + "/test"

ADReSS2020_FULLWAVE = "/Full_wave_enhanced_audio"
ADReSS2020_CHUNKSWAVE = "/Normalised_audio-chunks"
ADReSS2020_TRANSCRIPTION = "/transcription"
TRANSCRIPT_NAME = "transcription"

AD_data_txt = "/cd_meta_data.txt"
NAD_data_txt = "/cc_meta_data.txt"

import string
punctuations = string.punctuation

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
      dic_path = ADReSS2020_TRAINPATH if split == 'train' else ADReSS2020_TESTPATH
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
      dic_path = ADReSS2020_TRAINPATH if split == 'train' else ADReSS2020_TESTPATH
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

def save_model_summary(model, input_data: tuple, log_name: str):
    '''Save the model summary to a file'''
    # print(input_shape)
    model_summary = summary(model, input_data=input_data, device='cpu')
    # Save or log the model_summary (e.g., write to a file)
    with open(f'logs/{log_name}/model_summary.txt', 'w') as f:
        f.write(str(model_summary))

def save_lr_plot(lr_list, log_name: str):
    '''Save the learning rate plot to a file'''
    plt.figure(figsize=(10, 6))
    plt.plot(lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.savefig(f'{LOG_PATH + log_name}/learning_rate.png')

def read_par_utterances(file_path):
    """
    Read a CHAT file and return a list of merged *PAR (and *INV) utterances.
    This function merges continuation lines and removes trailing time codes.
    """
    utterances = []
    current_utterance = None

    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            # New utterance: lines starting with *PAR: or *INV:
            if line.startswith("*PAR:") or line.startswith("*INV:"):
                # If an utterance is in progress, finish and store it.
                if current_utterance is not None:
                    # Remove any trailing time code (text between two  symbols)
                    current_utterance = re.sub(r'.*?', '', current_utterance).strip()
                    utterances.append(current_utterance)
                # Start a new utterance (remove the marker)
                if line.startswith("*PAR:"):
                    current_utterance = line[len("*PAR:"):].strip()
                else:
                    current_utterance = line[len("*INV:"):].strip()
            # Continuation lines (indented or containing a time code marker) are appended.
            elif current_utterance is not None and ('' in line):
                current_utterance += " " + line.strip()
            # Otherwise, ignore the line.
    
    # Append the final utterance if one is in progress.
    if current_utterance:
        current_utterance = re.sub(r'.*?', '', current_utterance).strip()
        utterances.append(current_utterance)
    
    return utterances

def is_retracing(token):
    """
    Determine if a token is a retracing token that should be merged with the previous token.
    
    Returns True for tokens matching patterns like:
      - o(f)
      - fallin(g)
      - an(d)
      - stealin(g)
    (case-insensitive)
    
    But returns False for tokens such as (.), (..), or (...).
    """
    pattern = re.compile(r'^(an|o|stealin|takin)\([^)]*\)$', re.IGNORECASE)
    if pattern.match(token):
        return True
    return False

def merge_annotation_tokens(tokens, start_index):
    """
    Merge tokens that are part of an annotation enclosed in brackets.
    This function supports both square-bracket annotations (e.g., "[+ exc]") and
    angle-bracket annotations (e.g., "<walk with a>").
    
    Returns a tuple of (merged_token, next_index).
    """
    token = tokens[start_index]
    if token.startswith('['):
        closing = ']'
    elif token.startswith('<'):
        closing = '>'
    else:
        return token, start_index + 1

    merged = token
    i = start_index
    # If the token already ends with the closing bracket, return it.
    if merged.endswith(closing):
        return merged, i + 1
    i += 1
    # Merge subsequent tokens until we find one that ends with the closing bracket.
    while i < len(tokens) and not tokens[i].endswith(closing):
        merged += " " + tokens[i]
        i += 1
    if i < len(tokens):
        merged += " " + tokens[i]
        i += 1
    return merged, i

def tokenize_and_merge(utterance):
    """
    Tokenize an utterance into tokens with the following custom behavior:
    
      - If a token is immediately followed by a retracing token 
        (e.g., 'o(f)', 'fallin(g)', 'an(d)', 'stealin(g)'), merge them into a single token 
        by concatenating with the linking_token.
      - Merge bracketed annotations so that tokens like "[+ exc]" or "[: overflowing]" 
        and angle-bracket annotations like "<walk with a>" remain intact.
    
    Returns a list of tokens.
    """
    tokens = utterance.split()
    merged_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # If the token begins with '[' or '<' but does not end with the corresponding closing bracket,
        # merge the entire annotation.
        if (token.startswith('[') and not token.endswith(']')) or (token.startswith('<') and not token.endswith('>')):
            merged_token, i = merge_annotation_tokens(tokens, i)
            merged_tokens.append(merged_token)
            continue
        
        merged_tokens.append(token)
        i += 1
        
    return merged_tokens

def get_chat_data(split):
    """
    Get the CHAT data from the ADReSS-IS2020 dataset.
    
    Parameters:
        split (str): The split to load (either "train" or "test").
        data_path (str): The path to the ADReSS-IS2020 dataset.
    
    Returns:
        DataFrame: A DataFrame containing the CHAT data and labels.
    """
    path = ADReSS2020_TRAINPATH if split == "train" else ADReSS2020_TESTPATH
    # Define the path to the transcript files.
    transcript_path = os.path.join(path, TRANSCRIPT_NAME)
    # Read the CHAT data.
    reader = pylangacq.read_chat(transcript_path)

    file_paths = reader.file_paths()
    data = []

    test_df = pd.read_csv(ADReSS2020_DATAPATH + '/2020Labels.txt', delimiter=';', skipinitialspace=True)
    test_df = test_df.drop(columns=['age', 'mmse', 'gender'], axis=1)

    # Read and merge utterances from each file.
    for file_path in file_paths:
        # Read and merge *PAR utterances.
        utterances = read_par_utterances(file_path)

        # Tokenize and merge tokens from each utterance.
        all_tokens = []
        for utt in utterances:
            tokens = tokenize_and_merge(utt)
            all_tokens.extend([token for token in tokens if token not in list(punctuations)])

        if split == 'test':
            label = test_df[test_df['ID'] == os.path.basename(file_path).split('.')[0] + ' '].Label.iloc[0]
        
        elif split == 'train':
            label = 0 if 'cc' in file_path else 1

        data.append((all_tokens, label))

    return pd.DataFrame(data, columns=['tokens', 'label'])

def split_tokens(tokens, max_len=300):
    if len(tokens) <= max_len:
        return [tokens + [1] * (max_len - len(tokens))]
    else:
        split = [tokens[i:i+max_len] for i in range(0, len(tokens), max_len)]
        split[-1] += [1] * (max_len - len(split[-1]))
        return split

def load_special_tokens(file_path):
    """Load special tokens from file"""
    with open(file_path, 'r') as f:
        # Load all special tokens and regular tokens
        return [line.strip() for line in f if line.strip()]

def add_brackets(token):
    """Add <> brackets around a token"""
    return f'<{token}>'

def augment_dataset_with_special_tokens(tokens_list, special_tokens, 
                                      bracket_prob=0.2, 
                                      max_brackets_per_seq=2,
                                      num_special_tokens=2):
    """
    Augment dataset by:
    1. Adding special tokens from file
    2. Randomly adding <> to some existing tokens
    """
    augmented_dataset = []
    
    for tokens in tokens_list:
        augmented_tokens = tokens.copy()
        
        # 1. Add random special tokens - insert them one by one
        selected_special = random.sample(special_tokens, 
                                       k=min(num_special_tokens, len(special_tokens)))
        for special_token in selected_special:
            insert_pos = random.randint(0, len(augmented_tokens))
            augmented_tokens.insert(insert_pos, special_token)
        
        # 2. Randomly add brackets to existing tokens
        if random.random() < bracket_prob:
            eligible_positions = [i for i, token in enumerate(tokens) 
                               if '<' not in token and '>' not in token]
            
            if eligible_positions:
                num_to_modify = random.randint(1, min(len(eligible_positions), 
                                                    max_brackets_per_seq))
                positions_to_modify = random.sample(eligible_positions, num_to_modify)
                
                for pos in positions_to_modify:
                    augmented_tokens[pos] = add_brackets(augmented_tokens[pos])
        
        augmented_dataset.append(augmented_tokens)
    
    return augmented_dataset

def back_translation_augment(text):
    # Placeholder for back translation augmentation
    # You would need to implement actual translation logic here
    return text

def synonym_replacement(text, n=1):
    # Placeholder for synonym replacement
    # You would need to implement word replacement logic here 
    return text

def random_swap(text, n=1):
    # Randomly swap n pairs of words in the text
    words = text.split()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)

def random_deletion(text, p=0.1):
    # Randomly delete words with probability p
    words = text.split()
    words = [word for word in words if random.random() > p]
    return " ".join(words)