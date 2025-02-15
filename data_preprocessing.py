import os
import time
from tqdm import tqdm

import pandas as pd
import torchaudio 

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from scripts.utils import *

# Define PATH
ADReSS2020_DATAPATH = "./data/ADReSS-IS2020-data"
ADReSS2020_TRAINPATH = os.path.join(ADReSS2020_DATAPATH, "train")
ADReSS2020_TESTPATH = os.path.join(ADReSS2020_DATAPATH, "test")

FULL_WAVE_NAME = "Full_wave_enhanced_audio"
CHUNK_WAVE_NAME = "Normalised_audio-chunks"

# Function to get file paths and labels
def get_audio_files_and_labels(dataset_path, split_folder_path, split):
    audio_files = []
    labels = []

    if split == 'train':
        for folder in os.listdir(split_folder_path):
            folder_path = os.path.join(split_folder_path, folder)
            if os.path.isdir(folder_path) and os.path.basename(folder_path) == FULL_WAVE_NAME:
                for label in os.listdir(folder_path):
                    label_path = os.path.join(folder_path, label)
                    if os.path.isdir(label_path):
                        for file_name in os.listdir(label_path):
                            if file_name.endswith('.wav'):
                                audio_files.append(os.path.join(label_path, file_name))
                                if label == 'cc':
                                    labels.append(0)
                                elif label == 'cd':
                                    labels.append(1)
    
    elif split == 'test':
        test_df = pd.read_csv(dataset_path + '/2020Labels.txt', delimiter=';', skipinitialspace=True)
        test_df = test_df.drop(columns=['age', 'mmse', 'gender'], axis=1)
        
        for folder in os.listdir(split_folder_path):
            folder_path = os.path.join(split_folder_path, folder)
            if os.path.isdir(folder_path) and os.path.basename(folder_path) == FULL_WAVE_NAME:
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.wav'):
                        audio_name = file_name.split('.')[0] + ' '
                        audio_files.append(os.path.join(folder_path, file_name))
                        labels.append(test_df[test_df['ID'] == audio_name].Label.iloc[0])
                        
    return audio_files, labels

class ADreSS2020Dataset(Dataset):
    '''
    ADreSS2020Dataset class

    Args:

    Returns:
        (Waveform, Sample rate) of the audio file
        Label of the audio file
    '''
    def __init__(self, data_path, audio_files, labels, split, wave_type='full', feature_type='mfcc'):
        self.data_path = data_path
        self.audio_files = audio_files
        self.labels = labels
        self.split = split
        self.wave_type = wave_type
        self.feature_type = feature_type

        if self.feature_type == 'mfcc':
            if self.split == 'train':
                if os.path.exists(ADReSS2020_DATAPATH + '/preprocessed/' + 'mfcc_X_train.npy') and os.path.exists(ADReSS2020_DATAPATH + '/preprocessed/' + 'mfcc_y_train.npy'):
                    self.preprocess = np.load(ADReSS2020_DATAPATH + '/preprocessed/' + 'mfcc_X_train.npy'), np.load(ADReSS2020_DATAPATH + '/preprocessed/' + 'mfcc_y_train.npy')
                else:
                    self.preprocess = self._preprocess_mfcc(audio_files, labels)
            elif self.split == 'test':
                if os.path.exists(ADReSS2020_DATAPATH + '/preprocessed/' + 'mfcc_X_test.npy') and os.path.exists(ADReSS2020_DATAPATH + '/preprocessed/' + 'mfcc_y_test.npy'):
                    self.preprocess = np.load(ADReSS2020_DATAPATH + '/preprocessed/' + 'mfcc_X_test.npy'), np.load(ADReSS2020_DATAPATH + '/preprocessed/' + 'mfcc_y_test.npy')
                else:
                    self.preprocess = self._preprocess_mfcc(audio_files, labels)

        else:
            self.preprocess = prepare_test_data(audio_files, labels)

    def __len__(self):
        return len(self.preprocess[0])
    
    def __getitem__(self, idx):
        return np.expand_dims(self.preprocess[0][idx], -1).astype(np.float32), self.preprocess[1][idx]
    
    def _preprocess_mfcc(self, audio_files, labels):
        custom_window_size = 1024
        custom_hop_length = 256
        
        if self.split == 'train':
            # Prepare training data
            audio_data = []
            audio_labels = []

            for batch_data, batch_labels in process_batches(audio_files, labels, batch_size=8):
                # Process each batch (e.g., further pre-processing or saving results)
                audio_data.extend(batch_data)
                audio_labels.extend(batch_labels)

            segmented_data = []
            segmented_labels = []

            # Segment the audio data into 25-second segments
            for data, label in zip(audio_data, audio_labels):
                sr = librosa.get_samplerate(audio_files[0])
                segments = segment_audio(data, sr)
                segmented_data.extend(segments)
                segmented_labels.extend([label] * len(segments))
            del audio_data, audio_labels

            # Extract features
            features = []

            for segment in segmented_data:
                mfccs = extract_features(segment, sr, window_size=custom_window_size, hop_length=custom_hop_length)
                features.append(mfccs)

            X = np.array(features)
            y = np.array(segmented_labels)
            del features, segmented_data, segmented_labels

            # Save the preprocessed data
            np.save(ADReSS2020_DATAPATH + '/preprocessed/' + 'mfcc_X_train.npy', X)
            np.save(ADReSS2020_DATAPATH + '/preprocessed/' + 'mfcc_y_train.npy', y)

        elif self.split == 'test':
            segmented_test_data, segmented_test_labels = prepare_test_data(audio_files, labels)
            features_test = []
            for segment in segmented_test_data:
                sr = librosa.get_samplerate(audio_files[0])
                mfccs = extract_features(segment, sr, window_size=custom_window_size, hop_length=custom_hop_length)
                features_test.append(mfccs)

            X = np.array(features_test)
            y = segmented_test_labels
            del features_test, segmented_test_data, segmented_test_labels

            # Save the preprocessed data
            np.save(ADReSS2020_DATAPATH + '/preprocessed/' + 'mfcc_X_test.npy', X)
            np.save(ADReSS2020_DATAPATH + '/preprocessed/' + 'mfcc_y_test.npy', y)

        return X, y

def load_audio_data(data_name='ADReSS2020'):
    """Loads data from a CSV file.

    Args:
        data_name (str): Name of the dataset to load.

    Returns:
        train_audio_files (list), train_labels (list), test_audio_files (list), test_labels (list): Data from the specified dataset.
    """

    if data_name == 'ADReSS2020':
        # Load train and test data
        train_audio_files, train_labels = get_audio_files_and_labels(ADReSS2020_DATAPATH, ADReSS2020_TRAINPATH, split='train')
        test_audio_files, test_labels = get_audio_files_and_labels(ADReSS2020_DATAPATH, ADReSS2020_TESTPATH, split='test')

    return train_audio_files, train_labels, test_audio_files, test_labels

def create_audio_data_loaders(train_audio_files, train_labels, test_audio_files, test_labels, 
                        wave_type='full', feature_type='mfcc', batch_size=32, data_name='ADReSS2020'):
    """Creates PyTorch DataLoaders for training and testing sets.

    Args:

        mfcc (bool): Whether to use MFCC features.
        batch_size (int): Batch size for the DataLoaders.

    Returns:
        torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader: DataLoaders for training, validation, and testing sets.
    """
    # Make sure the results are reproducible and training data and validation data cannot overlap
    generator = torch.Generator()
    generator.manual_seed(42)

    # Create Datasets
    if data_name == 'ADReSS2020':
        train_ds = ADreSS2020Dataset(ADReSS2020_DATAPATH, train_audio_files, train_labels, split='train', wave_type=wave_type, feature_type=feature_type)
        test_ds = ADreSS2020Dataset(ADReSS2020_DATAPATH, test_audio_files, test_labels, split='test', wave_type=wave_type, feature_type=feature_type)

    # val_ds, train_ds = random_split(train_ds, [0.2, 0.8], generator=generator)
    val_ds = test_ds

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def create_dataloader(data_name, data_type, batch_size=32, feature_type='mfcc', wave_type='full'):
    """Creates PyTorch DataLoader for the specified dataset.

    Args:
        data_name (str): Name of the dataset to load.
        data_type (str): Type of data to load (e.g., 'train', 'test').
        batch_size (int): Batch size for the DataLoader.
        mfcc (bool): Whether to use MFCC features.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the specified dataset.
    """
    # Load data
    if data_type == 'audio':
        train_audio_files, train_labels, test_audio_files, test_labels = load_audio_data(data_name)
        print("Load data successful!")

        train_loader, val_loader, test_loader = create_audio_data_loaders(train_audio_files, train_labels, test_audio_files, test_labels, \
                                                                          data_name=data_name,
                                                                          feature_type=feature_type, batch_size=batch_size, wave_type=wave_type)
        print("DataLoaders created successfully!")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    start = time.time()

    train_loader, val_loader, test_loader = create_dataloader('ADReSS2020', 'audio', batch_size=32, feature_type='mfcc')

    print("Time taken: ", f'{time.time() - start:.2f} seconds')
    start = time.time()

    print("Train data: ", len(train_loader.dataset))
    print("Validation data: ", len(val_loader.dataset))
    print("Test data: ", len(test_loader.dataset))

    print("Sample data: ", next(iter(train_loader))[0].shape)
    print("Get data's time taken: ", f'{time.time() - start:.2f} seconds')