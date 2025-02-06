import os
import time
from tqdm import tqdm

import pandas as pd
import torchaudio 

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from scripts.utils import *

class ADreSS2020Dataset(Dataset):
    '''
    ADreSS2020Dataset class

    Args:
        df: Pandas dataframe containing the data
        split: 'train' or 'test'
        wave_type: 'full' or 'chunk'

    Returns:
        (Waveform, Sample rate) of the audio file
        Label of the audio file
    '''
    def __init__(self, df, transforms=None, split='train', wave_type='full',
                 transcript=False,
                 feature_type='mfcc',
                 max_time=3):
        self.df = df.copy()
        self.split = split
        self.wave_type = wave_type
        self.split_path = ADReSS2020_TRAIN_PATH if split == 'train' else ADReSS2020_TEST_PATH
        self.transforms = transforms
        self.max_time = max_time
        self.mfcc = feature_type == 'mfcc'
        self.mel_delta_delta2 = feature_type == 'mel_delta_delta2'

        self.fullwave_path = self.split_path + ADReSS2020_FULLWAVE
        self.chunkwave_path = self.split_path + ADReSS2020_CHUNKSWAVE
        self.transcription_path = self.split_path + ADReSS2020_TRANSCRIPTION

        # Cache paths for improved performance
        self.df = self._get_df_chunks() if wave_type == 'chunk' else self.df
        self.ID = 'ID   ' if self.wave_type == 'full' else 'ID'
        self.LABEL = 'Label ' if self.wave_type == 'full' else 'Label'

        self.cached_paths = self._cache_paths()
        _, self.sr = torchaudio.load(self.cached_paths[0][0])

        if self.wave_type == 'full':
            self.max_time = 150

        # Preprocess MFCC dataset
        if self.mfcc:
            self.cached_data = []
            if not os.path.exists(f'{ADReSS2020_DATAPATH}/preprocessed/mfcc_{self.wave_type}_{split}.pt'):
                self.cached_data = self._mfcc_preprocess_dataset(split)
            else :
                self.cached_data = torch.load(f'{ADReSS2020_DATAPATH}/preprocessed/mfcc_{self.wave_type}_{split}.pt', weights_only=False)

        # Preprocess log-mel delta delta2 dataset
        if self.mel_delta_delta2:
            self.cached_data = []
            if not os.path.exists(f'{ADReSS2020_DATAPATH}/preprocessed/logmel_delta_delta2_{self.wave_type}_{split}.pt'):
                self.cached_data = self._logmel_delta_delta2_preprocess_dataset(split)
            else :
                self.cached_data = torch.load(f'{ADReSS2020_DATAPATH}/preprocessed/logmel_delta_delta2_{self.wave_type}_{split}.pt', weights_only=False)

    def _mfcc_preprocess_dataset(self, split):
        """
        Preprocess the dataset in memory-efficient batches.
        """
        cached_data = []
        save_path = f'{ADReSS2020_DATAPATH}/preprocessed/mfcc_{self.wave_type}_{split}.pt'

        # Process paths in batches
        batch_size = 50  # Adjust this based on available memory
        num_batches = (len(self.cached_paths) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc=f'Preprocessing {split} dataset'):
            batch_paths = self.cached_paths[i * batch_size:(i + 1) * batch_size]

            batch_features = []
            batch_labels = []

            for path, label in batch_paths:
                # Load audio waveform
                waveform, sample_rate = torchaudio.load(path)
                waveform = waveform.numpy()

                # Apply augmentations
                augmented = [
                    waveform,
                    add_noise(waveform),
                    pitch_shift(waveform, sample_rate),
                    time_stretch(waveform, 0.9),
                    time_stretch(waveform, 1.1),
                ]

                for aug_data in augmented:
                    # Pad the waveform
                    padded_waveform = pad_audio(torch.tensor(aug_data), sample_rate, self.max_time).numpy()

                    # Extract features
                    features = extract_features(padded_waveform, sample_rate)
                    silence_percentage = calculate_silence_percentage(padded_waveform, sample_rate)

                    # Combine features
                    combined_features = np.append(features, silence_percentage)

                    # Store in batch
                    batch_features.append(combined_features)
                    batch_labels.append(label)

            # Append batch data to cached_data
            cached_data.extend(zip(batch_features, batch_labels))

            # Clear intermediate variables to free memory
            del batch_features, batch_labels, batch_paths
            torch.cuda.empty_cache()  # Optional, for GPU memory management

        # Save preprocessed dataset to disk
        torch.save(cached_data, save_path)
        print(f'Preprocessed dataset saved to {save_path}')

        return cached_data
    
    def _logmel_delta_delta2(self, waveform, sample_rate):
        '''
        Compute the log-mel spectrogram, delta, and delta-delta features of the waveform.
        
        Args:
            waveform: Audio waveform
            sample_rate: Sampling rate of the audio waveform
            
        Returns: (Shape: 224 x 224 x 3)
            Log-mel spectrogram, delta, and delta-delta features 
        '''
        # Ensure waveform is a tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)

        waveform = waveform.float()

        # Calculate the required length for 224 time frames
        required_length = 224 * 1024 + 1024  # (num_frames - 1) * hop_length + n_fft

        # Pad or truncate the waveform to the required length
        if waveform.size(1) < required_length:
            waveform = torch.nn.functional.pad(waveform, (0, required_length - waveform.size(1)))
        else:
            waveform = waveform[:, :required_length]

        # Compute log-mel spectrogram
        # 224 Melbands, hop length equal to 1024, and a Hanning window 
        # output: (224 x 224)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                                n_mels=224,
                                                                hop_length=1024,
                                                                n_fft=2048,
                                                                window_fn=torch.hann_window)(waveform)

        log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

        # Compute delta and delta-delta features
        delta = torchaudio.transforms.ComputeDeltas()(log_mel_spectrogram)
        delta_delta = torchaudio.transforms.ComputeDeltas()(delta)

        # Combine features
        features = torch.stack([log_mel_spectrogram, delta, delta_delta], dim=-1)
        
        return features
    
    def _logmel_delta_delta2_preprocess_dataset(self, split):
        """
        Preprocess the dataset in memory-efficient batches.
        """
        cached_data = []
        save_path = f'{ADReSS2020_DATAPATH}/preprocessed/logmel_delta_delta2_{self.wave_type}_{split}.pt'

        # Process paths in batches
        batch_size = 8
        num_batches = (len(self.cached_paths) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc=f'Preprocessing {split} dataset'):
            batch_paths = self.cached_paths[i * batch_size:(i + 1) * batch_size]

            batch_features = []
            batch_labels = []

            for path, label in batch_paths:
                # Load audio waveform
                waveform, sample_rate = torchaudio.load(path)

                features = self._logmel_delta_delta2(waveform, sample_rate)

                # Store in batch
                batch_features.append(features)
                batch_labels.append(label)

            # Append batch data to cached_data
            cached_data.extend(zip(batch_features, batch_labels))

            # Clear intermediate variables to free memory
            del batch_features, batch_labels, batch_paths
            torch.cuda.empty_cache()  # Optional, for GPU memory management

        # Save preprocessed dataset to disk
        torch.save(cached_data, save_path)
        print(f'Preprocessed dataset saved to {save_path}')

        return cached_data


    def _cache_paths(self):
        cached_paths = []
        for idx, row in self.df.iterrows():
            id = row[self.ID].strip()
            label = row[self.LABEL]
            path = self._get_path(label)

            if self.wave_type == 'full':
                file_path = os.path.join(path, f'{id}.wav')
            elif self.wave_type == 'chunk':
                # For chunk, we assume the paths are already available in the DataFrame
                file_path = row['Path']

            if os.path.exists(file_path):
                cached_paths.append((file_path, label))

        return cached_paths

    def __len__(self):
        if self.mfcc:
            return len(self.cached_data)
        else:
            return len(self.cached_paths)

    def __getitem__(self, idx):
        if self.mfcc or self.mel_delta_delta2:
            waveform, label = self.cached_data[idx]
            waveform = torch.tensor(waveform, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.long)

        else:
            path, label = self.cached_paths[idx]
            waveform, sample_rate = torchaudio.load(path)
            waveform = pad_audio(waveform, sample_rate, self.max_time)
            waveform = waveform.squeeze(0)

        if self.transforms:
            waveform = self.transforms(waveform)

        waveform = waveform.unsqueeze(-1)

        return waveform, label

    def _get_path(self, label):
        if self.split == 'train':
            AD_FOLDER = 'cc' if label == 0 else 'cd'
        elif self.split == 'test':
            AD_FOLDER = ''

        path = self.fullwave_path if self.wave_type == 'full' else self.chunkwave_path
        if AD_FOLDER:
            path = os.path.join(path, AD_FOLDER)

        return path

    def _get_df_chunks(self):
        df_chunks = []
        for id, label in zip(self.df['ID   '], self.df['Label ']):
            id = id.strip()
            path = self._get_path(label)

            for file_name in os.listdir(path):
                if file_name.endswith('.wav') and id in file_name:
                    file_path = os.path.join(path, file_name)
                    if os.path.exists(file_path):
                        df_chunks.append({'ID': id, 'Label': label, 'Path': file_path})
        return pd.DataFrame(df_chunks)

def load_data(data_name='ADReSS2020'):
    """Loads data from a CSV file.

    Args:
        data_name (str): Name of the dataset to load.

    Returns:
        train_df, test_df: Pandas DataFrames containing the training and testing data.
    """

    if data_name == 'ADReSS2020':
        # Train data
        train_AD_data = pd.read_csv(ADReSS2020_TRAIN_PATH + AD_data_txt, delimiter=';', skipinitialspace=True)
        train_AD_data['Label '] = 1
        train_AD_data = train_AD_data.drop(columns=['age', 'mmse', 'gender '], axis=1)

        train_NAD_data = pd.read_csv(ADReSS2020_TRAIN_PATH + NAD_data_txt, delimiter=';', skipinitialspace=True)
        train_NAD_data['Label '] = 0
        train_NAD_data = train_NAD_data.drop(columns=['age', 'mmse', 'gender '], axis=1)

        train_df = pd.concat([train_AD_data, train_NAD_data], ignore_index=True)

        # Test data
        test_df = pd.read_csv(ADReSS2020_DATAPATH + '/2020Labels.txt', delimiter=';', skipinitialspace=True)
        test_df = test_df.drop(columns=['age', 'mmse', 'gender'], axis=1)

    return train_df, test_df

def create_data_loaders(train_df, test_df, wave_type='full', feature_type='mfcc', batch_size=32, data_name='ADReSS2020'):
    """Creates PyTorch DataLoaders for training and testing sets.

    Args:
        train_df (pd.DataFrame): Pandas DataFrame containing the training data.
        test_df (pd.DataFrame): Pandas DataFrame containing the testing data.
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
        train_ds = ADreSS2020Dataset(train_df, split='train', wave_type=wave_type, feature_type=feature_type)
        test_ds = ADreSS2020Dataset(test_df, split='test', wave_type=wave_type, feature_type=feature_type)

    val_ds = random_split(train_ds, [0.8, 0.2], generator=generator)[1]
    train_ds = random_split(train_ds, [0.8, 0.2], generator=generator)[0]

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    start = time.time()

    train_df, test_df = load_data()
    print("Load data successful!")

    train_loader, val_loader, test_loader = create_data_loaders(train_df, test_df, wave_type='chunk', feature_type='mel_delta_delta2')
    # train_loader, val_loader, test_loader = create_data_loaders(train_df, test_df, wave_type='chunk', feature_type='mfcc')
    train_loader, val_loader, test_loader = create_data_loaders(train_df, test_df, wave_type='full', feature_type='mel_delta_delta2')
    # train_loader, val_loader, test_loader = create_data_loaders(train_df, test_df, wave_type='full', feature_type='mfcc')
    print("DataLoaders created successfully!")

    print("Time taken: ", f'{time.time() - start:.2f} seconds')
    start = time.time()

    print("Train data: ", len(train_loader.dataset))
    print("Validation data: ", len(val_loader.dataset))
    print("Test data: ", len(test_loader.dataset))

    print("Sample data: ", next(iter(train_loader))[0].shape)
    print("Get data's time taken: ", f'{time.time() - start:.2f} seconds')
