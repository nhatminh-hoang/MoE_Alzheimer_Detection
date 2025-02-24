import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from tqdm import tqdm

import tiktoken
from transformers import AutoTokenizer, AutoModelForMaskedLM, ModernBertModel
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
    def __init__(self, data_path, data_type, split, 
                 text_type='full',
                 wave_type='full', feature_type='mfcc'):
        self.data_path = data_path
        self.data_type = data_type
        self.split = split
        self.text_type = text_type
        self.wave_type = wave_type
        self.feature_type = feature_type

        preprocess_path = ADReSS2020_DATAPATH + '/preprocessed/'
        X_name = f'X_{self.split}.npy'
        y_name = f'y_{self.split}.npy'

        # Load data
        if data_type == 'audio':
            self.audio_files, self.labels = self._load_audio_data()

        if data_type == 'text':
            X_name = 'text_' + X_name
            y_name = 'text_' + y_name

            self.text_data = self._load_text_data()
            if os.path.exists(preprocess_path + X_name) and os.path.exists(preprocess_path + y_name):
                self.preprocess_text = np.load(preprocess_path + X_name), np.load(preprocess_path + y_name)
            else:
                self.preprocess_text = self._preprocess_text_data()

        # Preprocess data
        if data_type == 'audio' and self.feature_type == 'mfcc':
            X_name = 'mfcc_' + X_name
            y_name = 'mfcc_' + y_name

            if os.path.exists(preprocess_path + X_name) and os.path.exists(preprocess_path + y_name):
                self.preprocess_audio = np.load(preprocess_path + X_name), np.load(preprocess_path + y_name)
            else:
                self.preprocess_audio = self._preprocess_mfcc(self.audio_files, self.labels)

        elif data_type == 'audio':
            self.preprocess_audio = prepare_test_data(self.audio_files, self.labels)            

    def __len__(self):
        if self.data_type == 'audio':
            return len(self.preprocess_audio[0])
        elif self.data_type == 'text':
            return len(self.text_data)
    
    def __getitem__(self, idx):
        if self.data_type == 'audio':
            return  np.expand_dims(self.preprocess_audio[0][idx], -1).astype(np.float32), \
                    self.preprocess_audio[1][idx]
        
        elif self.data_type == 'text':
            if self.text_type == 'full':
                return self.preprocess_text[0][idx], self.preprocess_text[1][idx]
            elif self.text_type == 'sum':
                return np.sum(self.preprocess_text[0][idx], axis=0), self.preprocess_text[1][idx]
    
    def _load_audio_data(self):
        train_audio_files, train_labels, test_audio_files, test_labels = load_audio_data(data_name='ADReSS2020')
        if self.split == 'train':
            audio_files = train_audio_files
            labels = train_labels
        elif self.split == 'test':
            audio_files = test_audio_files
            labels = test_labels

        return audio_files, labels
    
    def _load_text_data(self):
        return get_chat_data(self.split)
    
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
    
    def _preprocess_text_data(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_id = "answerdotai/ModernBERT-base"
        tokenizer = AutoTokenizer.from_pretrained('./models/ModernBERT-base_tokenizer')
        bert_model = ModernBertModel.from_pretrained('./models/ModernBERT-base_model').to(device)
        bert_model.eval()
        
        preprocessed_text_data = []
        
        for idx in range(len(self.text_data)):
            inputs = " ".join(self.text_data.iloc[idx]['tokens'])
            inputs_token = tokenizer(inputs, 
                                     return_tensors='pt',
                                     truncation=True, 
                                     padding='max_length', 
                                     max_length=300).to(device)
            outputs = bert_model(**inputs_token).last_hidden_state.squeeze(0).cpu().detach().numpy()
            preprocessed_text_data.append(outputs)
        print(len(preprocessed_text_data))
        X = np.array(preprocessed_text_data)
        y = self.text_data['label'].to_numpy()
        
        # Save the preprocessed data
        if self.split == 'train':
            np.save(ADReSS2020_DATAPATH + '/preprocessed/' + 'text_X_train.npy', X)
            np.save(ADReSS2020_DATAPATH + '/preprocessed/' + 'text_y_train.npy', y)
        elif self.split == 'test':
            np.save(ADReSS2020_DATAPATH + '/preprocessed/' + 'text_X_test.npy', X)
            np.save(ADReSS2020_DATAPATH + '/preprocessed/' + 'text_y_test.npy', y)

        return preprocessed_text_data

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
    
    assert len(train_audio_files) == len(train_labels) and len(test_audio_files) == len(test_labels), "Data and labels do not match!"
    assert len(train_audio_files) > 0 and len(test_audio_files) > 0, "No data loaded!"

    print("Load data successful!")
    return train_audio_files, train_labels, test_audio_files, test_labels

def create_audio_data_loaders(data_type='audio', 
                              data_name='ADReSS2020',
                              wave_type='full', feature_type='mfcc',
                              batch_size=32):
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
        train_ds = ADreSS2020Dataset(ADReSS2020_DATAPATH, data_type=data_type, split='train', wave_type=wave_type, feature_type=feature_type)
        test_ds = ADreSS2020Dataset(ADReSS2020_DATAPATH, data_type=data_type, split='test', wave_type=wave_type, feature_type=feature_type)

    # val_ds, train_ds = random_split(train_ds, [0.2, 0.8], generator=generator)
    val_ds = test_ds

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def create_text_data_loaders(data_type='text', 
                             data_name='ADReSS2020',
                             text_type='full',
                             batch_size=32):
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
        train_ds = ADreSS2020Dataset(ADReSS2020_DATAPATH, data_type=data_type, split='train', text_type=text_type)
        test_ds = ADreSS2020Dataset(ADReSS2020_DATAPATH, data_type=data_type, split='test', text_type=text_type)

    # val_ds, train_ds = random_split(train_ds, [0.2, 0.8], generator=generator)
    val_ds = test_ds

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def create_dataloader(data_name, data_type, batch_size=32, 
                      text_type='full',
                      feature_type='mfcc', wave_type='full'):
    """Creates PyTorch DataLoader for the specified dataset.

    Args:
        data_name (str): Name of the dataset to load.
        data_type (str): Type of data to load (e.g., 'train', 'test').
        batch_size (int): Batch size for the DataLoader.
        text_type (str): Type of text feature to use (e.g., 'full', 'sum').
        feature_type (str): Type of feature to use (e.g., 'mfcc', 'spectrogram').
        wave_type (str): Type of audio waveform to use (e.g., 'full', 'chunk').

    Returns:
        torch.utils.data.DataLoader: DataLoader for the specified dataset.
    """
    # Load data

    if data_type == 'audio':
        train_loader, val_loader, test_loader = create_audio_data_loaders(data_type=data_type,
                                                                          data_name=data_name,
                                                                          feature_type=feature_type, 
                                                                          batch_size=batch_size, 
                                                                          wave_type=wave_type)
    elif data_type == 'text':
        train_loader, val_loader, test_loader = create_text_data_loaders(data_type=data_type,
                                                                         data_name=data_name,
                                                                         batch_size=batch_size,
                                                                         text_type=text_type)
    print("DataLoaders created successfully!")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    start = time.time()

    train_loader, val_loader, test_loader = create_dataloader('ADReSS2020', data_type='text', text_type='sum', batch_size=32)

    print("Time taken: ", f'{time.time() - start:.2f} seconds')
    start = time.time()

    print("Train data: ", len(train_loader.dataset))
    print("Validation data: ", len(val_loader.dataset))
    print("Test data: ", len(test_loader.dataset))

    print("Sample data: ", next(iter(train_loader))[0].shape)
    print("Get data's time taken: ", f'{time.time() - start:.2f} seconds')