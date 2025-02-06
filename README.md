# ADReSS2020 Audio Classification

This repository contains code for the ADReSS2020 audio classification project. The goal of this project is to classify audio data using machine learning models to detect Alzheimer's disease from audio recordings.

## Project Structure

```
├── __pycache__/
├── .gitignore
├── config/
│   └── experiment_configs/
│       └── ... (YAML configuration files for different experiments)
├── data/
│   └── ADReSS-IS2020-data/ (Contains the audio data and metadata)
├── 

data_preprocessing.py

 (Handles data loading, preprocessing, and DataLoader creation)
├── 

evaluate.py

 (Evaluates trained models on the test set)
├── logs/
│   ├── ADReSS2020_CNN_chunk_mfcc/ (Logs for CNN model trained on chunked audio with MFCC features)
│   ├── ADReSS2020_CNN_chunk_waveform/ (Logs for CNN model trained on chunked audio with waveform features)
│   ├── ADReSS2020_CNN_full_mfcc/ (Logs for CNN model trained on full audio with MFCC features)
│   ├── ADReSS2020_MLP_chunk_mfcc/ (Logs for MLP model trained on chunked audio with MFCC features)
│   ├── ADReSS2020_MLP_chunk_waveform/ (Logs for MLP model trained on chunked audio with waveform features)
│   └── ADReSS2020_MLP_full_mfcc/ (Logs for MLP model trained on full audio with MFCC features)
├── models/
│   ├── __init__.py
│   ├── base_model.py (Contains base model classes like MLP and CNN)
│   ├── my_custom_model.py (Example for custom model implementation)
│   └── saved_models/ (Directory to store trained model weights)
├── notebooks/
│   ├── 

exploration.ipynb

 (Jupyter Notebook for data exploration and visualization)
│   └── 

model_development.ipynb

 (Jupyter Notebook for model development and experimentation)
├── 

predict.py

 (Script for making predictions using a trained model)
├── 

README.md


├── 

requirements.txt

 (Lists the required Python packages)
├── scripts/
│   └── 

utils.py

 (Contains utility functions for training, logging, etc.)
└── 

train.py

 (Main script for training the models)
```

## Setup

1.  Clone the repository:

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Configuration

Experiment configurations are stored in the 

experiment_configs

 directory as YAML files. These files define the hyperparameters and settings for each training run.  Key parameters include:

*   

data_name

: Name of the dataset (e.g., `ADReSS2020`).
*   

wave_type

: Type of waveform used (`full` for full audio, `chunk` for smaller segments).
*   

feature_type

: Feature extraction method (`MFCC` for Mel-Frequency Cepstral Coefficients, `LogmelDelta` for Log-Mel spectrogram with deltas, or 

waveform

 for raw audio).
*   

model

: Model architecture (

MLP

 for Multilayer Perceptron, 

CNN

 for Convolutional Neural Network).
*   

epochs

: Number of training epochs.
*   

batch_size

: Batch size for training.  Adjust based on available GPU memory.
*   

lr

: Learning rate for the optimizer.
*   

hidden_size

: Size of the hidden layers in the model.
*   `dropout`: Dropout rate for regularization.
*   

early_stop

: Number of epochs to wait for improvement before early stopping.
*   

flatten

: Boolean flag to flatten the input features (relevant for some model types).

## Data Preprocessing

To preprocess the audio data and create DataLoaders, run the 

data_preprocessing.py

 script:

```sh
python data_preprocessing.py
```

This script loads the dataset, extracts audio features (MFCC, Log-Mel spectrogram, or raw waveform), and creates PyTorch DataLoaders for training, validation, and testing.  The 

wave_type

 and 

feature_type

 are specified within the script.

## Training

To train a model, run the 

train.py

 script with the desired configuration.  You can specify the training parameters using command-line arguments.  For example:

```sh
python train.py --data_name ADReSS2020 --wave_type full --feature_type MFCC --model MLP --epochs 100 --batch_size 128 --lr 0.001 --hidden_size 128 --dropout 0.5 --early_stop 10
```

This command trains an MLP model using MFCC features extracted from the full audio waveforms. The training will run for 100 epochs with a batch size of 128 and a learning rate of 0.001. Early stopping is enabled, and training will halt if the validation loss doesn't improve for 10 epochs. The script saves the best model weights based on validation loss to the 

saved_models

 directory and logs training progress to the 

logs

 directory.

The 

train.py

 script uses the following key components:

*   

MODEL

 dictionary: Maps model names (e.g., 

MLP

, 

CNN

) to their corresponding classes defined in 

base_model.py

.
*   

training()

 function: Performs one epoch of training.
*   

testing()

 function: Evaluates the model on the validation or test set.
*   

fit()

 function: Orchestrates the training loop, including forward and backward passes, optimization, and early stopping.
*   

evaluate()

 function: Evaluates the trained model on the test set and saves evaluation metrics.
*   

save_model_summary()

 function: Saves a summary of the model architecture to a text file in the logs directory, using the `torchinfo` library.
*   

save_training_images()

 function: Saves plots of the training and validation loss and accuracy curves to the logs directory.
*   

save_lr_plot()

 function: Saves a plot of the learning rate schedule to the logs directory.

## Evaluation

To evaluate a trained model on the test set, run the 

evaluate.py

 script:

```sh
python evaluate.py
```

This script loads the trained model specified in the configuration file and evaluates its performance on the test dataset. It calculates and saves evaluation metrics such as accuracy, precision, recall, and F1-score to the 

logs

 directory.

## Prediction

To make predictions on new audio data using a trained model, run the 

predict.py

 script:

```sh
python predict.py
```

This script loads a trained model and uses it to classify new audio samples.

## Notebooks

The 

notebooks

 directory contains Jupyter notebooks for data exploration and model development:

*   

exploration.ipynb

: Provides an overview of the dataset, including data distributions and audio characteristics.
*   

model_development.ipynb

:  A notebook for experimenting with different model architectures and training strategies.

## Logs

Training logs, model summaries, evaluation metrics, and plots are saved in the 

logs

 directory. Each experiment has its own subdirectory, named according to the model, wave type, and feature type used (e.g., `ADReSS2020_MLP_full_mfcc`).

## License

This project is licensed under the MIT License.