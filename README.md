# ADReSS2020 Audio Classification

This repository contains code for the ADReSS2020 audio classification project. The goal of this project is to classify audio data using machine learning models.

## Project Structure
├── pycache/ 

├── .gitignore 

├── config/ 

│ ├── default_config.yaml 

│ ├── experiment_configs/ 

│ │ ├── ADReSS2020_MLP_mfcc.yaml 

│ │ ├── ADReSS2020_MLP_waveform.yaml 

├── data/ 

│ ├── ADReSS-IS2020-data/ 

├── data_preprocessing.py 

├── evaluate.py 

├── logs/ 

│ ├── ADReSS2020_MLP_mfcc/ 

│ ├── ADReSS2020_MLP_waveform/ 

├── models/ 

│ ├── __init__.py 

│ ├── base_model.py 

│ ├── my_custom_model.py 

│ ├── saved_models/ 

├── notebooks/ 

│ ├── exploration.ipynb 

│ ├── model_development.ipynb 

├── predict.py 

├── README.md 

├── requirements.txt 

├── scripts/ 

│ ├── __init__.py 

│ ├── utils.py 

├── train.py


## Setup

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Configuration

Configuration files are located in the [config](http://_vscodecontentref_/17) directory. The [default_config.yaml](http://_vscodecontentref_/18) file contains the default configuration parameters. Experiment-specific configurations are stored in the [experiment_configs](http://_vscodecontentref_/19) directory.

## Data Preprocessing

To preprocess the data, run the [data_preprocessing.py](http://_vscodecontentref_/20) script:
```sh
python data_preprocessing.py
```

## Training
To train a model, run the train.py script with the desired configuration:

```sh
python train.py --model MLP --epochs 100 --batch_size 128 --lr 0.001 --mfcc True
```

## Evaluation
To evaluate a trained model, run the ```evaluate.py``` script:
```sh
python evaluate.py
```

## Prediction
To make predictions using a trained model, run the ```predict.py``` script:
```sh
python predict.py
```

## Notebooks
The notebooks directory contains Jupyter notebooks for data exploration and model development:

- ```exploration.ipynb```: Data exploration and visualization.
- ```model_development.ipynb```: Model development and testing.

## License
This project is licensed under the MIT License.
