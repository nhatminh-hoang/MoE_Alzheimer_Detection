# nlp_project

## Description

(Provide a description of your project here.)

## Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd nlp_project
    ```
2. Create a virtual environment (optional but recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data

-   See `data/README.md` for details about the data.

## Usage

### Data Preprocessing

```bash
python scripts/data_preprocessing.py
```

### Training

```bash
python scripts/train.py
```

### Evaluation

```bash
python scripts/evaluate.py
```

### Prediction

```bash
python scripts/predict.py
```

## Configuration

-   Modify configuration parameters in `config/default_config.yaml`.
-   Create different experiment configurations in `config/experiment_configs/`.

## HPC Usage (if applicable)

-   Provide instructions on how to run the project on an HPC environment here.

