# SHAPAttention Tutorial

This repository contains code for spectroscopy data analysis using a Convolutional Neural Network (CNN) model enhanced with SHAP (SHapley Additive exPlanations) values. The model is designed for spectroscopy data analysis with a focus on improving prediction accuracy through attention mechanisms.

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [Dataset Preparation](#dataset-preparation)
6. [Reproducing Results](#reproducing-results)
7. [Model Architecture](#model-architecture)
8. [Custom Training Parameters](#custom-training-parameters)

## Installation

```bash
# Clone the repository
git clone https://github.com/blueskyiame/SHAPAttention.git
cd SHAPAttention

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
SHAPAttention/
├── main.py                # Main script to run the analysis
├── model.py               # Model architecture definitions
├── preprocessing.py       # Data preprocessing functions
├── train.py               # Training and validation functions
├── utils.py               # Utility functions
├── config.py              # Configuration parameters
├── requirements.txt       # Dependencies
└── data/                  # Directory for datasets
    ├── Cal_ManufacturerB.xlsx
    ├── Ramandata_tablets.xlsx
    └── 高光谱1.xlsx
```

## Configuration

You can configure the model and training parameters by editing the `config.py` file:

```python
# Dataset selection
DATASET = "Cal_ManufacturerB"  # Options: "Cal_ManufacturerB", "Ramandata_tablets", "高光谱1"

# Model parameters
MODEL_PARAMS = {
    "Cal_ManufacturerB": {
        "hidden_size": 1024,
        "learning_rate": 1e-4,
    },
    "Ramandata_tablets": {
        "hidden_size": 1024,
        "learning_rate": 1e-4,
    },
    "高光谱1": {
        "hidden_size": 1024,
        "learning_rate": 1e-4,
    }
}

# Common parameters
COMMON_PARAMS = {
    "output_size": 1,
    "num_epochs": 4000,
    "batch_size": 64,
    "test_batch_size": 128,
    "dropout_rate": 0.2,
    "shap_update_frequency": 400,
    "weight_decay": 0.001,
    "patience": 1500,
    "random_seed": 100,
    "test_size": 0.2
}
```

## Usage

Run the main script to start the analysis:

```bash
python main.py
```

For custom parameters, you can also pass arguments:

```bash
python main.py --dataset Cal_ManufacturerB --hidden_size 512 --batch_size 32 --num_epochs 2000
```

## Dataset Preparation

The code expects the following datasets:

1. **Cal_ManufacturerB.xlsx**: Contains NIR spectra and protein content
2. **Ramandata_tablets.xlsx**: Contains Raman spectra and PE content
3. **高光谱1.xlsx**: Contains average hyperspectral data

Place these files in the `data/` directory. Each dataset should have:
- A target column ('Protein' for Cal_ManufacturerB, 'PE' for Ramandata_tablets)
- Spectral data columns

## Reproducing Results

To reproduce the results reported in the manuscript:

1. Ensure all datasets are in the `data/` directory
2. Use the default parameters in `config.py`
3. Run the main script:

```bash
python main.py
```

The results will be saved in Excel files with the naming pattern:
`Epoch {num_epochs}_frequency{shap_update_frequency}_val_loss_{dataset_name}_outputs.xlsx`

## Model Architecture

The model consists of:
1. **Convolutional layers**: Extract features from spectral data
2. **SHAP Attention mechanism**: Use SHAP values to enhance feature importance
3. **Fully connected layers**: Final prediction

The architecture is suitable for spectral data with varying input sizes.

## Custom Training Parameters

You can customize the training parameters in several ways:

1. **Edit config.py**: For permanent changes
2. **Command line arguments**: For one-time runs
3. **Direct code modification**: For advanced customization

Example of command line usage:

```bash
python main.py --dataset Ramandata_tablets --hidden_size 1024 --learning_rate 1e-5 --batch_size 32 --num_epochs 2000 --dropout_rate 0.1 --shap_update_frequency 200
```

For advanced users who want to tune the model architecture, edit `model.py` to modify the CNN layers or the SHAP attention mechanism.