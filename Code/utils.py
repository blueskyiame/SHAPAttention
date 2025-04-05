
# utils.py
"""
Utility functions for SHAPAttention.
"""
import torch
import random
import numpy as np
import shap
import matplotlib.pyplot as plt


def set_seed(seed):
    """
    Set seed for reproducibility.

    Args:
        seed (int): Random seed
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_shap_values(model, background_data, test_data):
    """
    Calculate SHAP values for test data.

    Args:
        model: The model to use
        background_data: Background data for SHAP
        test_data: Test data to explain

    Returns:
        numpy.ndarray: SHAP values
    """
    model.eval()
    explainer = shap.DeepExplainer(model.cpu(), background_data)

    try:
        # Try both approaches depending on the environment
        shap_values = explainer.shap_values(test_data)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Take first output for single output models
    except:
        shap_values = explainer.shap_values(test_data).squeeze(2)

    return shap_values


def plot_shap_values(shap_values, feature_names=None, title="SHAP Values", save_path=None):
    """
    Plot SHAP values.

    Args:
        shap_values: SHAP values to plot
        feature_names: Names of features
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, feature_names=feature_names)
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
    plt.show()

