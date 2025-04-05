import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
# Scale the data
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Create model
from SHAPAttention_1DCNN_HYPERVIEW_CosineAnnealingLR6 import Conv1DCNN  # Import your model class
# Create dataset and dataloader
from torch.utils.data import DataLoader
from SHAPAttention_1DCNN_HYPERVIEW_CosineAnnealingLR6 import NIRDataset  # Import your dataset class

def model_predict_CAL(model, data_loader, scaler_Y, device, model_path, save_results=True):
    """
    Load a trained model and evaluate it on the provided data loader

    Args:
        model: The PyTorch model architecture
        data_loader: DataLoader containing the data to predict
        scaler_Y: The scaler used to transform target values
        device: Device to run the model on (cuda or cpu)
        model_path: Path to the trained model weights
        save_results: Whether to save results to Excel

    Returns:
        Tuple of (predictions, actual_values, metrics_dict)
    """
    # Load the best model weights
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Initialize lists to store predictions and actual values
    all_predictions = []
    all_actual = []

    # Make predictions
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # No SHAP values for prediction

            # Store batch results
            all_predictions.extend(outputs.cpu().numpy())
            all_actual.extend(labels.cpu().squeeze(1).numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_actual = np.array(all_actual)

    # Inverse transform to original scale
    all_predictions_original = scaler_Y.inverse_transform(all_predictions)
    all_actual_original = scaler_Y.inverse_transform(all_actual)

    # Calculate metrics for each output
    metrics = {
        'r2': [],
        'rmse': [],
        'rpd': []
    }

    # Create a dictionary to store results for Excel
    excel_data = {
        'Actual_1': all_actual_original[:, 0],
        'Predicted_1': all_predictions_original[:, 0],
        'Actual_2': all_actual_original[:, 1],
        'Predicted_2': all_predictions_original[:, 1],
        'Actual_3': all_actual_original[:, 2],
        'Predicted_3': all_predictions_original[:, 2],
        'Actual_4': all_actual_original[:, 3],
        'Predicted_4': all_predictions_original[:, 3]
    }

    # Calculate and print metrics for each output
    print(f"\nPerformance Metrics for Model: {model_path}")
    print("-" * 60)

    for i in range(all_actual_original.shape[1]):
        # Calculate metrics
        r2 = r2_score(all_actual_original[:, i], all_predictions_original[:, i])
        rmse_val = np.sqrt(mean_squared_error(all_actual_original[:, i], all_predictions_original[:, i]))
        rpd_val = np.std(all_actual_original[:, i]) / rmse_val

        # Store metrics
        metrics['r2'].append(r2)
        metrics['rmse'].append(rmse_val)
        metrics['rpd'].append(rpd_val)

        # Print results
        print(f"Output {i + 1}:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse_val:.4f}")
        print(f"  RPD: {rpd_val:.4f}")
        print("-" * 40)

    # Save results to Excel if requested
    if save_results:
        # Add metrics to the Excel data
        for i in range(all_actual_original.shape[1]):
            excel_data[f'R2_Output_{i + 1}'] = [metrics['r2'][i]] + [np.nan] * (len(excel_data['Actual_1']) - 1)
            excel_data[f'RMSE_Output_{i + 1}'] = [metrics['rmse'][i]] + [np.nan] * (len(excel_data['Actual_1']) - 1)
            excel_data[f'RPD_Output_{i + 1}'] = [metrics['rpd'][i]] + [np.nan] * (len(excel_data['Actual_1']) - 1)

        # Create DataFrame and save to Excel
        results_df = pd.DataFrame(excel_data)
        model_name = os.path.basename(model_path).replace('.pth', '')
        results_df.to_excel(f'prediction_results_{model_name}.xlsx', index=False)
        print(f"Results saved to prediction_results_{model_name}.xlsx")

        # Create a summary DataFrame with just the metrics
        summary_data = {
            'Output': [f'Output_{i + 1}' for i in range(all_actual_original.shape[1])],
            'R2': metrics['r2'],
            'RMSE': metrics['rmse'],
            'RPD': metrics['rpd']
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(f'metrics_summary_{model_name}.xlsx', index=False)
        print(f"Metrics summary saved to metrics_summary_{model_name}.xlsx")

    # Create scatter plots for each output
    plot_prediction_scatter(all_actual_original, all_predictions_original, model_path)

    return all_predictions_original, all_actual_original, metrics

def model_predict_VAL(model, data_loader, scaler_Y, device, model_path, save_results=True):
    """
    Load a trained model and evaluate it on the provided data loader

    Args:
        model: The PyTorch model architecture
        data_loader: DataLoader containing the data to predict
        scaler_Y: The scaler used to transform target values
        device: Device to run the model on (cuda or cpu)
        model_path: Path to the trained model weights
        save_results: Whether to save results to Excel

    Returns:
        Tuple of (predictions, actual_values, metrics_dict)
    """
    # Load the best model weights
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Initialize lists to store predictions and actual values
    all_predictions = []
    all_actual = []

    # Make predictions
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # No SHAP values for prediction

            # Store batch results
            all_predictions.extend(outputs.cpu().numpy())
            all_actual.extend(labels.cpu().squeeze(1).numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_actual = np.array(all_actual)

    # Inverse transform to original scale
    all_predictions_original = scaler_Y.inverse_transform(all_predictions)
    all_actual_original = scaler_Y.inverse_transform(all_actual)

    # Calculate metrics for each output
    metrics = {
        'r2': [],
        'rmse': [],
        'rpd': []
    }

    # Create a dictionary to store results for Excel
    excel_data = {
        'Actual_1': all_actual_original[:, 0],
        'Predicted_1': all_predictions_original[:, 0],
        'Actual_2': all_actual_original[:, 1],
        'Predicted_2': all_predictions_original[:, 1],
        'Actual_3': all_actual_original[:, 2],
        'Predicted_3': all_predictions_original[:, 2],
        'Actual_4': all_actual_original[:, 3],
        'Predicted_4': all_predictions_original[:, 3]
    }

    # Calculate and print metrics for each output
    print(f"\nPerformance Metrics for Model: {model_path}")
    print("-" * 60)

    for i in range(all_actual_original.shape[1]):
        # Calculate metrics
        r2 = r2_score(all_actual_original[:, i], all_predictions_original[:, i])
        rmse_val = np.sqrt(mean_squared_error(all_actual_original[:, i], all_predictions_original[:, i]))
        rpd_val = np.std(all_actual_original[:, i]) / rmse_val

        # Store metrics
        metrics['r2'].append(r2)
        metrics['rmse'].append(rmse_val)
        metrics['rpd'].append(rpd_val)

        # Print results
        print(f"Output {i + 1}:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse_val:.4f}")
        print(f"  RPD: {rpd_val:.4f}")
        print("-" * 40)

    # Save results to Excel if requested
    if save_results:
        # Add metrics to the Excel data
        for i in range(all_actual_original.shape[1]):
            excel_data[f'R2_Output_{i + 1}'] = [metrics['r2'][i]] + [np.nan] * (len(excel_data['Actual_1']) - 1)
            excel_data[f'RMSE_Output_{i + 1}'] = [metrics['rmse'][i]] + [np.nan] * (len(excel_data['Actual_1']) - 1)
            excel_data[f'RPD_Output_{i + 1}'] = [metrics['rpd'][i]] + [np.nan] * (len(excel_data['Actual_1']) - 1)

        # Create DataFrame and save to Excel
        results_df = pd.DataFrame(excel_data)
        model_name = os.path.basename(model_path).replace('.pth', '')
        results_df.to_excel(f'prediction_results_{model_name}.xlsx', index=False)
        print(f"Results saved to prediction_results_{model_name}.xlsx")

        # Create a summary DataFrame with just the metrics
        summary_data = {
            'Output': [f'Output_{i + 1}' for i in range(all_actual_original.shape[1])],
            'R2': metrics['r2'],
            'RMSE': metrics['rmse'],
            'RPD': metrics['rpd']
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(f'metrics_summary_{model_name}.xlsx', index=False)
        print(f"Metrics summary saved to metrics_summary_{model_name}.xlsx")

    # Create scatter plots for each output
    plot_prediction_scatter(all_actual_original, all_predictions_original, model_path)

    return all_predictions_original, all_actual_original, metrics

def plot_prediction_scatter(actual, predicted, model_path):
    """
    Create scatter plots of predicted vs actual values for each output

    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        model_path: Path to the model (used for plot title)
    """
    model_name = os.path.basename(model_path).replace('.pth', '')

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]

        # Calculate metrics for display
        r2 = r2_score(actual[:, i], predicted[:, i])
        rmse = np.sqrt(mean_squared_error(actual[:, i], predicted[:, i]))
        rpd = np.std(actual[:, i]) / rmse

        # Create scatter plot
        ax.scatter(actual[:, i], predicted[:, i], alpha=0.6)

        # Add perfect prediction line
        min_val = min(np.min(actual[:, i]), np.min(predicted[:, i]))
        max_val = max(np.max(actual[:, i]), np.max(predicted[:, i]))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')

        # Add metrics as text
        text_str = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nRPD = {rpd:.4f}'
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Labels and title
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Output {i + 1}')

        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.suptitle(f'Prediction Results - {model_name}', y=1.02, fontsize=16)
    plt.savefig(f'prediction_scatter_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scatter plots saved to prediction_scatter_{model_name}.png")


# Example usage in main function
def main():

    print("-" * 20)
    print("TEST_CAL")
    print("-" * 20)
    # Set random seeds for reproducibility
    torch.manual_seed(100)
    np.random.seed(100)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your data as in your original code
    data = pd.read_excel("HYPERVIEW_train.xlsx", index_col=0)
    y = data.iloc[:, 0:4].values
    X = data.iloc[:, 4:].values


    scaler_Y = MinMaxScaler(feature_range=(0, 10))
    y = scaler_Y.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    all_dataset = NIRDataset(X_scaled, y)
    all_loader = DataLoader(all_dataset, batch_size=64, shuffle=False)  # No shuffle for prediction

    input_size = X.shape[1]
    hidden_size = 256
    output_size = 4
    model = Conv1DCNN(input_size, hidden_size, output_size).to(device)

    # Predict using your best model
    model_path = 'best_model_fold_5_epoch2.pth'
    predictions, actual, metrics = model_predict_CAL(model, all_loader, scaler_Y, device, model_path)

    baseline_predictions = actual.mean(axis=0)

    baselines = np.mean((actual - baseline_predictions) ** 2, axis=0)

    mse = np.mean((predictions - baseline_predictions) ** 2, axis=0)

    scores = mse / baselines
    # Calculate the final score
    final_score = np.mean(scores)

    print("final_score：", final_score)
    ''''''
    print("-" * 20)
    print("TEST_VAL")
    print("-" * 20)

    data = pd.read_excel("HYPERVIEW_test.xlsx")
    X = data.iloc[:, :].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = torch.FloatTensor(X_scaled)
    # Make predictions
    with torch.no_grad():
        outputs = model(X_scaled.to(device))
    outputs = np.array(outputs.cpu().numpy())

    # Inverse transform to original scale
    outputs = scaler_Y.inverse_transform(outputs)

    outputs_df = pd.DataFrame(outputs)

    # 保存为 Excel 文件
    outputs_df.to_excel('predict_testsets.xlsx', index=False)






if __name__ == "__main__":
    main()