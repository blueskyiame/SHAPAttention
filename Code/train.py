
# train.py
"""
Training and validation functions for SHAPAttention.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import random
import copy
import shap
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Dataset, DataLoader, TensorDataset

class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = (max_lr - base_lr) / warmup_epochs

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr + epoch * self.step_size
        else:
            # If using another scheduler, call it here
            pass
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def validate_model(model, val_loader, criterion, device, shap_values):
    """
    Validate the model on the validation set.

    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run on (cpu or cuda)
        shap_values: SHAP values for the validation data

    Returns:
        tuple: labels, outputs, val_loss, r2, rmse, rpd
    """
    model.eval()
    val_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            if shap_values is not None:
                start_idx = batch_idx * inputs.size(0)
                end_idx = start_idx + inputs.size(0)
                batch_shap = torch.FloatTensor(shap_values[start_idx:end_idx]).to(device)
                assert batch_shap.shape == inputs.shape
            else:
                batch_shap = None

            outputs = model(inputs, batch_shap)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)

    r2 = r2_score(all_labels, all_outputs)
    rmse = np.sqrt(mean_squared_error(all_labels, all_outputs))
    rpd = np.std(all_labels) / rmse  # Calculate RPD

    return all_labels, all_outputs, val_loss / len(val_loader.dataset), r2, rmse, rpd


def update_shap_values(model, train_loader, device, num_background=400):
    """
    Update SHAP values for the model.

    Args:
        model: The model to use
        train_loader: DataLoader for training data
        device: Device to run on (cpu or cuda)
        num_background (int): Number of background samples

    Returns:
        numpy.ndarray: SHAP values
    """
    model.eval()
    background = []
    model_cpu = copy.deepcopy(model).cpu()
    for inputs, _ in train_loader:
        background.append(inputs)
        if len(background) * inputs.size(0) >= num_background:
            break
    background = torch.cat(background)[:num_background]

    # Create a DeepExplainer
    explainer = shap.DeepExplainer(model_cpu, background)

    # Calculate SHAP values for all training data
    all_shap_values = []
    for inputs, _ in train_loader:
        inputs = inputs
        try:
            # Try both approaches depending on the environment
            batch_shap_values = explainer.shap_values(inputs)
            if isinstance(batch_shap_values, list):
                batch_shap_values = batch_shap_values[0]  # Take first output for single output models
        except:
            batch_shap_values = explainer.shap_values(inputs).squeeze(2)

        all_shap_values.append(batch_shap_values)

    all_shap_values = np.concatenate(all_shap_values, axis=0)
    model.to(device)

    return all_shap_values


def train_cv_model_with_shap_updates(model, train_loader, criterion,
                                     scheduler, warmup_scheduler, optimizer,
                                     learning_rate, num_epochs, device,
                                     shap_update_frequency=5, patience=1500,
                                     start_shap=200, dataset_name="dataset"):
    """
    Train model with cross-validation and periodic SHAP updates.

    Args:
        model: Model to train
        train_loader: DataLoader with all training data
        criterion: Loss function
        scheduler: Learning rate scheduler
        warmup_scheduler: Warmup scheduler for learning rate
        optimizer: Optimizer
        learning_rate: Learning rate
        num_epochs: Number of epochs to train
        device: Device to train on
        shap_update_frequency: How often to update SHAP values
        patience: Early stopping patience
        start_shap: Epoch to start SHAP updates
        dataset_name: Name of the dataset (for saving results)

    Returns:
        tuple: Trained model and final SHAP values
    """
    # Convert train_loader data to numpy arrays for cross-validation
    train_data = []
    train_labels = []
    for inputs, labels in train_loader:
        train_data.append(inputs.numpy())
        train_labels.append(labels.numpy())
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    # Create bins for stratified sampling
    y_binned = pd.qcut(train_labels.flatten(), q=5, labels=False)

    # Use stratified sampling for cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_val_losses = []
    fold_r2_scores = []
    fold_rmse_scores = []
    fold_rpd_scores = []

    for fold, (train_idx, cv_idx) in enumerate(kf.split(train_data, y_binned)):
        print(f"\nTraining Fold {fold + 1}/5")
        data_epoch = []

        # Reset model weights
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

        # Set seeds for reproducibility
        torch.manual_seed(100)
        random.seed(100)
        np.random.seed(100)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Prepare fold data
        fold_train_data = torch.FloatTensor(train_data[train_idx])
        fold_train_labels = torch.FloatTensor(train_labels[train_idx])
        fold_cv_data = torch.FloatTensor(train_data[cv_idx])
        fold_cv_labels = torch.FloatTensor(train_labels[cv_idx])

        fold_train_dataset = TensorDataset(fold_train_data, fold_train_labels)
        fold_cv_dataset = TensorDataset(fold_cv_data, fold_cv_labels)

        fold_train_loader = DataLoader(fold_train_dataset, batch_size=train_loader.batch_size, shuffle=True)
        fold_cv_loader = DataLoader(fold_cv_dataset, batch_size=train_loader.batch_size)

        # Initialize SHAP values
        shap_values = None
        shap_values_val = None

        # Training loop
        best_val_loss = float('inf')
        no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            # Learning rate schedule
            if num_epochs < 10:
                warmup_scheduler.step(num_epochs)
            else:
                scheduler.step()

            # Restart learning rate if it gets too small
            if scheduler.get_last_lr()[0] < 1e-6:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

            # Train one epoch
            for batch_idx, (inputs, labels) in enumerate(fold_train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                # Apply SHAP weights if available
                if shap_values is not None:
                    start_idx = batch_idx * inputs.size(0)
                    end_idx = start_idx + inputs.size(0)
                    if end_idx <= len(shap_values):
                        batch_shap = torch.FloatTensor(shap_values[start_idx:end_idx]).to(device)
                        assert batch_shap.shape == inputs.shape
                    else:
                        batch_shap = None
                else:
                    batch_shap = None

                optimizer.zero_grad()
                outputs = model(inputs, batch_shap)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(fold_train_loader.dataset)

            # Update SHAP values periodically after the start epoch
            if (epoch + 1) > start_shap:
                if (epoch + 1) % shap_update_frequency == 0:
                    print(f"Updating train SHAP values at epoch {epoch + 1}")
                    shap_values = update_shap_values(model, fold_train_loader, device)

                if (epoch + 1) % shap_update_frequency == 0:
                    print(f"Updating val SHAP values at epoch {epoch + 1}")
                    shap_values_val = update_shap_values(model, fold_cv_loader, device)

            # Validate
            labels, outputs, val_loss, r2, rmse, rpd = validate_model(
                model, fold_cv_loader, criterion, device, shap_values_val
            )

            # Print status periodically
            if (epoch + 1) % 100 == 0:
                print(f'Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.2f}, '
                      f'Val Loss: {val_loss:.2f}, R2: {r2:.2f}, RMSE: {rmse:.2f}, RPD: {rpd:.2f}')
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_model_fold_{fold + 1}.pth')
                data_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1

            # Early stopping
            if no_improve >= patience:
                print("Early stopping")
                break

        # Load best model for this fold
        model.load_state_dict(torch.load(f'best_model_fold_{fold + 1}.pth'))
        if os.path.exists(f"best_model_fold_{fold + 1}_epoch{data_epoch}.pth"):
            os.remove(f"best_model_fold_{fold + 1}_epoch{data_epoch}.pth")
        os.rename(f"best_model_fold_{fold + 1}.pth", f"best_model_fold_{fold + 1}_epoch{data_epoch}.pth")
        if shap_values is not None:
        # Update SHAP values for final evaluation
            if data_epoch > shap_update_frequency:
                shap_values = update_shap_values(model, fold_train_loader, device)
                shap_values_val = update_shap_values(model, fold_cv_loader, device)
            else:
                shap_values = None
                shap_values_val = None

        # Evaluate on training set
        labelsc, outputsc, cal_loss, r2c, rmsec, rpdc = validate_model(
            model, fold_train_loader, criterion, device, shap_values
        )

        # Evaluate on validation set
        labels, outputs, val_loss, r2, rmse, rpd = validate_model(
            model, fold_cv_loader, criterion, device, shap_values_val
        )

        print(f'Fold {fold + 1}, Final Results:\n'
              f'Cal Loss: {cal_loss:.2f}, R2C: {r2c:.2f}, RMSEC: {rmsec:.2f}, RPDC: {rpdc:.2f}\n'
              f'Val Loss: {val_loss:.2f}, R2: {r2:.2f}, RMSE: {rmse:.2f}, RPD: {rpd:.2f}')

        fold_val_losses.append(val_loss)
        fold_r2_scores.append(r2)
        fold_rmse_scores.append(rmse)
        fold_rpd_scores.append(rpd)

    # Save results
    data_save = pd.DataFrame({
        'val_loss': fold_val_losses,
        'r2': fold_r2_scores,
        'rmse': fold_rmse_scores,
        'rpd': fold_rpd_scores
    })

    # Save results to Excel
    output_file = f'Epoch {num_epochs}_frequency{shap_update_frequency}_val_loss_{dataset_name}_outputs.xlsx'
    data_save.to_excel(output_file, index=False, header=True)

    # Calculate and print average performance
    print("\nCross-validation Results:")
    print(f"Average Val Loss: {np.mean(fold_val_losses):.2f} ± {np.std(fold_val_losses):.2f}")
    print(f"Average R²: {np.mean(fold_r2_scores):.2f} ± {np.std(fold_r2_scores):.2f}")
    print(f"Average RMSE: {np.mean(fold_rmse_scores):.2f} ± {np.std(fold_rmse_scores):.2f}")
    print(f"Average RPD: {np.mean(fold_rpd_scores):.2f} ± {np.std(fold_rpd_scores):.2f}")

    return model, shap_values
