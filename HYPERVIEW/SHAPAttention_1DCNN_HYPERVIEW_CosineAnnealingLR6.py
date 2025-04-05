import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import copy
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import random
from math import sqrt
import math
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 数据加载和预处理
class NIRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # 确保y是2D

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SHAPAttention(nn.Module):
    def __init__(self, hidden_dim=256, input_dim=1024, shap_dim=1024):
        super().__init__()

        # Input feature normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # Dimension alignment projection
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # SHAP weight processing
        self.shap_proj = nn.Sequential(
            nn.Linear(shap_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,
            dropout=0.1,
            batch_first=True
        )

        # Dynamic parameters
        self.shap_scale = nn.Parameter(torch.ones(1))
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, shap_weights: torch.Tensor) -> tuple:
        batch_size = x.size(0)

        # Input normalization
        x = self.input_norm(x)

        # Feature projection
        x_proj = self.feature_proj(x)

        # SHAP projection - Handle multi-output SHAP values
        if len(shap_weights.shape) > 2:  # Multiple outputs
            # For multi-output models, we average SHAP values across outputs
            # This assumes shap_weights shape is [output_dim, batch_size, feature_dim]
            shap_weights = shap_weights.mean(dim=0)  # Average across outputs

        shap_proj = self.shap_proj(shap_weights)

        # Integrate SHAP information
        x_combined = x_proj * (1 + self.shap_scale * torch.sigmoid(shap_proj))

        # Attention calculation
        attn_output, attn_weights = self.attention(
            query=x_combined.unsqueeze(1),
            key=x_combined.unsqueeze(1),
            value=x_combined.unsqueeze(1)
        )

        # Residual connection
        output = self.layer_norm(attn_output.squeeze(1) + x_proj)
        return output, attn_weights

    def reset_parameters(self):
        """Reset parameters of all submodules"""
        self.apply(self._init_weights)


class Conv1DCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Conv1DCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(2)

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(2)

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool1d(2)

        # Calculate the flattened size
        self.flattened_size = 64 * (input_size // 64)  # This might need adjustment

        # Regular path
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_size)
        )

        # Path with attention
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_size)
        )

        # Attention module
        self.attention = SHAPAttention(hidden_size, self.flattened_size, input_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(self.flattened_size, hidden_size)
        self.dropout = nn.Dropout(p=0.2)

    def reset_parameters(self):
        """Reset all parameters in the model"""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, x, shap_weights=None):
        # Add channel dimension
        x = x.unsqueeze(1)

        # Convolutional layers
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        # Flatten
        out = x.view(x.size(0), -1)

        # Handle the case where out shape doesn't match expected flattened_size
        if out.size(1) != self.flattened_size:
            # Adjust the linear layer dynamically
            actual_size = out.size(1)
            self.fc[0] = nn.Linear(actual_size, 256).to(out.device)
            if hasattr(self, 'linear'):
                self.linear = nn.Linear(actual_size, self.attention.feature_proj[0].out_features).to(out.device)

        # Choose path based on whether SHAP weights are provided
        if shap_weights is not None:
            # SHAP attention path
            context_vector, attention_weights = self.attention(out, shap_weights)
            output = self.fc1(context_vector)
        else:
            # Regular path
            output = self.fc(out)

        return output

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
            # 如果使用其他调度器，在此处调用
            # 例如：self.scheduler.step()
            pass
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def train_cv_model_with_shap_updates(model, train_loader, scaler_Y,criterion, scheduler, warmup_scheduler, optimizer,
                                     learning_rate, num_epochs, device, shap_update_frequency=5):
    """
    Train the model with periodic SHAP value updates, using 5-fold cross-validation
    """
    # Convert train_loader data to numpy arrays for cross-validation
    train_data = []
    train_labels = []
    for inputs, labels in train_loader:
        train_data.append(inputs.numpy())
        train_labels.append(labels.numpy())
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels).squeeze(1)

    # Create stratified folds based on the first output column
    y_binned = pd.qcut(train_labels[:, 0], q=5, labels=False)

    # Initialize stratified K-fold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Track metrics for each fold
    fold_val_losses = [[] for _ in range(train_labels.shape[1])]
    fold_r2_scores = [[] for _ in range(train_labels.shape[1])]
    fold_rmse_scores = [[] for _ in range(train_labels.shape[1])]
    fold_rpd_scores = [[] for _ in range(train_labels.shape[1])]

    for fold, (train_idx, cv_idx) in enumerate(kf.split(train_data, y_binned)):
        print(f"\nTraining Fold {fold + 1}/5")

        # Reset model weights and set seeds for reproducibility
        # model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        torch.manual_seed(100)
        random.seed(100)
        np.random.seed(100)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Prepare data for this fold
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
        patience = 1500
        no_improve = 0
        start_shap = 200
        best_epoch = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            # Adjust learning rate
            if num_epochs < 10:
                warmup_scheduler.step(epoch)
            else:
                # Try-except to handle different scheduler implementation
                try:
                    scheduler.step()
                except:
                    scheduler.step(best_val_loss)

            # Restart learning rate if too small
            if scheduler.get_last_lr()[0] < 1e-6:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

            # Train for one epoch
            for batch_idx, (inputs, labels) in enumerate(fold_train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                # Apply SHAP values if available
                if shap_values is not None:
                    try:
                        start_idx = batch_idx * inputs.size(0)
                        end_idx = min(start_idx + inputs.size(0), shap_values.shape[1])
                        if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                            # For multiple outputs: [output_dim, batch_size, features]
                            batch_shap = torch.FloatTensor(shap_values[:, start_idx:end_idx, :]).to(device)
                        else:
                            # Fallback for single output
                            batch_shap = torch.FloatTensor(shap_values[start_idx:end_idx]).to(device)
                    except Exception as e:
                        print(f"Error in SHAP batch processing: {e}")
                        print(f"SHAP values shape: {shap_values.shape}, Inputs shape: {inputs.shape}")
                        batch_shap = None
                else:
                    batch_shap = None

                # Forward pass and optimization
                optimizer.zero_grad()
                outputs = model(inputs, batch_shap)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(fold_train_loader.dataset)

            # Update SHAP values periodically
            if (epoch + 1) > start_shap:
                if (epoch + 1) % shap_update_frequency == 0:
                    print(f"Updating train SHAP values at epoch {epoch + 1}")
                    try:
                        shap_values = update_shap_values(model, fold_train_loader, device)
                    except Exception as e:
                        print(f"Error updating train SHAP values: {e}")
                        shap_values = None

                if (epoch + 1) % shap_update_frequency == 0:
                    print(f"Updating val SHAP values at epoch {epoch + 1}")
                    try:
                        shap_values_val = update_shap_values(model, fold_cv_loader, device)
                    except Exception as e:
                        print(f"Error updating val SHAP values: {e}")
                        shap_values_val = None

            # Validation
            labels, outputs, val_loss, r2, rmse, rpd = validate_model(model, fold_cv_loader, scaler_Y,criterion, device,
                                                                      shap_values_val)

            # Print progress
            if (epoch + 1) % 100 == 0:
                if isinstance(r2, list) and isinstance(rmse, list) and isinstance(rpd, list):
                    for i, (r, rm, rp) in enumerate(zip(r2, rmse, rpd)):
                        print(f'Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Output {i + 1}, '
                              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                              f'R2: {r:.4f}, RMSE: {rm:.4f}, RPD: {rp:.4f}')

            # Learning rate scheduler step
            try:
                scheduler.step(val_loss)
            except:
                pass  # Some schedulers don't take val_loss as parameter

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), f'best_model_fold_{fold + 1}.pth')
                no_improve = 0
            else:
                no_improve += 1

            # Early stopping
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model for this fold
        model.load_state_dict(torch.load(f'best_model_fold_{fold + 1}.pth'))

        # Rename the file with epoch number
        if os.path.exists(f"best_model_fold_{fold + 1}_epoch{best_epoch}.pth"):
            os.remove(f"best_model_fold_{fold + 1}_epoch{best_epoch}.pth")
        os.rename(f"best_model_fold_{fold + 1}.pth", f"best_model_fold_{fold + 1}_epoch{best_epoch}.pth")

        # Final SHAP update for best model if applicable
        if best_epoch > shap_update_frequency:
            try:
                shap_values = update_shap_values(model, fold_train_loader, device)
                shap_values_val = update_shap_values(model, fold_cv_loader, device)
            except Exception as e:
                print(f"Error in final SHAP update: {e}")
                shap_values = None
                shap_values_val = None
        else:
            shap_values = None
            shap_values_val = None

        # Final evaluation
        labelsc, outputsc, cal_loss, r2c, rmsec, rpdc = validate_model(model, fold_train_loader,scaler_Y, criterion, device,
                                                                       shap_values)
        labels, outputs, val_loss, r2, rmse, rpd = validate_model(model, fold_cv_loader, scaler_Y,criterion, device,
                                                                  shap_values_val)

        # Print final results for this fold
        if (isinstance(r2c, list) and isinstance(rmsec, list) and isinstance(rpdc, list) and
                isinstance(r2, list) and isinstance(rmse, list) and isinstance(rpd, list)):
            for i, (rc, rmc, rpdc_val, r, rm, rpd_val) in enumerate(zip(r2c, rmsec, rpdc, r2, rmse, rpd)):
                print(f'Fold {fold + 1}, Final Results, Output {i + 1}: '
                      f'Cal Loss: {cal_loss:.4f}, R2C: {rc:.4f}, RMSEC: {rmc:.4f}, RPDC: {rpdc_val:.4f}, '
                      f'Val Loss: {val_loss:.4f}, R2: {r:.4f}, RMSE: {rm:.4f}, RPD: {rpd_val:.4f}')

                # Store metrics for this output
                fold_val_losses[i].append(val_loss)
                fold_r2_scores[i].append(r)
                fold_rmse_scores[i].append(rm)
                fold_rpd_scores[i].append(rpd_val)

    # Save results to Excel
    results_data = {}
    for i in range(train_labels.shape[1]):
        results_data.update({
            f'val_loss_{i + 1}': fold_val_losses[i],
            f'r2_{i + 1}': fold_r2_scores[i],
            f'rmse_{i + 1}': fold_rmse_scores[i],
            f'rpd_{i + 1}': fold_rpd_scores[i]
        })

    data_save = pd.DataFrame(results_data)
    data_save.to_excel(f'Epoch_{num_epochs}_frequency{shap_update_frequency}_val_loss_outputs.xlsx',
                       index=False, header=True)

    # Print average metrics for each output
    for i in range(train_labels.shape[1]):
        print(f"\nCross-validation Results for Output {i + 1}:")
        print(f"Average Val Loss: {np.mean(fold_val_losses[i]):.4f} ± {np.std(fold_val_losses[i]):.4f}")
        print(f"Average R²: {np.mean(fold_r2_scores[i]):.4f} ± {np.std(fold_r2_scores[i]):.4f}")
        print(f"Average RMSE: {np.mean(fold_rmse_scores[i]):.4f} ± {np.std(fold_rmse_scores[i]):.4f}")
        print(f"Average RPD: {np.mean(fold_rpd_scores[i]):.4f} ± {np.std(fold_rpd_scores[i]):.4f}")

    return model, shap_values


def validate_model(model, val_loader, scaler_Y,criterion, device, shap_values):
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
                # Check if SHAP values have the correct shape for multi-output
                if len(shap_values) == 4:  # For 4 outputs
                    # Handle multiple SHAP values (one per output)
                    batch_shap = torch.FloatTensor(shap_values[0][start_idx:end_idx]).to(device)
                else:
                    # Single SHAP value array
                    batch_shap = torch.FloatTensor(shap_values[start_idx:end_idx]).to(device)

                # Ensure SHAP values match input shape
                if batch_shap.shape != inputs.shape:
                    # Reshape if necessary
                    batch_shap = batch_shap.reshape(inputs.shape)
            else:
                batch_shap = None

            outputs = model(inputs, batch_shap)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    all_outputs = scaler_Y.inverse_transform(all_outputs)
    all_labels = scaler_Y.inverse_transform(all_labels)
    # Calculate metrics for each output dimension
    r2 = []
    rmse = []
    rpd = []

    for i in range(all_labels.shape[1]):  # For each of the 4 outputs
        r2.append(r2_score(all_labels[:, i], all_outputs[:, i]))
        current_rmse = np.sqrt(mean_squared_error(all_labels[:, i], all_outputs[:, i]))
        rmse.append(current_rmse)
        rpd.append(np.std(all_labels[:, i]) / current_rmse)

    return all_labels, all_outputs, val_loss / len(val_loader.dataset), r2, rmse, rpd


def update_shap_values(model, train_loader, device, num_background=400):
    model.eval()
    background = []
    model_cpu = copy.deepcopy(model).cpu()

    # Collect background data
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
        # Calculate SHAP values - will return a list with one element per output
        batch_shap_values = explainer.shap_values(inputs)

        # For multiple outputs (output_size=4), batch_shap_values will be a list of 4 arrays
        if isinstance(batch_shap_values, list) and len(batch_shap_values) > 1:
            # Stack the values for each output into a single array [output_dim, batch_size, features]
            batch_shap_values = np.array(batch_shap_values)

        all_shap_values.append(batch_shap_values)

    # Properly combine all batches
    if isinstance(all_shap_values[0], np.ndarray) and len(all_shap_values[0].shape) == 3:
        # For multiple outputs: concatenate along batch dimension (axis=1)
        all_shap_values = np.concatenate(all_shap_values, axis=1)
    else:
        # For single output or different format: handle accordingly
        temp = []
        for output_idx in range(len(all_shap_values[0])):
            output_values = np.concatenate([batch[output_idx] for batch in all_shap_values], axis=0)
            temp.append(output_values)
        all_shap_values = np.array(temp)

    model.to(device)
    return all_shap_values

# 主函数
def main():
    torch.manual_seed(100)
    random.seed(100)
    np.random.seed(100)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ############数据集3###############################################
    data = pd.read_excel("HYPERVIEW_train.xlsx", index_col=0)
    y = data.iloc[:, 0:4].values #
    X = data.iloc[:, 4:].values
    scaler_Y = MinMaxScaler(feature_range=(0, 10))
    y = scaler_Y.fit_transform(y)
    scaler = StandardScaler()  # 标准化
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2022)
    ###############################################################

    train_dataset = NIRDataset(X_train, y_train)
    test_dataset = NIRDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    all_dataset = NIRDataset(X_scaled, y)
    all_loader = DataLoader(all_dataset, batch_size=64, shuffle=True)

    input_size = X_train.shape[1]
    hidden_size = 256
    output_size = 4
    learning_rate = 1e-3
    num_epochs = 4000

    shap_update_frequency = 20000

    model = Conv1DCNN(input_size, hidden_size, output_size).to(device)

    # 首先训练模型而不使用SHAP权重
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    warmup_scheduler = WarmupScheduler(optimizer, 100, 1e-5, 1e-3)

    trained_model, final_shap_values = train_cv_model_with_shap_updates(
        model, all_loader,scaler_Y, criterion,
        scheduler, warmup_scheduler, optimizer, learning_rate, num_epochs, device, shap_update_frequency
    )


if __name__ == "__main__":
    main()
