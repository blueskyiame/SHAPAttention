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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import random
from math import sqrt
import math
import warnings
from Preprocessing import *
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

        # 输入特征标准化
        self.input_norm = nn.LayerNorm(input_dim)

        # 维度对齐投影
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # SHAP权重处理
        self.shap_proj = nn.Sequential(
            nn.Linear(shap_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,
            dropout=0.1,
            batch_first=True
        )

        # 动态参数
        self.shap_scale = nn.Parameter(torch.ones(1))
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 初始化
        nn.init.xavier_normal_(self.feature_proj[0].weight)
        nn.init.kaiming_normal_(self.shap_proj[0].weight)

    def forward(self, x: torch.Tensor, shap_weights: torch.Tensor) -> tuple:
        # 输入标准化
        x = self.input_norm(x)

        # 特征投影
        x_proj = self.feature_proj(x)
        shap_proj = self.shap_proj(shap_weights)

        # 整合SHAP信息
        x_combined = x_proj * (1 + self.shap_scale * torch.sigmoid(shap_proj))

        # 注意力计算（使用标准多头注意力）
        attn_output, attn_weights = self.attention(
            query=x_combined.unsqueeze(1),
            key=x_combined.unsqueeze(1),
            value=x_combined.unsqueeze(1)
        )

        # 残差连接
        output = self.layer_norm(attn_output.squeeze(1) + x_proj)
        return output, attn_weights


class Conv1DCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Conv1DCNN, self).__init__()

        # 第一组卷积块
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(2)

        # 第二组卷积块
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(2)

        # 第三组卷积块
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool1d(2)

        # 全连接层
        self.fc = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(64 * (input_size // 64), 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_size)
        )

        self.fc1 = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_size)
        )

        self.attention = SHAPAttention(hidden_size, 64 * (input_size // 64), input_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.Linear = nn.Linear(64 * (input_size // 64), hidden_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, shap_weights=None):
        x = x.unsqueeze(1)  # 增加通道维度
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        out = x.view(x.size(0), -1)  # 展平

        if shap_weights is not None:
            context_vector, attention_weights = self.attention(out, shap_weights)
            out = context_vector
            x = self.fc1(out)
        else:
            x = self.fc(out)
        return x


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


def train_cv_model_with_shap_updates(model, train_loader, criterion, scheduler, warmup_scheduler, optimizer,
                                     learning_rate, num_epochs, device, shap_update_frequency=5):
    """
    训练模型并定期更新 SHAP 值，包含5折交叉验证
    """

    # 将train_loader中的数据转换为numpy数组用于交叉验证
    train_data = []
    train_labels = []
    for inputs, labels in train_loader:
        train_data.append(inputs.numpy())
        train_labels.append(labels.numpy())
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    y_binned = pd.qcut(train_labels.flatten(), q=5, labels=False)
    # 2. 使用分层抽样进行交叉验证
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # 初始化5折交叉验证
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_val_losses = []
    fold_r2_scores = []
    fold_rmse_scores = []
    fold_rpd_scores = []

    for fold, (train_idx, cv_idx) in enumerate(kf.split(train_data, y_binned)):
        print(f"\nTraining Fold {fold + 1}/5")
        data_epoch = []
        # 重置模型权重
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        torch.manual_seed(100)
        random.seed(100)
        np.random.seed(100)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        fold_train_data = torch.FloatTensor(train_data[train_idx])
        fold_train_labels = torch.FloatTensor(train_labels[train_idx])
        fold_cv_data = torch.FloatTensor(train_data[cv_idx])
        fold_cv_labels = torch.FloatTensor(train_labels[cv_idx])

        fold_train_dataset = TensorDataset(fold_train_data, fold_train_labels)
        fold_cv_dataset = TensorDataset(fold_cv_data, fold_cv_labels)

        fold_train_loader = DataLoader(fold_train_dataset, batch_size=train_loader.batch_size, shuffle=True)
        fold_cv_loader = DataLoader(fold_cv_dataset, batch_size=train_loader.batch_size)

        # 初始化 SHAP 值
        shap_values = None
        shap_values_val = None

        # 训练循环
        best_val_loss = float('inf')
        patience = 1500
        no_improve = 0
        start_shap = 200

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            if num_epochs < 10:
                warmup_scheduler.step(num_epochs)
            else:
                scheduler.step()

            # 重启学习率
            if scheduler.get_last_lr()[0] < 1e-6:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

            for batch_idx, (inputs, labels) in enumerate(fold_train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                if shap_values is not None:
                    start_idx = batch_idx * inputs.size(0)
                    end_idx = start_idx + inputs.size(0)
                    batch_shap = torch.FloatTensor(shap_values[start_idx:end_idx]).to(device)
                    assert batch_shap.shape == inputs.shape
                else:
                    batch_shap = None

                optimizer.zero_grad()
                outputs = model(inputs, batch_shap)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(fold_train_loader.dataset)
            if (epoch + 1) > start_shap:
                # 更新 SHAP 值
                if (epoch + 1) % shap_update_frequency == 0:
                    print(f"Updating train SHAP values at epoch {epoch + 1}")
                    shap_values = update_shap_values(model, fold_train_loader, device)

                if (epoch + 1) % shap_update_frequency == 0:
                    print(f"Updating val SHAP values at epoch {epoch + 1}")
                    shap_values_val = update_shap_values(model, fold_cv_loader, device)

            # 验证
            labels, outputs, val_loss, r2, rmse, rpd = validate_model(model, fold_cv_loader, criterion, device,
                                                                      shap_values_val)
            if (epoch + 1) % 100 == 0:
                print(f'Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.2f}, '
                      f'Val Loss: {val_loss:.2f}, R2: {r2:.2f}, RMSE: {rmse:.2f}, RPD: {rpd:.2f}')

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_model_fold_{fold + 1}.pth')
                data_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print("Early stopping")
                break

        model.load_state_dict(torch.load(f'best_model_fold_{fold + 1}.pth'))
        if os.path.exists(f"best_model_fold_{fold + 1}_epoch{data_epoch}.pth"):
            os.remove(f"best_model_fold_{fold + 1}_epoch{data_epoch}.pth")
        os.rename(f"best_model_fold_{fold + 1}.pth", f"best_model_fold_{fold + 1}_epoch{data_epoch}.pth")
        if shap_values is not None:
            if data_epoch > shap_update_frequency:
                shap_values = update_shap_values(model, fold_train_loader, device)
                shap_values_val = update_shap_values(model, fold_cv_loader, device)
            else:
                shap_values = None
                shap_values_val = None
        labelsc, outputsc, cal_loss, r2c, rmsec, rpdc = validate_model(model, fold_train_loader, criterion, device,
                                                                       shap_values)
        labels, outputs, val_loss, r2, rmse, rpd = validate_model(model, fold_cv_loader, criterion, device,
                                                                  shap_values_val)

        print(f'Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, '
              f'Cal Loss: {cal_loss:.2f}, R2C: {r2c:.2f}, RMSEC: {rmsec:.2f}, RPDC: {rpdc:.2f},'
              f'Val Loss: {val_loss:.2f}, R2: {r2:.2f}, RMSE: {rmse:.2f}, RPD: {rpd:.2f}')

        fold_val_losses.append(val_loss)
        fold_r2_scores.append(r2)
        fold_rmse_scores.append(rmse)
        fold_rpd_scores.append(rpd)

    data_save = pd.DataFrame({
        'val_loss': fold_val_losses,
        'r2': fold_r2_scores,
        'rmse': fold_rmse_scores,
        'rpd': fold_rpd_scores
    })
    data_save.to_excel(f'Epoch {num_epochs}_frequency{shap_update_frequency}_val_loss_Cal_ManufacturerB_outputs.xlsx',
                       index=False, header=True)
    # data_save.to_excel(f'Epoch {num_epochs}_frequency{shap_update_frequency}_val_loss_Ramandata_tablets_outputs.xlsx',
    #                    index=False, header=True)
    # 计算并打印所有折的平均性能
    print("\nCross-validation Results:")
    print(f"Average Val Loss: {np.mean(fold_val_losses):.2f} ± {np.std(fold_val_losses):.2f}")
    print(f"Average R²: {np.mean(fold_r2_scores):.2f} ± {np.std(fold_r2_scores):.2f}")
    print(f"Average RMSE: {np.mean(fold_rmse_scores):.2f} ± {np.std(fold_rmse_scores):.2f}")
    print(f"Average RPD: {np.mean(fold_rpd_scores):.2f} ± {np.std(fold_rpd_scores):.2f}")

    return model, shap_values


def validate_model(model, val_loader, criterion, device, shap_values):
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
    rpd = np.std(all_labels) / rmse  # 计算RPD

    return all_labels, all_outputs, val_loss / len(val_loader.dataset), r2, rmse, rpd


def update_shap_values(model, train_loader, device, num_background=400):
    model.eval()
    background = []
    model_cpu = copy.deepcopy(model).cpu()
    for inputs, _ in train_loader:
        background.append(inputs)
        if len(background) * inputs.size(0) >= num_background:
            break
    background = torch.cat(background)[:num_background]

    # 创建一个 DeepExplainer
    explainer = shap.DeepExplainer(model_cpu, background)

    # 计算所有训练数据的 SHAP 值
    all_shap_values = []
    for inputs, _ in train_loader:
        inputs = inputs
        # batch_shap_values = explainer.shap_values(inputs).squeeze(2)  ###python3.9环境正常
        batch_shap_values = explainer.shap_values(inputs)  ###wupytorch3.8环境正常
        all_shap_values.append(batch_shap_values)  # 假设模型只有一个输出

    all_shap_values = np.concatenate(all_shap_values, axis=0)
    model.to(device)

    return all_shap_values


def calculate_shap_values(model, background_data, test_data):
    model.eval()
    explainer = shap.DeepExplainer(model.cpu(), background_data)
    shap_values = explainer.shap_values(test_data)  ###wupytorch3.8环境正常
    # shap_values = explainer.shap_values(test_data).squeeze(2)###python3.9环境正常
    return shap_values


# 主函数
def main():

    for Data in range(1, 4):
        torch.manual_seed(100)
        random.seed(100)
        np.random.seed(100)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if Data == 1:
            # ############数据集1###############################################
            data = pd.read_excel("Cal_ManufacturerB.xlsx")
            # Get reference values
            y = data['Protein'].values
            X = data.drop(['ID', 'Protein'], axis=1).values
            scaler = StandardScaler()  # 标准化
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2022)
            ###############################################################
        if Data == 2:
            # ############数据集2###############################################
            data = pd.read_excel("Ramandata_tablets.xlsx")
            y = data['PE'].values
            # Get spectra
            X = data.drop(['PE'], axis=1).values
            scaler = StandardScaler()  # 标准化
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2022)
            ###############################################################
        if Data == 3:
            ############数据集3###############################################
            data = pd.read_excel("高光谱1.xlsx", index_col=0)
            y = data.iloc[:, 0].values
            X = data.iloc[:, 1:].values
            y = np.log2(y + 1)
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
        hidden_size = 256  # Cal_ManufacturerB 1024  Ramandata_tablets 3392
        output_size = 1
        learning_rate = 1e-4  # Cal_ManufacturerB 1e-2  Ramandata_tablets 1e-4  高光谱1 2e-3
        num_epochs = 4000

        shap_update_frequency = 400

        model = Conv1DCNN(input_size, hidden_size, output_size).to(device)

        # 首先训练模型而不使用SHAP权重
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=0.001)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        warmup_scheduler = WarmupScheduler(optimizer, 100, 1e-5, 1e-3)

        trained_model, final_shap_values = train_cv_model_with_shap_updates(
            model, all_loader, criterion,
            scheduler, warmup_scheduler, optimizer, learning_rate, num_epochs, device, shap_update_frequency
        )


if __name__ == "__main__":
    main()
