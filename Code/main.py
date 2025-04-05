
# main.py
"""
Main script for SHAPAttention.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import time
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
# Import custom modules
from config import get_config
from model import Conv1DCNN
from preprocessing import load_and_preprocess_data, create_data_loaders
from train import train_cv_model_with_shap_updates, WarmupScheduler
from utils import set_seed

warnings.filterwarnings("ignore")


def main():
    """Main function to run the analysis."""
    start_time = time.time()

    # Get configuration
    args = get_config()

    # Set seed for reproducibility
    set_seed(args.random_seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Load and preprocess data
    X_train, X_test, y_train, y_test, X_scaled, y, input_size = load_and_preprocess_data(
        args.dataset, args.test_size, args.random_seed
    )

    # Create data loaders
    train_loader, test_loader, all_loader = create_data_loaders(
        X_train, X_test, y_train, y_test, X_scaled, y,
        args.batch_size, args.test_batch_size
    )

    # Create model
    model = Conv1DCNN(
        input_size=input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        dropout_rate=args.dropout_rate
    ).to(device)

    # Define loss function
    criterion = nn.MSELoss()

    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Define schedulers
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5, verbose=True)
    # warmup_scheduler = WarmupScheduler(optimizer, 100, 1e-5, args.learning_rate)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    warmup_scheduler = WarmupScheduler(optimizer, 100, 1e-5, 1e-3)



    print(f"\nStarting training with the following parameters:")
    print(f"Dataset: {args.dataset}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Dropout rate: {args.dropout_rate}")
    print(f"SHAP update frequency: {args.shap_update_frequency}")
    print(f"Early stopping patience: {args.patience}")

    # Train model
    trained_model, final_shap_values = train_cv_model_with_shap_updates(
        model=model,
        train_loader=all_loader,
        criterion=criterion,
        scheduler=scheduler,
        warmup_scheduler=warmup_scheduler,
        optimizer=optimizer,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=device,
        shap_update_frequency=args.shap_update_frequency,
        patience=args.patience,
        start_shap=args.start_shap,
        dataset_name=args.dataset
    )

    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")

    # Save final model
    torch.save(trained_model.state_dict(), f"results/final_model_{args.dataset}.pth")
    print(f"Model saved to results/final_model_{args.dataset}.pth")


if __name__ == "__main__":
    main()
