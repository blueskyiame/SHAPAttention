# config.py
"""
Configuration parameters for the SHAPAttention model.
"""
import argparse


def get_config():
    """
    Get configuration from command line arguments.
    Returns: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='SHAPAttention')

    # Dataset selection
    parser.add_argument('--dataset', type=str, default="Cal_ManufacturerB",
                        choices=["Cal_ManufacturerB", "Ramandata_tablets", "高光谱1"],
                        help='Dataset to use')

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=None,
                        help='Hidden size for the model (if None, uses dataset default)')
    parser.add_argument('--output_size', type=int, default=1,
                        help='Output size for the model')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=4000,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=128,
                        help='Batch size for testing')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (if None, uses dataset default)')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=1500,
                        help='Patience for early stopping')

    # SHAP parameters
    parser.add_argument('--shap_update_frequency', type=int, default=400,
                        help='Frequency of SHAP updates')
    parser.add_argument('--start_shap', type=int, default=200,
                        help='Epoch to start SHAP updates')

    # Other parameters
    parser.add_argument('--random_seed', type=int, default=100,
                        help='Random seed for reproducibility')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test size ratio')

    args = parser.parse_args()

    # Set default values based on dataset
    MODEL_PARAMS = {
        "Cal_ManufacturerB": {
            "hidden_size": 256,
            "learning_rate": 1e-4,
        },
        "Ramandata_tablets": {
            "hidden_size": 256,
            "learning_rate": 1e-4,
        },
        "高光谱1": {
            "hidden_size": 256,
            "learning_rate": 1e-4,
        }
    }

    # Set defaults if not provided
    if args.hidden_size is None:
        args.hidden_size = MODEL_PARAMS[args.dataset]["hidden_size"]
    if args.learning_rate is None:
        args.learning_rate = MODEL_PARAMS[args.dataset]["learning_rate"]

    return args


