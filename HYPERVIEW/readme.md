### Project Overview
This project focuses on the analysis of the HYPERVIEW dataset using a specific algorithm implemented in the `SHAPAttention_1DCNN_HYPERVIEW_CosineAnnealingLR6` model. The following details provide an overview of the training, testing, and result - saving processes.
HYPERVIEW：https://platform.ai4eo.eu/seeing-beyond-the-visible-permanent
	HYPERVIEW was a part of the IEEE International Conference on Image Processing (ICIP) 2022 conference.
### Code Files
1. **Training Code**: `SHAPAttention_1DCNN_HYPERVIEW_CosineAnnealingLR6`
    - This code is used to train the model on the HYPERVIEW dataset. It applies a specific algorithm that is tailored to the characteristics of the dataset. During training, a five - fold cross - validation strategy is employed to evaluate the model's performance.
2. **Testing Code**: `SHAPAttention_1DCNN_HYPERVIEW_CosineAnnealingLR6_prdict`
    - This code is responsible for testing the trained model on the test set of the HYPERVIEW dataset.

### Training Results
#### Five - Fold Cross - Validation
The five - fold cross - validation on the training set was carried out, and the optimal epochs for each fold are as follows:
- Fold 1: `best_model_fold_1_epoch220`
- Fold 2: `best_model_fold_2_epoch1`
- Fold 3: `best_model_fold_3_epoch9`
- Fold 4: `best_model_fold_4_epoch1`
- Fold 5: `best_model_fold_5_epoch2`

#### Saved Results for Fold 5
The results obtained from the best - performing model in Fold 5 (`best_model_fold_5_epoch2`) are saved in the following files:
- `metrics_summary_best_model_fold_5_epoch2.xlsx`: This file contains a summary of the evaluation metrics for the model on the training set during the five - fold cross - validation.
- `prediction_results_best_model_fold_5_epoch2.xlsx`: It stores the prediction results of the model on the training set for Fold 5.
- `prediction_results_best_model_fold_5_epoch2.png`: This is a visual representation of the prediction results on the training set for Fold 5.

### Testing Results
The prediction results of the model on the test set are saved in the `predict_testsets.xlsx` file. This file includes the model's predictions for the test set data.

### Usage Instructions
1. **Training**: Run the `SHAPAttention_1DCNN_HYPERVIEW_CosineAnnealingLR6` code to train the model and obtain the five - fold cross - validation results.
2. **Testing**: After training, use the `SHAPAttention_1DCNN_HYPERVIEW_CosineAnnealingLR6_prdict` code to test the model on the test set and generate the test set prediction results.

Please note that the data used in this project is from the HYPERVIEW dataset, and the code uses specific pre - processing steps such as standardizing the spectral bands and applying MinMaxScaler scaling to the predicted Y values. 