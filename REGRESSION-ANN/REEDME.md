# Regression Model using Neural Network

## Introduction

This Python script, named `regression_ann_NSA.py`, implements a neural network-based regression model for predicting output parameters based on input specifications. It utilizes TensorFlow and Keras for building and training the model. The script includes hyperparameter tuning with Keras Tuner and provides options for model evaluation and prediction.

## Usage

1. **Dataset Loading:**
   - The script loads the dataset from a CSV file, where the file name is constructed based on the specified model name.

2. **Data Preprocessing:**
   - The input and output variables are separated from the dataset.
   - The output variables are scaled using Min-Max scaling.
   - The dataset is split into training, validation, and test sets.

3. **Model Architecture Tuning:**
   - The hyperparameter tuning is performed using Keras Tuner to find the optimal architecture for the neural network model.

4. **Model Training:**
   - The best-tuned model is then trained on the training dataset with early stopping to avoid overfitting.

5. **Model Evaluation:**
   - The script evaluates the trained model on the test set and prints the Mean Squared Error (MSE) for both the test and training sets.

6. **Model Saving:**
   - The trained model is saved to the 'REGRESSION-ANN/models/' directory.

7. **Outputs:**
   - The script generates various outputs:
     - TensorBoard logs in 'REGRESSION-ANN/tb_logs/' for visualizing training metrics.
     - Model summary printed to the console.
     - Model architecture plot saved as an image in 'REGRESSION-ANN/models/'.
     - MSE for the test and training sets.

## Outputs
After running the script, several files and outputs are generated:
- Trained models are saved in 'REGRESSION-ANN/models/'.
- TensorBoard logs are stored in 'REGRESSION-ANN/tb_logs/'.
- Model architecture plots are saved in 'REGRESSION-ANN/models/'.
- The script prints MSE for the test and training sets.


