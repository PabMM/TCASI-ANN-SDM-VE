# Confusion Matrix Calculator and Model Saving Utility

This Python module, named `Lib.py`, provides essential functions for handling confusion matrices and saving machine learning models. The two main functionalities are:

## Confusion Matrix Calculator

### Class: `calculate_confusion_matrix`
This class initializes a confusion matrix calculator, allowing for the plotting and saving of normalized confusion matrices. It has the following methods:

#### Method: `__init__(self, classes, show=False)`
Initializes the Confusion Matrix Calculator.

**Parameters:**
- `classes` (list): List of class labels.
- `show` (bool): Flag to control whether to display the confusion matrix plot.

#### Method: `plot_confusion_matrix(self, true_class, predict_class, model_name, name)`
Plots a normalized confusion matrix and saves it as an EPS file.

**Parameters:**
- `true_class` (array-like): True class labels.
- `predict_class` (array-like): Predicted class labels.
- `model_name` (str): Name of the classification model.
- `name` (str): Name identifier for the confusion matrix.

## Model Saving Utility

### Function: `save_model(model, name, relative_path='')`
This function saves a trained machine learning model to a file using the `pickle` library.

**Parameters:**
- `model`: The trained machine learning model.
- `name` (str): Name identifier for the saved model file.
- `relative_path` (str): Relative path where the model file will be saved.

These utilities can be used in conjunction with machine learning scripts to facilitate the analysis of model performance and enable the persistence of trained models for future use.
