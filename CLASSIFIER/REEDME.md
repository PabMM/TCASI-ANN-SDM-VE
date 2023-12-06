# Species Classifier using Different Classifiers

The Python script `classifiers.py` trains and evaluates several classifiers for a species classification task. It covers the following steps:

## Data Reading and Normalization

The script reads a CSV file containing species classification data. It then preprocesses the data by removing unnecessary columns and optionally performs Min-Max scaling on the remaining features.

## Splitting the Data

The data is split into training and testing sets using the `train_test_split` function from `sklearn`. The script prepares input features (`x`) and target labels (`y`) for both sets.

## Confusion Matrix Initialization

The confusion matrix is initialized using the `calculate_confusion_matrix` class from the custom library `Lib`. This class helps in visualizing and analyzing the performance of the classifiers.

## Classifier Training and Evaluation

The script proceeds to train and evaluate several classifiers:

### Gaussian Naive Bayes (GNB) Classifier
- The Gaussian Naive Bayes model is trained using `sklearn.naive_bayes.GaussianNB`.
- The fitting time and accuracy are printed for the training, testing, and entire datasets.

### Logistic Regression Classifier
- The Logistic Regression model is trained using `sklearn.linear_model.LogisticRegression`.
- Similar to GNB, the fitting time and accuracy are printed for different datasets.

### Multinomial Naive Bayes (MNB) Classifier
- The Multinomial Naive Bayes model is trained using `sklearn.naive_bayes.MultinomialNB`.
- Fitting time and accuracy are displayed for various datasets.

### Quadratic Discriminant Analysis (QDA) Classifier
- The Quadratic Discriminant Analysis model is trained using `sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`.
- Fitting time and accuracy are shown for different datasets.

### Random Forest (RF) Classifier
- The Random Forest model is trained using `sklearn.ensemble.RandomForestClassifier`.
- Fitting time and accuracy metrics are printed for various datasets.

### Linear Discriminant Analysis (LDA) Classifier
- The Linear Discriminant Analysis model is trained using `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`.
- Fitting time and accuracy are displayed for different datasets.

### Decision Tree (DT) Classifier
- The Decision Tree model is trained using `sklearn.tree.DecisionTreeClassifier`.
- Fitting time and accuracy are presented for training, testing, and entire datasets.

### Support Vector Machine (SVM) Classifier
- Support Vector Machine models with different kernels (linear, polynomial, radial basis function) are trained using `sklearn.svm.SVC`.
- Fitting time, accuracy, and confusion matrices are displayed for each SVM variant.

### Gradient Boosting (GB) Classifier
- The script optionally allows adjusting the learning rate for the Gradient Boosting Classifier.
- Fitting time, accuracy, and confusion matrices are shown for the selected GB model.

### TensorFlow Neural Network (ANN) Classifier
- A simple neural network model is created using TensorFlow's Keras API.
- The model is compiled, trained, and evaluated on the dataset.

## Model Saving
Trained models are saved using the custom `save_model` function from the `Lib` library.

This script provides a comprehensive analysis of various classifiers for the species classification task, including fitting times, accuracy metrics, and confusion matrices.

## Outputs

### Displays
Informational displays, including fitting times, accuracies, and confusion matrices, are saved in the `displays.txt` file. This file serves as a comprehensive record of the classifiers' evaluations on different datasets.

### Model Saving
Trained models are saved in the `model` folder. Each classifier has its corresponding model file saved within this directory. These saved models can be later used for inference or further analysis.

### Confusion Matrices
The confusion matrices for each classifier on the training, testing, and overall datasets are saved in the `Confusion_Matrices` folder. These visualizations provide a detailed breakdown of the model's predictions, aiding in the analysis of classification performance.

These output files offer valuable information for understanding and comparing the performance of the different classifiers employed in the species classification task.
