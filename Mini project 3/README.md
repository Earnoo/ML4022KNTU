# Iris Dataset Visualization and Classification & Credit Card Fraud Detection

This repository contains two main parts:

1. Visualization and classification of the Iris dataset using PCA, t-SNE, and SVM.
2. Credit card fraud detection using SMOTE, Autoencoders, and Neural Networks.

## Table of Contents

- Iris Dataset Visualization and Classification
  - Visualization
  - Classification
- Credit Card Fraud Detection
  - Data Preprocessing
  - Autoencoder Training
  - Classifier Training
  - Evaluation

## Iris Dataset Visualization and Classification

### Visualization

1. Data Loading: Load the Iris dataset and create a DataFrame with appropriate feature names and target labels.
2. 2D and 3D Plotting: Define functions to plot the data in 2D and 3D using matplotlib and seaborn.
3. PCA and t-SNE: Visualize the dataset using PCA and t-SNE for dimensionality reduction.

### Classification

1. Train/Test Split: Split the dataset into training and testing sets.
2. Standardization: Standardize the features using StandardScaler.
3. SVM Classification: Train SVM classifiers with different kernels (linear and polynomial) and evaluate their performance using confusion matrices and classification reports.
4. Decision Boundaries: Plot the decision boundaries for the SVM classifiers after dimensionality reduction using PCA.
<img src="https://github.com/user-attachments/assets/20f67e35-cc57-4cbe-9220-4797bf504638" alt="Q1" width="400" height="300">

## Credit Card Fraud Detection

### Data Preprocessing

1. Data Loading: Load the credit card fraud dataset and remove the 'Time' column.
2. Standardization: Standardize the 'Amount' feature.
3. Class Distribution: Visualize the class distribution of the dataset.
4. Train/Test/Validation Split: Split the dataset into training, validation, and test sets.
5. SMOTE: Apply SMOTE to handle class imbalance in the training set.

### Autoencoder Training

1. Add Noise: Add Gaussian noise to the training and test data.
2. Autoencoder Model: Define and train an autoencoder model to denoise the data.
3. Denoising: Denoise the training, validation, and test sets using the trained autoencoder.

### Classifier Training

1. Neural Network Model: Define and train a neural network classifier using the denoised data.
2. Model Checkpointing: Save the best model during training based on validation loss.

### Evaluation

1. Model Evaluation: Evaluate the classifier on the validation set using various metrics (accuracy, recall, precision, F1-score).
2. Confusion Matrix: Plot and visualize the confusion matrix.
3. Threshold Adjustment: Adjust classification thresholds to optimize recall and accuracy.
4. SMOTE Sampling Strategies: Evaluate the impact of different SMOTE sampling strategies on model performance.

## Usage

To use the code in this repository, follow these steps:

1. Clone the repository.
2. Install the required dependencies.
3. Run the provided scripts to reproduce the results.

## License

This project is licensed under the MIT License.
