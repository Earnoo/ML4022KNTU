# Classification and Data Preprocessing

## Introduction

This repository provides an overview of Logistic Regression, essential preprocessing techniques, and how to use the Orange application for data mining and classification tasks. Logistic Regression is a statistical method for binary classification problems. Preprocessing is crucial to ensure the quality of data and improve the performance of machine learning models. Orange is a powerful tool for visual programming in data mining.

## Table of Contents

- [Logistic Regression](#logistic-regression)
- [Data Preprocessing](#data-preprocessing)
- [Using Orange for Data Mining](#using-orange-for-data-mining)
- [Datasets](#datasets)
- [Installation](#installation)
- [References](#references)

## Logistic Regression

Logistic Regression is a predictive analysis algorithm used for binary classification problems. It models the probability of a binary outcome based on one or more predictor variables.

### Key Points

- **Binary Classification**: Logistic Regression is used when the dependent variable is binary (e.g., yes/no, 0/1).
- **Sigmoid Function**: The logistic function (sigmoid) is used to model the probability that a given input belongs to a particular class.
- **Log-Odds**: Logistic Regression models the log-odds of the probability of an event occurring.

### Formula

The logistic function is defined as:

$$ P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n)}} $$

where $P(y=1|X)$ is the probability of the event occurring, $\beta_0$ is the intercept, and $\beta_1, \beta_2, ..., \beta_n$ are the coefficients of the predictor variables.

## Data Preprocessing

Preprocessing is a crucial step in the data mining pipeline. It involves preparing and cleaning the data to improve the performance of machine learning models.

### Important Preprocessing Steps

1. **Missing Value Imputation**: Handle missing values by removing them or imputing with mean, median, or mode.
2. **Normalization/Standardization**: Scale the data to ensure all features contribute equally to the model.
3. **Encoding Categorical Variables**: Convert categorical variables into numerical format using techniques like one-hot encoding.
4. **Feature Selection**: Select the most relevant features to reduce dimensionality and improve model performance.
5. **Splitting the Dataset**: Divide the dataset into training and testing sets to evaluate the model's performance.

## Using Orange for Data Mining
![image](https://github.com/user-attachments/assets/73dda0f2-8a5f-4d2a-ac50-16e1e6dfa444)

Orange is an open-source data visualization and analysis tool for novice and expert users. It allows for interactive data analysis through visual programming.

## Datasets
- [Weather in Szeged 2006-2016](https://www.kaggle.com/datasets/budincsevity/szeged-weather/data)
- [CWRU Bearing](https://engineering.case.edu/bearingdatacenter/download-data-file)
### Key Features

- **Visual Programming**: Build workflows with drag-and-drop widgets.
- **Data Visualization**: Explore data with a variety of visualizations.
- **Machine Learning**: Apply various machine learning algorithms and evaluate their performance.
- **Preprocessing Widgets**: Perform data preprocessing tasks such as normalization, imputation, and feature selection.

### Steps to Use Orange for Logistic Regression

1. **Load Data**: Use the 'File' widget to load your dataset.
2. **Preprocess Data**: Use preprocessing widgets like 'Impute', 'Normalize', and 'Select Columns'.
3. **Split Data**: Use the 'Data Sampler' widget to split the data into training and testing sets.
4. **Train Model**: Use the 'Logistic Regression' widget to train the model on the training data.
5. **Evaluate Model**: Use the 'Test & Score' widget to evaluate the model's performance on the test data.
6. **Visualize Results**: Use the 'Confusion Matrix' and other visualization widgets to interpret the results.

## References

- ** R. Magar, L. Ghule, J. Li, Y. Zhao, and A. B. Farimani, “FaultNet: A Deep Convolutional Neural
Network for Bearing Fault Classification,” IEEE Access, vol. 9. Institute of Electrical and Electronics
Engineers (IEEE), pp. 25189–25199, 2021. doi: 10.1109/access.2021.3056944.


## Installation

### Prerequisites
- Python 3.6 or later

## Usage

To use the code in this repository, follow these steps:

1. Clone the repository.
2. Install the required dependencies.
3. Run the provided scripts to reproduce the results.

## License

This project is licensed under the MIT License.


