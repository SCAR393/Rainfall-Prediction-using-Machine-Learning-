# Rainfall Prediction Project

## Project Description
This project develops a machine learning model to predict rainfall. It includes data preprocessing, exploratory data analysis, and training a RandomForestClassifier with hyperparameter tuning, followed by model evaluation and saving the best model for future predictions.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Collection and Processing](#data-collection-and-processing)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Processing](#data-processing)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Prediction on Unknown Data](#prediction-on-unknown-data)
8. [Saving and Loading the Model](#saving-and-loading-the-model)

## Introduction
This project aims to build a robust machine learning model capable of predicting rainfall based on various weather parameters.

## Data Collection and Processing
The dataset (`Rainfall.csv`) was loaded and initial checks were performed for its shape and structure. Irrelevant columns like 'day' were dropped, and missing values in 'winddirection' and 'windspeed' were handled using mode and median imputation respectively. The target variable 'rainfall' was converted from categorical ('yes', 'no') to numerical (1, 0).

## Exploratory Data Analysis (EDA)
EDA was conducted to understand the distribution of features, identify outliers, and visualize relationships between variables. Histograms and box plots were used to visualize feature distributions, and a count plot showed the distribution of the target variable. A correlation heatmap was generated to identify highly correlated features.

## Data Processing
Highly correlated features such as 'maxtemp', 'temparature', and 'mintemp' were dropped to avoid multicollinearity. The dataset was then balanced using downsampling, specifically for the majority class ('rainfall' = 1), to ensure equal representation of both classes. The balanced dataset was shuffled.

## Model Training
The data was split into training and testing sets. A `RandomForestClassifier` was chosen for the prediction task. Hyperparameter tuning was performed using `GridSearchCV` to find the optimal parameters for the Random Forest model.

## Model Evaluation
The best model from `GridSearchCV` was evaluated using cross-validation on the training set. Its performance on the test set was assessed using accuracy score, confusion matrix, and a classification report (precision, recall, f1-score).

## Prediction on Unknown Data
The trained model was demonstrated by making a prediction on a sample of unknown input data.

## Saving and Loading the Model
The trained model and feature names were saved to a pickle file (`rainfall_prediction_model.pkl`) to enable easy loading and reuse for future predictions without retraining.
