
# User Activity Analysis and Prediction

This Jupyter notebook provides a comprehensive analysis and prediction of user activity based on a dataset containing various user metrics. The notebook includes data preprocessing, feature engineering, exploratory data analysis (EDA), and machine learning model training and evaluation.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Feature Engineering](#feature-engineering)
5. [Machine Learning Models](#machine-learning-models)
6. [Model Evaluation](#model-evaluation)
7. [Gradio Interface](#gradio-interface)
8. [Conclusion](#conclusion)

## Introduction
The goal of this notebook is to analyze user activity data and build machine learning models to predict whether a user will complete their tasks. The dataset includes various features such as age, device type, subscription tier, and user activity metrics.

## Data Preprocessing
Data preprocessing steps include:
- Handling missing values using `SimpleImputer`.
- Converting data types for certain columns.
- Creating new features such as age groups and day of the week.

## Exploratory Data Analysis (EDA)
EDA involves visualizing the data to gain insights into user behavior. Key visualizations include:
- Average number of tasks completed by age group.
- Count of each device type and subscription tier.
- Average time spent per day of the week.
- Distribution of completion rate.
- Focus score per time of day.
- Relationship between tasks attempted and tasks completed.
- Average interruption count per subscription tier.
- Average battery level per device type.

## Feature Engineering
Feature engineering steps include:
- Creating interaction features such as tasks per minute and completion ratio.
- Extracting date and time features.
- Aggregating features using rolling means.

## Machine Learning Models
The notebook trains and evaluates several machine learning models:
- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- XGBoost
- Ensemble Learning using Voting Classifier

## Model Evaluation
Models are evaluated using metrics such as accuracy, precision, recall, and F1 score. The best-performing models are saved for deployment.

## Gradio Interface
A Gradio interface is created to allow users to input feature values and get predictions from the trained models. The interface includes dropdowns for categorical features and number inputs for numerical features.

## Conclusion
The notebook provides a detailed analysis and prediction of user activity. By leveraging machine learning models and interactive visualizations, it offers valuable insights into user behavior and helps in making data-driven decisions.

## Usage
To run the notebook, ensure you have the required libraries installed. You can install them using:
```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib gradio joblib
```
Open the notebook in Jupyter and run the cells sequentially to perform the analysis and train the models.

## Author
Philip - Dexter Cyberlab
