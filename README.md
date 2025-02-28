# User Activity Analysis and Prediction

This repository contains Jupyter notebooks for analyzing and predicting user activity based on various user metrics. The notebooks include data preprocessing, feature engineering, exploratory data analysis (EDA), and machine learning model training and evaluation.

## Table of Contents
1. [Introduction](#introduction)
2. [Notebooks](#notebooks)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Feature Engineering](#feature-engineering)
6. [Machine Learning Models](#machine-learning-models)
7. [Model Evaluation](#model-evaluation)
8. [Usage](#usage)
9. [Author](#author)

## Introduction
The goal of these notebooks is to analyze user activity data and build machine learning models to predict whether a user will complete their tasks. The dataset includes various features such as age, device type, subscription tier, and user activity metrics.

## Notebooks
### 1. activity.ipynb
This notebook uses a synthenic data that focuses on analyzing user activity data, performing data preprocessing, feature engineering, and training machine learning models to predict task completion likelihood and also deploy using gradio.

Since all your work is in a Jupyter Notebook, I'll structure the **README** accordingly, assuming the main notebook is named **`ML_Prediction_App.ipynb`**.

---

# **ML Model Prediction with Gradio**

## **Overview**
This project provides an interactive machine learning model prediction system using **Gradio**. The notebook performs:
- **Data Preprocessing**: Handling categorical and numerical features, feature scaling, and selection.
- **Model Training**: Training multiple classification models (Logistic Regression, SVM, KNN, Decision Tree, etc.).
- **Model Deployment**: Creating a Gradio web application for user-friendly model inference.

---

## **Installation**
### **1. Clone the Repository**
```bash
git clone https://github.com/your-repo/ml-prediction-app.git
cd ml-prediction-app
```

### **2. Install Dependencies**
Make sure you have Python and Jupyter Notebook installed, then run:
```bash
pip install -r requirements.txt
```

### **3. Open the Notebook**
Start Jupyter Notebook:
```bash
jupyter notebook
```
Then open **`ML_Prediction_App.ipynb`**.

---

## **Project Structure**
```
ml-prediction-app/
│── models/                    # Folder containing trained model files
│── scaler.pkl                  # Pre-trained scaler for feature scaling
│── selector.pkl                # Pre-trained feature selector
│── label_encoders.pkl          # LabelEncoders for categorical features
│── ML_Prediction_App.ipynb     # Jupyter Notebook with full implementation
│── dataset.csv                 # Sample dataset for training
│── requirements.txt             # Required Python libraries
│── README.md                    # Project documentation
```

---

## **Steps in the Notebook**
### **1. Data Preprocessing**
- **Categorical Encoding**: `experience_level`, `time_of_day`, `device_type`, `subscription_tier` are encoded using `LabelEncoder`.
- **Numerical Features**: Standardization using `StandardScaler()`.
- **Feature Selection**: `SelectKBest` to select top 5 features.

### **2. Model Training**
The following models are trained and saved:
- **Logistic Regression**
- **Support Vector Classifier (SVC)**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**
- Models are saved as `.pkl` files using `joblib`.

### **3. Model Deployment with Gradio**
- A **Gradio interface** is built for easy predictions.
- Users can **select a model**, **input feature values**, and **get predictions**.

---

## **How to Use**
### **1. Train Models**
Run all cells in **`ML_Prediction_App.ipynb`** to train models and save preprocessing tools.

### **2. Run the Gradio App**
At the end of the notebook, execute:
```python
gr.Interface(
    fn=predict_class,
    inputs=[model_dropdown] + input_fields,
    outputs="text",
    title="ML Model Prediction",
    description="Select a model and input feature values to get a prediction."
).launch()
```
This will launch a **Gradio web UI**.

---

## **Dependencies**
Ensure you have the following Python libraries installed:
```bash
numpy
pandas
scikit-learn
joblib
gradio
```

If missing, install with:
```bash
pip install numpy pandas scikit-learn joblib gradio
```

---

## **Future Improvements**
- Extend to more advanced models (e.g., Random Forest, XGBoost).
- Improve UI with more visualization.
- Deploy as a standalone web app using **FastAPI** or **Flask**.



### 2. fitbit_data_on_kaggle.ipynb
This notebook uses a sample dataset from Kaggle representing user activity, such as task completion times, app usage, or wellness metrics. It includes data preprocessing, feature engineering, model development, and evaluation.

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
The notebooks train and evaluate several machine learning models:
- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- XGBoost
- Ensemble Learning using Voting Classifier

## Model Evaluation
Models are evaluated using metrics such as accuracy, precision, recall, and F1 score. The best-performing models are saved for deployment.

## Usage
To run the notebooks, ensure you have the required libraries installed. You can install them using:
```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib gradio joblib
```
Open the notebooks in Jupyter and run the cells sequentially to perform the analysis and train the models.

## Author
---

## **Author**
**Adetunji Philip Adetayo**  
📧 Email: adetunjiphilip47@gmail.com 
🔗 [LinkedIn](https://www.linkedin.com/in/adetunji-philip-adetayo/)  

---
