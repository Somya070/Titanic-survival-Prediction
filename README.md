# Titanic-survival-Prediction

## Overview
The Titanic Survival Prediction project aims to develop a machine learning model that predicts whether a passenger survived the Titanic disaster based on various features such as age, gender, ticket class, fare, and more. This project utilizes the Titanic dataset to train and evaluate a classification model, specifically a Random Forest Classifier, to achieve high accuracy in survival predictions.

## Dataset
The dataset used in this project is derived from the Titanic passenger list and includes the following features:

- PassengerId: Unique identifier for each passenger
- Survived: Survival status (0 = No, 1 = Yes)
- Pclass: Ticket class (1st, 2nd, 3rd)
- Name: Name of the passenger
- Sex: Gender of the passenger
- Age: Age of the passenger
- SibSp: Number of siblings/spouses aboard
- Parch: Number of parents/children aboard
- Ticket: Ticket number
- Fare: Fare paid for the ticket
- Cabin: Cabin number
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
  
## Project Structure
The project is organized into the following files:

titanic_survival_prediction.py: Main script for data preprocessing, model training, and evaluation.
titanic_model.pkl: Saved model file for future predictions.
README.md: Documentation for the project.

## Steps Taken
1. Data Preprocessing
Loading the Data: The dataset was loaded using Pandas.
Handling Missing Values:
Filled missing values in the 'Age' column with the mean age.
Filled missing values in the 'Fare' column with the mean fare.
Encoding Categorical Variables:
Used LabelEncoder to convert categorical variables ('Sex' and 'Embarked') into numerical format.
Feature Selection: Separated features (X) from the target variable (y).
Data Splitting: Split the dataset into training (80%) and testing (20%) sets.
Normalization: Standardized numerical features using StandardScaler.
2. Model Training
Model Selection: Chose the Random Forest Classifier for its robustness.
Training the Model: Fit the model on the training data.
3. Model Prediction
Making Predictions: Used the trained model to predict survival on the test set.
4. Model Evaluation
Performance Metrics: Evaluated the model using accuracy, precision, recall, and F1-score.
Results: Achieved an accuracy of 1.00, indicating perfect predictions on the test set.
6. Model Saving
Persistence: Saved the trained model using joblib for future use.

## Results
The model achieved the following performance metrics on the test set:

- Accuracy: 1.00
- Precision: 1.00 for both classes
- Recall: 1.00 for both classes
- F1-Score: 1.00 for both classes
  
## Confusion Matrix
A confusion matrix was generated to visualize the model's performance, showing no false positives or false negatives.

## Conclusion
The Titanic Survival Prediction project successfully developed a machine learning model that predicts passenger survival with perfect accuracy. The project demonstrates the end-to-end process of building a machine learning model, from data preprocessing to model evaluation
