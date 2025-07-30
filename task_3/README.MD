# Titanic Survival Prediction

This project is focused on building a supervised machine learning model to predict the survival of passengers on the Titanic. It is part of an internship task and demonstrates the full machine learning pipeline including data cleaning, feature engineering, model training, evaluation, and cross-validation.

## 📁 Project Structure

```
├── data/
│ └── titanic.csv # Raw dataset
├── dataset/
│ └── titanic_processed.csv # Preprocessed dataset with feature engineering
├── models/
│ ├── rf_model.pkl # Trained Random Forest model
│ └── scaler.pkl # Scaler used for Logistic Regression
├── notebooks/
│ ├── 1_data_cleaning.ipynb # Data preprocessing and feature engineering
│ ├── 2_modeling.ipynb # Model training and evaluation
│ └── 3_cross_validation.ipynb # Cross-validation and model saving
├── README.md # Project documentation
```


---

## 1. Objective

Build a machine learning model to predict whether a passenger survived the Titanic disaster based on attributes such as age, gender, ticket class, family size, and fare.

---

## 2. Dataset Information

The dataset used is based on the [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic/data). Key features include:

- `Pclass` - Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Sex` - Gender of the passenger
- `Age` - Age in years
- `SibSp` - Number of siblings/spouses aboard
- `Parch` - Number of parents/children aboard
- `Fare` - Passenger fare
- `Embarked` - Port of embarkation (C, Q, S)
- `Cabin` - Cabin number (many missing)
- `Name`, `Ticket`, `PassengerId` - Dropped as not useful

---

## 3. Approach Overview

### Notebook 1: Data Cleaning and Feature Engineering

- Handled missing values:
  - `Age`: Imputed using median age grouped by `Pclass` and `Sex`
  - `Embarked`: Imputed with mode
  - `Cabin`: Dropped due to high missingness
- Created new features:
  - `Title` extracted from `Name`
  - `FamilySize` = `SibSp` + `Parch` + 1
  - `IsAlone`: Boolean for solo passengers
- Encoded categorical variables using `pd.get_dummies`
- Final processed dataset saved to `titanic_processed.csv`

### Notebook 2: Model Training and Evaluation

- Applied two models:
  - **Logistic Regression** with feature scaling using `StandardScaler`
  - **Random Forest Classifier** without scaling
- Evaluation Metrics:
  - Accuracy
  - Confusion Matrix
  - ROC AUC Score
  - Classification Report

### Notebook 3: Cross-Validation and Saving Models

- 5-Fold Cross-Validation used to evaluate model stability
- Saved the best-performing model (Random Forest) and scaler using `joblib`

---

## 4. Results

| Metric                    | Logistic Regression | Random Forest |
|---------------------------|---------------------|----------------|
| Accuracy (Holdout Set)    | ~81%                | ~83%           |
| ROC AUC Score             | ~0.86               | ~0.89          |
| 5-Fold CV Accuracy        | ~80.9%              | ~81.7%         |

**Random Forest** outperformed Logistic Regression and was selected for deployment.

---

## 5. Key Learnings & Justifications

- Feature engineering had a noticeable impact on model performance.
- Logistic Regression required feature scaling and more iterations for convergence (`max_iter=5000`) due to default solver limitations.
- Random Forest was robust to scaling and captured non-linear relationships better.
- Cross-validation provided a more reliable performance estimate beyond the holdout test set.

---

## 6. How to Run

1. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. Place raw dataset as data/titanic.csv

3. Run the notebooks in order:

    01_eda_cleaning.ipynb

    02_post_processing.ipynb

    03_model_training.ipynb

4. Trained model and scaler will be saved in the models/ directory.

## 7. Dependencies
Python 3.8+

pandas

numpy

scikit-learn

matplotlib

seaborn

joblib

## Author
Mohit Narayan Vashisth |
Data Science Intern @codsoft |
[LinkedIn](https://www.linkedin.com/in/mohit-n-vashisth/)
| [GitHub](https://github.com/mohit-vashisth)
