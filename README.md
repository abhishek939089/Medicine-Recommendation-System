# Personalized Medical Recommendation System with Machine Learning

Welcome to our cutting-edge **Personalized Medical Recommendation System**, a powerful platform designed to assist users in understanding and managing their health. Leveraging machine learning, the system analyzes user-input symptoms to predict potential diseases accurately and offers personalized recommendations.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Models and Performance](#models-and-performance)
7. [Contributors](#contributors)
8. [License](#license)

---

## Overview

The Personalized Medical Recommendation System utilizes advanced machine learning techniques to predict diseases based on symptoms and provide tailored recommendations, including medications, diet plans, precautions, and exercises. This end-to-end platform empowers users to take control of their health with precision and ease.

---

## Features

- Predict diseases based on symptoms with high accuracy.
- Recommendations include:
  - Medications
  - Diet plans
  - Precautions
  - Workout suggestions
- Interactive symptom input and prediction logic.
- Integration of multiple machine learning models for performance comparison.
- Easily extensible and customizable for further enhancements.

---

## Dataset

The dataset used contains **4,920 records** and **133 columns**, including symptoms and their corresponding diagnoses (prognosis). It is loaded as follows:

```python
import pandas as pd

# Load dataset
dataset = pd.read_csv('Training.csv')
print(dataset.shape)  # Output: (4920, 133)
```

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/abhishek939089/medicine-recommendation-system.git
   cd medicine-recommendation-system
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the following files are in the project directory:
   - `Training.csv` (Main dataset)
   - `symptoms_df.csv` (Symptom descriptions)
   - `precautions_df.csv` (Precautionary measures)
   - `workout_df.csv` (Workout suggestions)
   - `description.csv` (Disease descriptions)
   - `medications.csv` (Medication data)
   - `diets.csv` (Diet plans)

---

## Usage

1. **Train and Evaluate Models:**

   ```python
   from sklearn.svm import SVC
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score

   # Split data
   X = dataset.drop('prognosis', axis=1)
   y = dataset['prognosis']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

   # Train SVC model
   svc = SVC(kernel='linear')
   svc.fit(X_train, y_train)

   # Evaluate
   predictions = svc.predict(X_test)
   print("Accuracy:", accuracy_score(y_test, predictions))
   ```

2. **Make Predictions:**

   ```python
   import pickle

   # Save and load the model
   pickle.dump(svc, open('svc.pkl', 'wb'))
   svc = pickle.load(open('svc.pkl', 'rb'))

   # Predict disease
   sample = X_test.iloc[0].values.reshape(1, -1)
   print("Predicted Disease:", svc.predict(sample))
   ```

3. **Get Recommendations:**

   ```python
   # Helper function
   def helper(disease):
       desc = description[description['Disease'] == disease]['Description'].values[0]
       precautions_list = precautions[precautions['Disease'] == disease].values.tolist()
       medications_list = medications[medications['Disease'] == disease].values.tolist()
       diet_list = diets[diets['Disease'] == disease].values.tolist()
       workout_list = workout[workout['disease'] == disease].values.tolist()
       return desc, precautions_list, medications_list, diet_list, workout_list

   # Example usage
   disease = "Psoriasis"
   details = helper(disease)
   print(details)
   ```

---

## Models and Performance

The following models were trained and evaluated:

| Model               | Accuracy |
|---------------------|----------|
| Support Vector Machine (SVM) | 1.0      |
| Random Forest        | 1.0      |
| Gradient Boosting    | 1.0      |
| K-Nearest Neighbors  | 1.0      |
| Multinomial Naive Bayes | 1.0 |

---

## Contributors

- **Your Name** - [GitHub Profile](https://github.com/abhishek939089)

---
