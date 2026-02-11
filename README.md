# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and perform preprocessing by removing unnecessary columns like sl_no and salary.
2. Convert categorical data into numerical form using Label Encoding.
3. Split the dataset into training and testing sets and train the Logistic Regression model using the training data.
4. Predict and evaluate the model using the test data and calculate accuracy and confusion matrix.

## Program:
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("Placement_Data.csv")

# Drop serial number and salary
# Salary is dropped because it contains NaNs for unplaced students
# and is a target-leaking variable (you only have it if status is 'Placed')
drops = ['sl_no', 'salary']
data = data.drop([c for c in drops if c in data.columns], axis=1)

# Convert categorical columns into numbers
le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

# Define features and target
X = data.drop('status', axis=1)
y = data['status']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

## Output:
<img width="933" height="349" alt="Screenshot 2026-02-11 112758" src="https://github.com/user-attachments/assets/ffafd1a8-de02-4d5b-9507-246d1789ea9d" />
<img width="796" height="681" alt="Screenshot 2026-02-11 112823" src="https://github.com/user-attachments/assets/2faba4d2-0df2-4039-957a-4e8319405711" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
