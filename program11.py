import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

data = pd.DataFrame({
    'Age': [22, 25, 47, 52, 46, 56, 55, 60, 45, 34],
    'Salary': [20000, 25000, 50000, 52000, 48000, 60000, 58000, 62000, 52000, 42000],
    'Purchased': [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
})

print("----- Dataset -----")
print(data, "\n")

X = data[['Age', 'Salary']]
y = data['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("----- Model Evaluation -----")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

result = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print("----- Actual vs Predicted -----")
print(result, "\n")

new_data = pd.DataFrame({'Age': [30, 50], 'Salary': [30000, 55000]})
new_pred = model.predict(new_data)
print("----- New Predictions -----")
print(pd.concat([new_data, pd.Series(new_pred, name='Purchased_Prediction')], axis=1))
