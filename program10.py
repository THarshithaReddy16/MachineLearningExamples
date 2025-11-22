import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

data = pd.DataFrame({
    'Experience': [1, 3, 5, 7, 9, 11, 13, 15],
    'Age': [22, 25, 28, 32, 36, 40, 45, 50],
    'Salary': [25000, 29000, 35000, 39000, 45000, 49000, 56000, 62000]
})

print("----- Dataset -----")
print(data, "\n")

X = data[['Experience', 'Age']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("----- Model Evaluation -----")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred), "\n")

result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("----- Actual vs Predicted -----")
print(result, "\n")

X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()
print("----- Statsmodels Summary -----")
print(model_sm.summary())
