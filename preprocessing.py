# -----------------------------------------
# Data Pre-processing Demonstration Program
# -----------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# -------------------------------
# Step 1: Create a sample dataset
# -------------------------------
data = {
    'Name': ['John', 'Anna', 'Peter', 'Linda', 'James', np.nan],
    'Age': [28, 22, np.nan, 32, 45, 36],
    'Salary': [50000, 54000, 58000, np.nan, 62000, 60000],
    'Department': ['HR', 'IT', 'Finance', 'IT', 'Finance', 'HR']
}
df = pd.DataFrame(data)
print("🔹 Original Dataset:\n", df)

# -------------------------------
# Step 2: Handling Missing Values
# -------------------------------
df['Age'].fillna(df['Age'].mean(), inplace=True)        # Replace NaN in Age with mean
df['Salary'].fillna(df['Salary'].median(), inplace=True) # Replace NaN in Salary with median
df['Name'].fillna("Unknown", inplace=True)              # Replace missing Name with "Unknown"
print("\n🔹 After Handling Missing Values:\n", df)

# -------------------------------
# Step 3: Label Encoding
# -------------------------------
label_encoder = LabelEncoder()
df['Dept_Label'] = label_encoder.fit_transform(df['Department'])
print("\n🔹 After Label Encoding:\n", df)

# -------------------------------
# Step 4: One-Hot Encoding
# -------------------------------
df = pd.get_dummies(df, columns=['Department'])
print("\n🔹 After One-Hot Encoding:\n", df)

# -------------------------------
# Step 5: Feature Scaling
# -------------------------------
# Normalization (Min-Max Scaling)
scaler = MinMaxScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

# Standardization (Z-score)
std_scaler = StandardScaler()
df[['Age', 'Salary']] = std_scaler.fit_transform(df[['Age', 'Salary']])
print("\n🔹 After Feature Scaling:\n", df)

# -------------------------------
# Step 6: Train-Test Split
# -------------------------------
X = df.drop(['Name'], axis=1)  # Features
y = df['Name']                 # Target variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

print("\n🔹 Training Features:\n", X_train)
print("\n🔹 Testing Features:\n", X_test)
