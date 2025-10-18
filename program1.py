import pandas as pd
data = pd.read_csv("sample_data.csv")   # Make sure file exists in same folder
print("Data imported from CSV:")
print(data.head())   # Display first 5 rows

# Step 2: Import data from Excel
# Requires: pip install open pyxl

excel_data = pd.read_excel("sample_data.xlsx")
print("\nData imported from Excel:")
print(excel_data.head())

# Step 3: Export data to CSV

data.to_csv("exported_data.csv", index=False)
print("\nData exported to 'exported_data.csv'")

# Step 4: Export data to Excel

excel_data.to_excel("exported_data.xlsx", index=False)
print("Data exported to 'exported_data.xlsx'")
