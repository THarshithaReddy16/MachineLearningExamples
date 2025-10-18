import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset (student scores in different subjects)
data = {
    'Student': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    'Math': [88, 92, 80, 89, 100, 67, 78, 85],
    'Science': [90, 85, 88, 95, 92, 70, 75, 80],
    'English': [70, 78, 85, 82, 89, 60, 65, 72]
}

# Create DataFrame
df = pd.DataFrame(data)

# 1. Line Plot
plt.figure(figsize=(8,5))
plt.plot(df['Student'], df['Math'], marker='o', label='Math')
plt.plot(df['Student'], df['Science'], marker='s', label='Science')
plt.plot(df['Student'], df['English'], marker='^', label='English')
plt.title("Line Plot of Student Scores")
plt.xlabel("Students")
plt.ylabel("Scores")
plt.legend()
plt.show()

# 2. Bar Chart
plt.figure(figsize=(8,5))
plt.bar(df['Student'], df['Math'], color='skyblue')
plt.title("Bar Chart - Math Scores")
plt.xlabel("Students")
plt.ylabel("Scores")
plt.show()

# 3. Histogram
plt.figure(figsize=(8,5))
plt.hist(df['Math'], bins=5, color='lightgreen', edgecolor='black')
plt.title("Histogram of Math Scores")
plt.xlabel("Score Range")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot
plt.figure(figsize=(8,5))
plt.scatter(df['Math'], df['Science'], color='red')
plt.title("Scatter Plot - Math vs Science")
plt.xlabel("Math Scores")
plt.ylabel("Science Scores")
plt.show()

# 5. Box Plot
plt.figure(figsize=(8,5))
sns.boxplot(data=df[['Math','Science','English']])
plt.title("Box Plot of Subject Scores")
plt.ylabel("Scores")
plt.show()

# 6. Heatmap (Correlation between subjects)
plt.figure(figsize=(6,4))
sns.heatmap(df[['Math','Science','English']].corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap - Correlation Between Subjects")
plt.show()




