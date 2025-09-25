# --------------------------------------
# Task 1: Load and Explore the Dataset
# --------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Error handling for file loading
try:
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: File not found. Please check the dataset path.")

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check data types and structure
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Clean dataset if needed (Iris dataset has no missing values)
df.fillna(df.mean(numeric_only=True), inplace=True)

# --------------------------------------
# Task 2: Basic Data Analysis
# --------------------------------------

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Group by species and compute mean of numerical columns
grouped_means = df.groupby('species').mean()
print("\nMean values grouped by species:")
print(grouped_means)

# Observations: Example insights
print("\nObservation: Iris-virginica species has the highest average petal length and width.")

# --------------------------------------
# Task 3: Data Visualization
# --------------------------------------

# 1. Line chart - Average sepal length per species
plt.figure(figsize=(6,4))
grouped_means['sepal length (cm)'].plot(kind='line', marker='o')
plt.title('Average Sepal Length per Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.grid(True)
plt.show()

# 2. Bar chart - Average petal length per species
plt.figure(figsize=(6,4))
grouped_means['petal length (cm)'].plot(kind='bar', color='skyblue')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# 3. Histogram - Distribution of sepal length
plt.figure(figsize=(6,4))
plt.hist(df['sepal length (cm)'], bins=10, color='lightgreen', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter plot - Sepal length vs Petal length
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='viridis')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()
# 5. Box plot - Sepal width by species
plt.figure(figsize=(6,4))
