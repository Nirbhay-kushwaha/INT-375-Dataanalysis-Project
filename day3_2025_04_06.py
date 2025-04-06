import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"C:\Users\kmrnr\Desktop\tmdb_5000_movies.csv"
df = pd.read_csv(file_path)

# -----------------------------
# Data Cleaning and Preparation
# -----------------------------

# Check for missing values
missing_data = df.isnull().sum()
print("\nMissing Values in Each Column:\n")
print(missing_data[missing_data > 0])

# Fill missing numerical columns with median
df['runtime'].fillna(df['runtime'].median(), inplace=True)

# Convert 'release_date' to datetime format
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Drop rows with missing release dates
df = df.dropna(subset=['release_date'])

# Create a new column for release year
df['release_year'] = df['release_date'].dt.year

# -----------------------------
# NumPy Operations
# -----------------------------

# Example: Calculate ROI (Return on Investment)
df['ROI'] = np.where(df['budget'] > 0, (df['revenue'] - df['budget']) / df['budget'], 0)

# Replace infinite or NaN ROI values with 0
df['ROI'].replace([np.inf, -np.inf], 0, inplace=True)
df['ROI'].fillna(0, inplace=True)

# -----------------------------
# Data Aggregation
# -----------------------------

# Group by release year and calculate average revenue and budget
yearly_data = df.groupby('release_year')[['revenue', 'budget']].mean().reset_index()

# -----------------------------
# Visualization with Matplotlib and Seaborn
# -----------------------------

plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_data, x='release_year', y='revenue', label='Avg Revenue')
sns.lineplot(data=yearly_data, x='release_year', y='budget', label='Avg Budget')
plt.title('Average Revenue and Budget Over the Years')
plt.xlabel('Year')
plt.ylabel('Amount')
plt.legend()
plt.tight_layout()
plt.show()

# ROI Distribution Plot
plt.figure(figsize=(10, 5))
sns.histplot(df['ROI'], bins=50, kde=True, color='green')
plt.title('Distribution of ROI')
plt.xlabel('ROI')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()

# Top 10 Movies with Highest ROI
top_roi = df.sort_values(by='ROI', ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x='ROI', y='title', data=top_roi, hue='title', palette='magma', legend=False)
plt.title('Top 10 Movies with Highest ROI')
plt.xlabel('Return on Investment')
plt.ylabel('Movie Title')
plt.tight_layout()
plt.show()
