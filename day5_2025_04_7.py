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
df['runtime'] = df['runtime'].fillna(df['runtime'].median())

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
df['ROI'] = df['ROI'].replace([np.inf, -np.inf], 0)
df['ROI'] = df['ROI'].fillna(0)

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

# -------------------------------------
# Genre Analysis: Top 10 Common Genres
# -------------------------------------

import ast

# Safely convert stringified list to Python list
df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)] if pd.notnull(x) else [])

# Explode genres to create a row for each genre per movie
genre_exploded = df.explode('genres')

# Count the frequency of each genre
top_genres = genre_exploded['genres'].value_counts().head(10)

# Plot Top 10 Genres
plt.figure(figsize=(10, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, hue=top_genres.index, palette='viridis', legend=False)
plt.title('Top 10 Movie Genres')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()

# -------------------------------------
# Runtime Distribution Plot
# -------------------------------------
plt.figure(figsize=(10, 5))
sns.histplot(df['runtime'], bins=40, kde=True, color='purple')
plt.title('Distribution of Movie Runtimes')
plt.xlabel('Runtime (minutes)')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()

# -------------------------------------
# Correlation Heatmap for Numeric Features
# -------------------------------------

# Select numerical columns for correlation
numeric_cols = ['budget', 'revenue', 'popularity', 'vote_average', 'vote_count', 'runtime', 'ROI']
corr = df[numeric_cols].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Movie Features')
plt.tight_layout()
plt.show()
# -------------------------------------
# Movies Released Per Year
# -------------------------------------

movies_per_year = df['release_year'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x=movies_per_year.index, y=movies_per_year.values, color='teal')
plt.title('Number of Movies Released Each Year')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()

# -------------------------------------
# Average Vote by Genre (Top 8 Genres)
# -------------------------------------

genre_avg_vote = genre_exploded.groupby('genres')['vote_average'].mean().sort_values(ascending=False).head(8)

plt.figure(figsize=(10, 6))
sns.barplot(x=genre_avg_vote.values, y=genre_avg_vote.index, palette='crest')
plt.title('Top 8 Genres with Highest Average Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()

# -------------------------------------
# Budget vs Revenue Scatter Plot
# -------------------------------------

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='budget', y='revenue', hue='release_year', palette='Spectral', legend=False)
plt.title('Budget vs Revenue Scatter Plot')
plt.xlabel('Budget')
plt.ylabel('Revenue')
plt.tight_layout()
plt.show()
