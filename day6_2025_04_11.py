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

# -------------------------------------
# Top 15 Most Popular Movies
# -------------------------------------

top_popular = df.sort_values(by='popularity', ascending=False).head(15)
plt.figure(figsize=(12, 6))
sns.barplot(x='popularity', y='title', data=top_popular, palette='rocket')
plt.title('Top 15 Most Popular Movies')
plt.xlabel('Popularity Score')
plt.ylabel('Movie Title')
plt.tight_layout()
plt.show()

# -------------------------------------
# Vote Count Distribution
# -------------------------------------

plt.figure(figsize=(10, 5))
sns.histplot(df['vote_count'], bins=50, color='darkblue', kde=True)
plt.title('Distribution of Vote Counts')
plt.xlabel('Vote Count')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()

# -------------------------------------
# Top 10 Highest Revenue Movies
# -------------------------------------

top_revenue = df.sort_values(by='revenue', ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x='revenue', y='title', data=top_revenue, palette='inferno')
plt.title('Top 10 Highest Revenue Movies')
plt.xlabel('Revenue')
plt.ylabel('Movie Title')
plt.tight_layout()
plt.show()

# -------------------------------------
# Top 10 Highest Budget Movies
# -------------------------------------

top_budget = df.sort_values(by='budget', ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x='budget', y='title', data=top_budget, palette='flare')
plt.title('Top 10 Highest Budget Movies')
plt.xlabel('Budget')
plt.ylabel('Movie Title')
plt.tight_layout()
plt.show()

# -------------------------------------
# Average Runtime by Genre (Top 8)
# -------------------------------------

genre_runtime = genre_exploded.groupby('genres')['runtime'].mean().sort_values(ascending=False).head(8)

plt.figure(figsize=(10, 6))
sns.barplot(x=genre_runtime.values, y=genre_runtime.index, palette='spring')
plt.title('Top 8 Genres by Average Runtime')
plt.xlabel('Average Runtime (mins)')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()

# -------------------------------------
#  Vote Average Distribution
# -------------------------------------

plt.figure(figsize=(10, 5))
sns.histplot(df['vote_average'], bins=30, kde=True, color='orange')
plt.title('Distribution of Average Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()

# -------------------------------------
#  Movies with Runtime > 180 min
# -------------------------------------

long_movies = df[df['runtime'] > 180].sort_values(by='runtime', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='runtime', y='title', data=long_movies.head(10), palette='Blues_r')
plt.title('Top 10 Longest Movies (Runtime > 180 min)')
plt.xlabel('Runtime (mins)')
plt.ylabel('Movie Title')
plt.tight_layout()
plt.show()

# -------------------------------------
#  Scatterplot: Popularity vs Vote Average
# -------------------------------------

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='popularity', y='vote_average', hue='release_year', palette='coolwarm', legend=False)
plt.title('Popularity vs Average Vote')
plt.xlabel('Popularity')
plt.ylabel('Average Vote')
plt.tight_layout()
plt.show()

# -------------------------------------
# ROI vs Vote Average
# -------------------------------------

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='ROI', y='vote_average', hue='release_year', palette='plasma', legend=False)
plt.title('ROI vs Average Vote')
plt.xlabel('Return on Investment')
plt.ylabel('Average Vote')
plt.tight_layout()
plt.show()




















