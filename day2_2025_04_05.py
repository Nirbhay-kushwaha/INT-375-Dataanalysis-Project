import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the TMDB dataset from the specified path
file_path = r"C:\Users\kmrnr\Desktop\tmdb_5000_movies.csv"
df = pd.read_csv(file_path)

# Convert release_date to datetime format to enable year-wise analysis
if 'release_date' in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Remove movies that have zero revenue or zero budget to clean the data
df = df[(df['revenue'] > 0) & (df['budget'] > 0)]

# Calculate ROI = revenue / budget for each movie
df['ROI'] = df['revenue'] / df['budget']

# Find the top 10 movies with the highest ROI
top_roi = df.sort_values(by='ROI', ascending=False).head(10)
print("Top 10 Movies by ROI:")
print(top_roi[['title', 'budget', 'revenue', 'ROI']])

# Visualize the top 10 ROI movies using a horizontal bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='ROI', y='title', hue='title', data=top_roi, palette='coolwarm', dodge=False, legend=False)
plt.title('Top 10 Movies by ROI')
plt.xlabel('ROI (Revenue / Budget)')
plt.ylabel('Movie Title')
plt.tight_layout()
plt.show()

# Extract the release year for each movie to analyze yearly trends
df['year'] = df['release_date'].dt.year

# Group the data by year and calculate average budget and revenue
yearly_data = df.groupby('year')[['budget', 'revenue']].mean().dropna().reset_index()

# Plot the average budget and revenue by year to understand the trend
plt.figure(figsize=(12, 6))
sns.lineplot(x='year', y='budget', data=yearly_data, label='Average Budget')
sns.lineplot(x='year', y='revenue', data=yearly_data, label='Average Revenue')
plt.title('Average Budget and Revenue by Year')
plt.xlabel('Year')
plt.ylabel('Amount ($)')
plt.legend()
plt.tight_layout()
plt.show()
