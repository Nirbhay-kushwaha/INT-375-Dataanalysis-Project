import pandas as pd

# Load the dataset
file_path = r"C:\Users\kmrnr\Desktop\tmdb_5000_movies.csv"
df = pd.read_csv(file_path)

# Display basic information
print("Dataset Info:")
print(df.info())

# Display first few rows
print("\nFirst 5 Rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())
