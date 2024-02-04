import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Data Loading and Exploration
def load_and_explore_data(df):

    # Display basic information about the dataset
    print("Dataset Structure:")
    print(df.info())

    # Display summary statistics
    print("\nSummary Statistics:")
    print(df.describe())

    # Display the first few rows of the dataset
    print("\nFirst Few Rows:")
    print(df.head())

    return df

def handle_missing_data(df):
    # Check for missing values
    missing_values = df.isnull().sum()

    # Drop rows with missing values
    df.dropna(inplace=True)

    print(f"\nHandling Missing Data:")
    print(f"Number of rows before handling missing data: {len(df) + missing_values.sum()}")
    print(f"Number of rows after handling missing data: {len(df)}")

    # print(f'\nUnique values in {"UserID"}: {df["UserID"].unique()}')
    print('\nWe have:', len(df["UserID"].unique()), 'unique customers')
    
    # print(f'\nUnique values in {"ProductID"}: {df["ProductID"].unique()}')
    print('\nWe have:', len(df["ProductID"].unique()), 'unique products')
    
    print('\nWe have:', len(df["Category"].unique()), 'unique categories')
    print(f'Unique values in {"Category"}: {df["Category"].unique()}')
    
    print('\nWe have:', len(df["PaymentMethod"].unique()), 'unique Payment Method')
    print(f'Unique values in {"PaymentMethod"}: {df["PaymentMethod"].unique()}')
    
    # Check data types and ranges for numerical columns
    for column in df.select_dtypes(include='number').columns:
        print(f"\nData type and range for '{column}': {df[column].dtype}, min: {df[column].min()}, max: {df[column].max()}")
    
    return df

def create_derived_column(df):
   
    df['Total_Price'] = df['Quantity'] * df['Price']

    print("\nCreating a New Derived Column:")
    print(df[['Quantity', 'Price', 'Total_Price']].head())

    return df

# Detailed Data Analysis
def detailed_data_analysis(df):
    # Analysis of the 'Category' column
    column_to_analyze = 'Category'
    print(f"\nAnalysis of {column_to_analyze}:")
    print(df[column_to_analyze].value_counts())

    # Investigate correlations between numerical variables
    numerical_columns = df.select_dtypes(include='number').columns
    correlation_matrix = df[numerical_columns].corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Visualize correlation matrix using a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Matrix')
    plt.show()

    # Pairplot for numerical columns
    sns.pairplot(df[numerical_columns])
    plt.show()

    df['OrderDate'] = pd.to_datetime(df['OrderDate'])  # Convert OrderDate to datetime
    df.set_index('OrderDate', inplace=True)
    trend_data = df['Quantity'].resample('M').sum()  # Resample by month and sum quantity
    trend_data.plot(title='Monthly Trend Analysis')
    plt.show()

    X = df['Price']
    y = df['Quantity']
    model = sm.OLS(y, sm.add_constant(X)).fit()
    print(model.summary())

    # Data Visualization
    # Distribution plot for numerical columns
    for column in df.select_dtypes(include='number').columns:
        df[column].hist()
        plt.title(f'Distribution of {column}')
        plt.show()

    # Bar chart for categorical columns
    for column in df.select_dtypes(include='object').columns:
        df[column].value_counts().plot(kind='bar')
        plt.title(f'Bar Chart of {column}')
        plt.show()

    return df

# Advanced Analysis
def advanced_analysis(df):
    # Explore trends or patterns using advanced methods (example)
    # ...

    return df

# Data Visualization
def data_visualization(df):
    # Create meaningful visualizations (example)
    # Plot a histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['example_column'], bins=20, kde=True)
    plt.title('Distribution of Example Column')
    plt.xlabel('Example Column Values')
    plt.ylabel('Frequency')
    plt.show()

    # Additional visualizations
    # ...

# Main function
def main():
    # Specify the file path of your dataset
    file_path = 'ecommerce_data.csv'
    df = pd.read_csv(file_path)

    # Load and explore data
    # df = load_and_explore_data(df)

    # Data cleaning and manipulation
    # df = handle_missing_data(df)
    # df = create_derived_column(df)

    # Detailed data analysis
    df = detailed_data_analysis(df)

    # Advanced analysis
    # df = advanced_analysis(df)

    # Data visualization
    # data_visualization(df)

if __name__ == "__main__":
    main()
