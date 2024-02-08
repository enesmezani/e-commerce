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
    print(f"\nHandling Missing Data:")
    print(f"Missing values in each column:\n{df.isnull().sum()}")

    # Drop rows with missing values
    df.dropna(inplace=True)

    print('\nWe have:', len(df["UserID"].unique()), 'unique customers')
    
    print('\nWe have:', len(df["ProductID"].unique()), 'unique products')
    
    print('\nWe have:', len(df["Category"].unique()), 'unique categories')
    print(f'Unique values in {"Category"}: {df["Category"].unique()}')
    
    print('\nWe have:', len(df["PaymentMethod"].unique()), 'unique Payment Method')
    print(f'Unique values in {"PaymentMethod"}: {df["PaymentMethod"].unique()}')
    
    # Check data types and ranges for numerical columns
    for column in df.select_dtypes(include='number').columns:
        print(f"\nData type and range for '{column}': {df[column].dtype}, min: {df[column].min()}, max: {df[column].max()}")
    
    #Duplicate rows
    print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
    
    return df

def create_derived_column(df):
   
    df['OrderPrice'] = df['Quantity'] * df['Price']

    print("\nCreating a New Derived Column:")
    print(df[['Quantity', 'Price', 'OrderPrice']].head())
    
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['Year'] = df['OrderDate'].dt.year
    df['Month'] = df['OrderDate'].dt.month
    df['Day'] = df['OrderDate'].dt.day

    print("\nCreating New Derived Columns:")
    print(df[['OrderDate', 'Year', 'Month', 'Day']].head(50))

    #Number of months
    print('\nNumber of months:', len(df['Month'].unique()))

    return df

# Detailed Data Analysis
def detailed_data_analysis(df):
    # #Average Price per Category
    # print('\nAverage Price per Category:')
    # print(df.groupby('Category')['Price'].mean())

    # #Average Quantity per Category
    # print('\nAverage Quantity per Category:')
    # print(df.groupby('Category')['Quantity'].mean())

    # #Number of Orders per Payment Method 
    # df['PaymentMethod'].value_counts().plot(kind='bar')
    # plt.show()
    # Number of Orders per Category
    # df['Category'].value_counts().plot(kind='bar')
    # plt.show()

    # df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    # df.set_index('OrderDate', inplace=True)
    # trend_data = df['OrderPrice'].resample('ME').sum()
    # trend_data.plot(title='Monthly Order Price Trend Analysis')
    # plt.show()

    # df['OrderDate'] = pd.to_datetime(df['OrderDate'])  # Convert OrderDate to datetime
    # df.set_index('OrderDate', inplace=True)
    # trend_data = df['Quantity'].resample('ME').sum()  # Resample by month and sum quantity
    # trend_data.plot(title='Monthly Quantity Trend Analysis')
    # plt.show()


    #Top spending customers
    # top_customers = df.groupby('UserID')['OrderPrice'].sum().sort_values(ascending=False)
    # print(top_customers.head())
    # # #Top selling products
    # top_products = df.groupby('ProductID')['Quantity'].sum().sort_values(ascending=False)
    # print(top_products.head())
    # # #Top selling categories
    # top_categories = df.groupby('Category')['Quantity'].sum().sort_values(ascending=False)
    # print(top_categories.head())

    # #Correlation between Price and Quantity
    # correlation = df[['OrderPrice', 'Quantity']].corr()
    # print(correlation)

    # # Investigate correlations between numerical variables
    # numerical_columns = df.select_dtypes(include='number').columns
    # correlation_matrix = df[numerical_columns].corr()
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    # plt.title('Correlation Matrix')
    # plt.show()

    # # Data Visualization
    # # Distribution plot for numerical columns
    # for column in df.select_dtypes(include='number').columns:
    #     df[column].hist()
    #     plt.title(f'Distribution of {column}')
    #     plt.show()

    # df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    # df.set_index('OrderDate', inplace=True)

    #Total sales based on day over time
    # df.resample('D')['OrderPrice'].sum().plot()
    # plt.title('Daily Sales Analysis')
    # plt.xlabel('Order Date')
    # plt.ylabel('Sales')
    # plt.show()

    #Average price based on month over time
    # df.resample('ME')['Price'].mean().plot()
    # plt.title('Average Price of Product Over Time')
    # plt.xlabel('Order Date')
    # plt.ylabel('Average Price')
    # plt.show()

    #Number of orders based on month over time
    # df.resample('ME').size().plot()
    # plt.title('Number of Orders Over Time')
    # plt.xlabel('Order Date')
    # plt.ylabel('Number of Orders')
    # plt.show()

    #Pie charts 

    #Sales by category
    # category_sales = df.groupby('Category')['OrderPrice'].sum()
    # plt.pie(category_sales, labels = category_sales.index, autopct='%1.1f%%')
    # plt.title('Revenue by Category')
    # plt.show()

    #Orders by Payment Method
    # payment_counts = df['PaymentMethod'].value_counts()
    # plt.pie(payment_counts, labels = payment_counts.index, autopct='%1.1f%%')
    # plt.title('Orders by Payment Method')
    # plt.show()

    #Quantiy Sold by Category 
    # product_sales = df.groupby('Category')['Quantity'].sum()
    # plt.pie(product_sales, labels = product_sales.index, autopct='%1.1f%%')
    # plt.title('Quantity of Products by Category')
    # plt.show()

    #Sales Heatmap by Month and Category
    # df['Month'] = df['OrderDate'].dt.month
    # sales_pivot = df.pivot_table(values='Price', index='Month', columns='Category', aggfunc='sum')
    # sns.heatmap(sales_pivot, annot=True, cmap='coolwarm')
    # plt.title('Sales Heatmap by Month and Category')
    # plt.show()

    # Scatter

    #Price vs Quantity Sold
    # plt.scatter(df['Price'], df['Quantity'])
    # plt.title('Price vs Quantity Sold')
    # plt.xlabel('Price')
    # plt.ylabel('Quantity Sold')
    # plt.show()
    
    #Product Price Distribution
    # plt.hist(df['Price'], bins=10, edgecolor='black')
    # plt.title('Product Price Distribution')
    # plt.xlabel('Price')
    # plt.ylabel('Frequency')
    # plt.show()

    #Quantity Sold Distribution
    # plt.hist(df['Quantity'], bins=10, edgecolor='black')
    # plt.title('Quantity Sold Distribution')
    # plt.xlabel('Quantity Sold')
    # plt.ylabel('Frequency')
    # plt.show()

    #Total Sales Distribution
    # plt.hist(df['OrderPrice'], bins=10, edgecolor='black')
    # plt.title('Total Sales Distribution')
    # plt.xlabel('Total Sales')
    # plt.ylabel('Frequency')
    # plt.show()

    return df

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
    # df = detailed_data_analysis(df)

if __name__ == "__main__":
    main()
