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
   
    df['Total_Price'] = df['Quantity'] * df['Price']

    print("\nCreating a New Derived Column:")
    print(df[['Quantity', 'Price', 'Total_Price']].head())
    
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
    # df['Category'].value_counts().plot(kind='bar')
    # plt.show()

    # df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    # df.set_index('OrderDate', inplace=True)
    # df.resample('ME')['Total_Price'].sum().plot()
    # plt.show()

    # df['OrderDate'] = pd.to_datetime(df['OrderDate'])  # Convert OrderDate to datetime
    # df.set_index('OrderDate', inplace=True)
    # trend_data = df['Quantity'].resample('ME').sum()  # Resample by month and sum quantity
    # trend_data.plot(title='Monthly Trend Analysis')
    # plt.show()


    #Top spending customers
    # top_customers = df.groupby('UserID')['Total_Price'].sum().sort_values(ascending=False)
    # print(top_customers.head())
    # #Top selling products
    # top_products = df.groupby('ProductID')['Quantity'].sum().sort_values(ascending=False)
    # print(top_products.head())
    # #Top selling categories
    # top_categories = df.groupby('Category')['Quantity'].sum().sort_values(ascending=False)
    # print(top_categories.head())

    # #Correlation between Price and Quantity
    # correlation = df[['Price', 'Quantity']].corr()
    # print(correlation)


    # # Analysis of the 'Category' column
    # column_to_analyze = 'Category'
    # print(f"\nAnalysis of {column_to_analyze}:")
    # print(df[column_to_analyze].value_counts())

    # # Investigate correlations between numerical variables
    # numerical_columns = df.select_dtypes(include='number').columns
    # correlation_matrix = df[numerical_columns].corr()
    # print("\nCorrelation Matrix:")
    # print(correlation_matrix)

    # # Visualize correlation matrix using a heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    # plt.title('Correlation Matrix')
    # plt.show()

    # # Pairplot for numerical columns
    # sns.pairplot(df[numerical_columns])
    # plt.show()

    # X = df['Price']
    # y = df['Quantity']
    # model = sm.OLS(y, sm.add_constant(X)).fit()
    # print(model.summary())

    # # Data Visualization
    # # Distribution plot for numerical columns
    # for column in df.select_dtypes(include='number').columns:
    #     df[column].hist()
    #     plt.title(f'Distribution of {column}')
    #     plt.show()

    # df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    # df.set_index('OrderDate', inplace=True)

    #Total sales based on day over time
    # df.resample('D')['Total_Price'].sum().plot()
    # plt.title('Total Sales Over Time')
    # plt.xlabel('Order Date')
    # plt.ylabel('Sales')
    # plt.show()

    #Quantity of sales based on month over time
    # df.resample('ME')['Quantity'].sum().plot()
    # plt.title('Quantity Of Product Sales Over Time')
    # plt.xlabel('Order Date')
    # plt.ylabel('Sales')
    # plt.show()

    #Average price based on month over time
    # df.resample('ME')['Price'].mean().plot()
    # plt.title('Average Price Over Time')
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
    # category_sales = df.groupby('Category')['Total_Price'].sum()
    # plt.pie(category_sales, labels = category_sales.index, autopct='%1.1f%%')
    # plt.title('Sales by Category')
    # plt.show()

    #Orders by Payment Method
    # payment_counts = df['PaymentMethod'].value_counts()
    # plt.pie(payment_counts, labels = payment_counts.index, autopct='%1.1f%%')
    # plt.title('Orders by Payment Method')
    # plt.show()

    #Quantiy Sold by Category 
    # product_sales = df.groupby('Category')['Quantity'].sum()
    # plt.pie(product_sales, labels = product_sales.index, autopct='%1.1f%%')
    # plt.title('Quantity Sold by Category')
    # plt.show()

    #Sales Heatmap by Month and Category
    # df['Month'] = df['OrderDate'].dt.month
    # sales_pivot = df.pivot_table(values='Price', index='Month', columns='Category', aggfunc='sum')
    # sns.heatmap(sales_pivot, annot=True, cmap='coolwarm')
    # plt.title('Sales Heatmap by Month and Category')
    # plt.show()

    #Price vs Quantity Sold
    # plt.scatter(df['Price'], df['Quantity'])
    # plt.title('Price vs Quantity Sold')
    # plt.xlabel('Price')
    # plt.ylabel('Quantity Sold')
    # plt.show()

    #Price vs Total Sales
    # plt.scatter(df['Price'], df['Total_Price'])
    # plt.title('Price vs Total Sales')
    # plt.xlabel('Price')
    # plt.ylabel('Total Sales')
    # plt.show()

    #Quantity vs Total Sales
    # plt.scatter(df['Quantity'], df['Total_Price'])
    # plt.title('Quantity Sold vs Total Sales')
    # plt.xlabel('Quantity Sold')
    # plt.ylabel('Total Sales')
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
    # plt.hist(df['Total_Price'], bins=10, edgecolor='black')
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
    df = create_derived_column(df)

    # Detailed data analysis
    df = detailed_data_analysis(df)

if __name__ == "__main__":
    main()
