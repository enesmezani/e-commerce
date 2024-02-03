import pandas as pd
import matplotlib.pyplot as plt

file_path = 'C:\\Users\\Perdorues\\Downloads\\ecommerce_data.csv'

df = pd.read_csv(file_path)

# the first few rows of the dataset
rows = df.head()
print("\nsome rows")
print(rows)

# general information about the dataset
generealInformation = df.info()
print("\ngeneral information")
print(generealInformation)

# summary statistics for numerical columns
statistics = df.describe()
print("\nsummary statistics")
print(statistics)

# missing values
missingValues = df.isnull().sum()
print("\nmissing values")
print(missingValues)

# unique values in categorical columns
print("\nunique values")
for column in df.select_dtypes(include='object').columns:
    print(f'\nUnique values in {column}: {df[column].unique()}')

# Correlation
# correlation_matrix = df.corr()
# print("\nCorrelation Matrix:")
# print(correlation_matrix)

# Count and Frequency
for column in df.columns:
    if df[column].dtype == 'object':
        value_counts = df[column].value_counts()
        print(f"\nValue counts for '{column}':")
        print(value_counts)

# Distribution Plots (for numerical columns)
for column in df.select_dtypes(include='number').columns:
    df[column].hist()
    plt.title(f'Distribution of {column}')
    plt.show()

# GroupBy and Aggregation (mean)
grouped_data_mean = df.groupby('group_column')['value_column'].mean()
print("\nGrouped Data (mean):")
print(grouped_data_mean)

# GroupBy and Aggregation (median)
grouped_data_median = df.groupby('group_column')['value_column'].median()
print("\nGrouped Data (median):")
print(grouped_data_median)

# GroupBy and Aggregation (standard deviation)
grouped_data_std = df.groupby('group_column')['value_column'].std()
print("\nGrouped Data (standard deviation):")
print(grouped_data_std)

# Visualize mean values using a bar chart
grouped_data_mean.plot(kind='bar', rot=45)
plt.title('Mean Value by Group')
plt.xlabel('Group')
plt.ylabel('Mean Value')
plt.show()

# Visualize median values using a bar chart
grouped_data_median.plot(kind='bar', rot=45, color='orange')
plt.title('Median Value by Group')
plt.xlabel('Group')
plt.ylabel('Median Value')
plt.show()

# Visualize standard deviation using a bar chart
grouped_data_std.plot(kind='bar', rot=45, color='green')
plt.title('Standard Deviation by Group')
plt.xlabel('Group')
plt.ylabel('Standard Deviation')
plt.show()