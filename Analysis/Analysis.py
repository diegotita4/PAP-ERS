# The central code for PAP-II ! =) 
# EDA FOR ECONOMIC INDICATORS. 
# import libraries.....

import pandas as pd

# Load the Excel file
file_path = 'Data/economic_indicators_data.xlsx'
excel_data = pd.ExcelFile(file_path)

# Display sheet names to understand the structure
sheet_names = excel_data.sheet_names
sheet_names

# Load the data from the 'data' sheet
data_df = pd.read_excel(excel_data, sheet_name='data')

# Display the first few rows of the dataframe
data_df.head()

# Summary statistics of the dataset
summary_stats = data_df.describe()

# Check for missing values
missing_values = data_df.isnull().sum()

summary_stats, missing_values
