# The central code for PAP-II ! =) 
# EDA FOR ECONOMIC INDICATORS. 
# import libraries.....
import pandas as pd
import matplotlib.pyplot as plt
from utils import FredIndicatorFetcher as FIF


api_key = '2df41cfada1473ef26fa8dede4c9bef5 '
fetcher = FIF(api_key)
data_tlei = fetcher.fetch_indicator('TLEI')
data_cpi = fetcher.fetch_indicator('CPI')
data_gdp = fetcher.fetch_indicator('GDP')
data_cci = fetcher.fetch_indicator('CCI')
data_cei = fetcher.fetch_indicator('CEI')
fetcher.save_to_excel({'TLEI': data_tlei, 'CPI': data_cpi, 
                       'GDP': data_gdp, 'CCI': data_cci,
                       'CEI': data_cei}, 'economic_indicatorsv2.xlsx')




















# Load the Excel file
file_path = 'PAP-ERS/Data/economic_indicators_data.xlsx'
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

# Set the date as the index for better plotting
data_df['Date'] = pd.to_datetime(data_df['Date'])
data_df.set_index('Date', inplace=True)

# Plotting each indicator
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(data_df.index, data_df['CLI'], label='CLI', color='blue')
plt.title('Composite Leading Indicator (CLI)')
plt.xlabel('Date')
plt.ylabel('CLI')

plt.subplot(2, 2, 2)
plt.plot(data_df.index, data_df['BCI'], label='BCI', color='orange')
plt.title('Business Confidence Indicator (BCI)')
plt.xlabel('Date')
plt.ylabel('BCI')

plt.subplot(2, 2, 3)
plt.plot(data_df.index, data_df['GDP'], label='GDP', color='green')
plt.title('Gross Domestic Product (GDP)')
plt.xlabel('Date')
plt.ylabel('GDP')

plt.subplot(2, 2, 4)
plt.plot(data_df.index, data_df['CCI'], label='CCI', color='red')
plt.title('Consumer Confidence Index (CCI)')
plt.xlabel('Date')
plt.ylabel('CCI')

plt.tight_layout()
plt.show()

# Correlation Matrix

# Correlation matrix
correlation_matrix = data_df.corr()

# Plotting the correlation matrix
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar(label='Correlation coefficient')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix of Economic Indicators')
plt.show()

correlation_matrix

