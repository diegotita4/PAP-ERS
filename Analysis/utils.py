# Describe utils.py 
# Import libraries.....
from fredapi import Fred
import pandas as pd
from datetime import datetime
import openpyxl
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yfinance as yf 

# Class to handle FRED API requests
class FredIndicatorFetcher:
    """
        FredIndicatorFetcher is a class for interacting with the FRED API to fetch economic 
        indicator data and save it to an Excel file.

        Attributes:
            api_key (str): The API key for accessing the FRED API.
            fred (Fred): An instance of the Fred class initialized with the provided API key.
            indicators (dict): A dictionary mapping human-readable indicator names to their 
                            corresponding FRED series IDs.
        """
    def __init__(self, api_key):
        self.api_key = api_key
        self.fred = Fred(api_key=self.api_key)
        self.indicators = {
            "TLEI": "M16005USM358SNBR",  # Composite Index of Three Lagging Indicators, Amplitude-Adjusted, Weighted for United States (Monthly)
            #"LEI": "USSLIND",  # Leading Economic Index for United States (Monthly)
            "CPI": "CPIAUCSL",  # Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
            "GDP":  "USALORSGPNOSTSAM",  # Gross Domestic Product for United states (Monthly)
            "CCI": "CSCICP03USM665S",  # Consumer Confidence Index (Monthly)
            "CEI": "USPHCI" # Coincident Economic Activity Index for the United States  (Monthly)
            #"BCI": "USABCI",  # Proxy for Business Confidence Indicator (Monthly)
        }

    def fetch_indicator(self, indicator_key, start_date='1960-01-31', end_date=None):
        if indicator_key not in self.indicators:
            raise ValueError(f"Indicator '{indicator_key}' not found in the predefined indicators list.")
        
        end_date = end_date if end_date else datetime.today().strftime('%Y-%m-%d')
        series_id = self.indicators[indicator_key]
        data = self.fred.get_series(series_id, start_date, end_date)
        return data

    def save_to_excel(self, data_dict, file_name):
        # Define the directory where the file will be saved
        save_directory = 'PAP-ERS/Data/'
        # Ensure the directory exists; if not, create it
        os.makedirs(save_directory, exist_ok=True)
        # Combine the directory and file name to get the full path
        file_path = os.path.join(save_directory, file_name)
        # Initialize an empty DataFrame to combine all indicators
        combined_df = pd.DataFrame()
        # Iterate over each indicator and its data
        for indicator, data in data_dict.items():
            # Convert the Series to a DataFrame
            df = pd.DataFrame(data, columns=[indicator])
            # Combine the data by joining on the index (date)
            if combined_df.empty:
                combined_df = df
            else:
                combined_df = combined_df.join(df, how='outer')
        
        # Save the combined DataFrame to a single sheet in the Excel file
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            combined_df.to_excel(writer, sheet_name='Combined_Indicators')


# Example usage:
# api_key = '2df41cfada1473ef26fa8dede4c9bef5 '
# fetcher = FredIndicatorFetcher(api_key)
# data_cli = fetcher.fetch_indicator('CLI')
# data_lei = fetcher.fetch_indicator('LEI')
# fetcher.save_to_excel({'CLI': data_cli, 'LEI': data_lei}, 'economic_indicators.xlsx')

# Class EDA for datasets:
class EDAAnalysis:
    def __init__(self, file_path, date_column=None, columns_to_analyze=None, sheet_name='Sheet1'):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.date_column = date_column
        self.columns_to_analyze = columns_to_analyze
        self.data_df = None
        self.summary_stats = None
        self.missing_values = None
        self.correlation_matrix = None
    
    def load_data(self):
        # Load the Excel file
        excel_data = pd.ExcelFile(self.file_path)
        
        # Display sheet names to understand the structure
        print(f"Sheet names: {excel_data.sheet_names}")
        
        # Load the data from the specified sheet
        self.data_df = pd.read_excel(excel_data, sheet_name=self.sheet_name)
        
        # Convert the specified column to datetime and set it as the index
        if self.date_column and self.date_column in self.data_df.columns:
            self.data_df[self.date_column] = pd.to_datetime(self.data_df[self.date_column])
            self.data_df.set_index(self.date_column, inplace=True)
        
        # Filter the data to include only the specified columns
        if self.columns_to_analyze:
            self.data_df = self.data_df[self.columns_to_analyze]
    
    def eda_summary(self):
        # Display the first few rows of the dataframe
        print("First few rows of the data:")
        print(self.data_df.head())
        
        # Summary statistics of the dataset
        self.summary_stats = self.data_df.describe()
        print("\nSummary statistics:")
        print(self.summary_stats)
        
        # Check for missing values
        self.missing_values = self.data_df.isnull().sum()
        print("\nMissing values:")
        print(self.missing_values)
    
    def calculate_statistics(self):
        # Calculate and display mean, median, mode, variance
        mean = self.data_df.mean()
        median = self.data_df.median()
        mode = self.data_df.mode().iloc[0]  # Select the first mode
        variance = self.data_df.var()
        
        print("\nMean of each column:")
        print(mean)
        print("\nMedian of each column:")
        print(median)
        print("\nMode of each column:")
        print(mode)
        print("\nVariance of each column:")
        print(variance)
    
    def plot_indicators(self):
        # Plot all indicators in a single graph with custom colors
        plt.figure(figsize=(14, 8))
        colors = ['red', 'blue', 'black', 'green']  # Custom colors for each column
        for col, color in zip(self.columns_to_analyze, colors):
            plt.plot(self.data_df.index, self.data_df[col], label=col, color=color)
        plt.title('All Economic Indicators')
        plt.xlabel('Date')
        plt.ylabel('Indicator Value')
        
        # Move the legend to the bottom left
        plt.legend(loc='lower left')
        plt.show()
        
        # Plot each indicator in separate figures
        for col, color in zip(self.columns_to_analyze, colors):
            plt.figure(figsize=(10, 6))
            plt.plot(self.data_df.index, self.data_df[col], label=col, color=color)
            plt.title(col)
            plt.xlabel('Date')
            plt.ylabel(col)
            plt.legend()
            plt.show()

    
    def plot_correlation_matrix(self):
        # Correlation matrix
        self.correlation_matrix = self.data_df.corr()
        
        # Plotting the correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Selected Indicators')
        plt.show()

    def plot_histograms(self):
        # Plot histograms for all numerical columns
        self.data_df.hist(figsize=(14, 10), bins=20, color='teal', edgecolor='black')
        plt.suptitle('Histograms of Selected Columns')
        plt.show()
    
    def plot_boxplots(self):
        # Plot boxplots for all numerical columns
        plt.figure(figsize=(14, 10))
        sns.boxplot(data=self.data_df, palette='Set2')
        plt.title('Boxplots of Selected Columns')
        plt.xticks(rotation=45)
        plt.show()

    def perform_eda(self):
        # Perform all EDA steps
        self.load_data()
        self.eda_summary()
        self.calculate_statistics()
        self.plot_indicators()
        self.plot_histograms()
        self.plot_boxplots()
        self.plot_correlation_matrix()

class SP500ComparisonEDA:
    def __init__(self, sp500_data, indicators_data, date_column='Date'):
        """
        Class to compare the performance of the S&P 500 with various economic indicators,
        and perform EDA (Exploratory Data Analysis).

        sp500_data: DataFrame containing historical S&P 500 data.
        indicators_data: DataFrame containing economic indicators data.
        date_column: Name of the date column present in both DataFrames.
        """
        self.sp500_data = sp500_data
        self.indicators_data = indicators_data
        self.date_column = date_column

        # Ensure that the dates are in datetime format
        self.sp500_data[self.date_column] = pd.to_datetime(self.sp500_data[self.date_column])
        self.indicators_data[self.date_column] = pd.to_datetime(self.indicators_data[self.date_column])

        # Align both DataFrames based on the date
        self.merged_data = pd.merge(self.sp500_data, self.indicators_data, on=self.date_column, how='inner')

    def plot_comparison(self, sp500_column='^GSPC CLOSE_x', indicators_columns=None):
        """
        Generates plots to compare the S&P 500 with various economic indicators.

        sp500_column: Name of the S&P 500 column.
        indicators_columns: List of names of the economic indicator columns to compare.
        """
        if indicators_columns is None:
            indicators_columns = self.indicators_data.columns.drop(self.date_column).tolist()

        # Create a subplot with 2 rows
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot all indicators in the first subplot
        for indicator in indicators_columns:
            ax1.plot(self.merged_data[self.date_column], self.merged_data[indicator], label=indicator)
        ax1.set_title('Economic Indicators')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)

        # Plot S&P 500 in the second subplot
        ax2.plot(self.merged_data[self.date_column], self.merged_data[sp500_column], label=sp500_column, color='purple')
        ax2.set_title('S&P 500 Close Price')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Close Price')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_histograms(self, sp500_column='^GSPC CLOSE_x', indicators_columns=None):
        """
        Generates histograms for the S&P 500 and various economic indicators.

        sp500_column: Name of the S&P 500 column.
        indicators_columns: List of names of the economic indicator columns.
        """
        if indicators_columns is None:
            indicators_columns = self.indicators_data.columns.drop(self.date_column).tolist()

        # Create a subplot with 2 rows
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot histograms for all indicators in the first subplot
        for indicator in indicators_columns:
            ax1.hist(self.merged_data[indicator].dropna(), bins=30, alpha=0.5, label=indicator)
        ax1.set_title('Histograms of Economic Indicators')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True)

        # Plot histogram for S&P 500 in the second subplot
        ax2.hist(self.merged_data[sp500_column].dropna(), bins=30, color='purple', alpha=0.5, label=sp500_column)
        ax2.set_title('Histogram of S&P 500 Close Price')
        ax2.set_xlabel('Close Price')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_boxplots(self, sp500_column='^GSPC CLOSE_x', indicators_columns=None):
        """
        Generates boxplots for the S&P 500 and various economic indicators.

        sp500_column: Name of the S&P 500 column.
        indicators_columns: List of names of the economic indicator columns.
        """
        if indicators_columns is None:
            indicators_columns = self.indicators_data.columns.drop(self.date_column).tolist()

        # Create a subplot with 2 rows
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot boxplots for all indicators in the first subplot
        ax1.boxplot([self.merged_data[indicator].dropna() for indicator in indicators_columns], vert=False, labels=indicators_columns)
        ax1.set_title('Boxplots of Economic Indicators')
        ax1.set_xlabel('Value')
        ax1.grid(True)

        # Plot boxplot for S&P 500 in the second subplot
        ax2.boxplot(self.merged_data[sp500_column].dropna(), vert=False, labels=[sp500_column], patch_artist=True, boxprops=dict(facecolor='purple'))
        ax2.set_title('Boxplot of S&P 500 Close Price')
        ax2.set_xlabel('Close Price')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

# Class 
        
class HistoricalDataDownloader:
    """
    This class is used to download historical data from a specified source.

    Attributes:
    ----------
    data_source : str
    The source of the historical data.
    data_url : str
    The URL of the historical data.
    data_type : str
    The type of the historical data.
    data_format : str
    The format of the historical data.
    data_path : str
    The path where the historical data will be saved.
    """
    def __init__(self, companies):
            self.companies = companies
            self.data = pd.DataFrame()
    
    def download_data(self, start_date="2000-01-01", end_date=None):
        """
        Downloads historical 'Close' prices for the S&P 500 index and the specified companies.
        
        :param start_date: The start date for downloading data (format: "YYYY-MM-DD").
        :param end_date: The end date for downloading data (format: "YYYY-MM-DD"). If None, fetches up to the current date.
        """
        # Initialize an empty DataFrame for the final combined data
        self.data = pd.DataFrame()
        
        # List of all tickers, including S&P 500
        tickers = ['^GSPC'] + self.companies
        
        for ticker in tickers:
            try:
                # Download the 'Close' prices for the ticker
                df = yf.download(ticker, start=start_date, end=end_date)[['Close']]
                df.rename(columns={'Close': f'{ticker} CLOSE'}, inplace=True)
                
                # Merge with the main DataFrame on the date index
                if self.data.empty:
                    self.data = df
                else:
                    self.data = self.data.merge(df, left_index=True, right_index=True, how='outer')
                    
            except Exception as e:
                print(f"Failed to download data for {ticker}: {e}")
        
        # Reset the index to have 'Date' as a column
        self.data.reset_index(inplace=True)
    
    def get_data(self):
        """
        Returns the combined historical 'Close' prices data.
        
        :return: A pandas DataFrame with the historical 'Close' prices.
        """
        return self.data
    
    def save_data(self, filepath="Data/historical_data.xlsx"):
        """
        Saves the downloaded 'Close' prices to a single sheet in an Excel file.
        
        :param filepath: The full path where the Excel file should be saved.
        """
        # Create the directory if it doesn't exist
        directory = os.path.dirname(filepath)
        os.makedirs(directory, exist_ok=True)
        
        # Save data to Excel
        self.data.to_excel(filepath, index=False, sheet_name="Historical Data")
        print(f"Data saved to {filepath}")