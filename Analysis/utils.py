
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Estrategias de Rotación Sectorial (ERS)                                                    -- #
# -- script: utils.py - Python script with the main functionality                                        -- #
# -- authors: diegotita4 - Antonio-IF - JoAlfonso - Oscar148                                             -- #
# -- license: GNU GENERAL PUBLIC LICENSE - Version 3, 29 June 2007                                       -- #
# -- repository: https://github.com/diegotita4/PAP-ERS                                                   -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# --------------------------------------------------

# LIBRARIES
import os
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
from fredapi import Fred
from datetime import datetime

# --------------------------------------------------

# 
class EDA:

    def __init__(self, file_path, date_column=None, columns_to_analyze=None, sheet_name='Sheet1'):

        self.file_path = file_path
        self.sheet_name = sheet_name
        self.date_column = date_column
        self.columns_to_analyze = columns_to_analyze
        self.data_df = None
        self.summary_stats = None
        self.missing_values = None
        self.correlation_matrix = None

    # ------------------------------

    def data_load(self):

        excel_data = pd.ExcelFile(self.file_path)
        self.data_df = pd.read_excel(excel_data, sheet_name=self.sheet_name)

        if self.date_column in self.data_df.columns:
            self.data_df[self.date_column] = pd.to_datetime(self.data_df[self.date_column])
            self.data_df.set_index(self.date_column, inplace=True)

        if self.columns_to_analyze:
            self.data_df = self.data_df[self.columns_to_analyze]

    # ------------------------------

    def data_summary(self):

        print("First few rows of the data:")
        print(self.data_df.head())

        print("\nSummary statistics:")
        print(self.data_df.describe())

        print("\nMissing values:")
        print(self.data_df.isnull().sum())

    # ------------------------------

    def data_statistics(self):

        print("\nMean of each column:")
        print(self.data_df.mean())

        print("\nMedian of each column:")
        print(self.data_df.median())

        print("\nMode of each column:")
        print(self.data_df.mode().iloc[0])

        print("\nVariance of each column:")
        print(self.data_df.var())

    # ------------------------------

    def plot_performance(self):

        plt.figure(figsize=(14, 8))
        plt.plot(self.data_df.index, self.data_df[self.columns_to_analyze], label=self.columns_to_analyze)
        plt.title('Performance')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend(loc='best')
        plt.show()

    # ------------------------------

    def plot_correlation_matrix(self):

        plt.figure(figsize=(8, 6))
        sns.heatmap(self.data_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()

    # ------------------------------

    def plot_histogram(self):

        self.data_df.hist(figsize=(14, 10), bins=20, color='teal', edgecolor='black')
        plt.suptitle('Histogram(s)')
        plt.show()

    # ------------------------------

    def plot_boxplot(self):

        plt.figure(figsize=(14, 10))
        sns.boxplot(data=self.data_df, palette='Set2')
        plt.title('Boxplot(s)')
        plt.show()

    # ------------------------------

    def perform_EDA(self):

        self.data_load()
        self.data_summary()
        self.data_statistics()
        self.plot_performance()
        self.plot_histogram()
        self.plot_boxplot()

        if len(self.columns_to_analyze) > 1:
            try:
                self.plot_correlation_matrix()

            except:
                pass

# --------------------------------------------------

# 
class EDA_comparison:

    def __init__(self, sp500_data, economic_indicators_data, date_column='Date'):

        self.sp500_data = sp500_data
        self.economic_indicators_data = economic_indicators_data
        self.date_column = date_column
        self.sp500_data[self.date_column] = pd.to_datetime(self.sp500_data[self.date_column])
        self.economic_indicators_data[self.date_column] = pd.to_datetime(self.economic_indicators_data[self.date_column])
        self.merged_data = pd.merge(self.sp500_data, self.economic_indicators_data, on=self.date_column, how='inner')

    # ------------------------------

    def plot_performance(self, sp500_column='^GSPC CLOSE', indicators_columns=None):

        if indicators_columns is None:
            indicators_columns = self.economic_indicators_data.columns.drop(self.date_column).tolist()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        for indicator in indicators_columns:
            ax1.plot(self.merged_data[self.date_column], self.merged_data[indicator], label=indicator)

        ax1.set_title('Economic Indicators Performance')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend(loc='best')

        ax2.plot(self.merged_data[self.date_column], self.merged_data[sp500_column], label=sp500_column, color='black')
        ax2.set_title('S&P 500 Performance')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Value')
        ax2.legend(loc='best')

        plt.tight_layout()
        plt.show()

    # ------------------------------

    def plot_histograms(self, sp500_column='^GSPC CLOSE', indicators_columns=None):

        if indicators_columns is None:
            indicators_columns = self.economic_indicators_data.columns.drop(self.date_column).tolist()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        for indicator in indicators_columns:
            ax1.hist(self.merged_data[indicator].dropna(), bins=30, alpha=0.5, label=indicator)

        ax1.set_title('Economic Indicators Histograms')

        ax2.hist(self.merged_data[sp500_column].dropna(), bins=30, color='purple', alpha=0.5, label=sp500_column)
        ax2.set_title('S&P 500 Histogram')

        plt.tight_layout()
        plt.show()

    # ------------------------------

    def plot_boxplots(self, sp500_column='^GSPC CLOSE', indicators_columns=None):

        if indicators_columns is None:
            indicators_columns = self.economic_indicators_data.columns.drop(self.date_column).tolist()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        ax1.boxplot([self.merged_data[indicator].dropna() for indicator in indicators_columns], vert=False, labels=indicators_columns)
        ax1.set_title('Economic Indicators Boxplots')
        ax1.set_xlabel('Value')

        ax2.boxplot(self.merged_data[sp500_column].dropna(), vert=False, labels=[sp500_column], patch_artist=True, boxprops=dict(facecolor='purple'))
        ax2.set_title('S&P 500 Boxplot')
        ax2.set_xlabel('Value')

        plt.tight_layout()
        plt.show()
    
    def perform_EDA_comparison(self):

        self.plot_performance()
        self.plot_histograms()
        self.plot_boxplots()

# --------------------------------------------------

# 
class HistoricalDataDownloader:

    def __init__(self, tickers, start_date):

        self.tickers = tickers
        self.start_date = start_date
        self.data = pd.DataFrame()

    # ------------------------------

    def download_data(self, end_date=None):

        self.data = pd.DataFrame()
        
        for ticker in self.tickers:
            try:
                df = yf.download(ticker, start=self.start_date, end=end_date)[['Close']].copy()
                df.rename(columns={'Close': f'{ticker} CLOSE'}, inplace=True)

                if self.data.empty:
                    self.data = df

                else:
                    self.data = self.data.merge(df, left_index=True, right_index=True, how='outer')

            except Exception as e:
                print(f"Failed to download data for {ticker}: {e}")

        self.data.reset_index(inplace=True)

    # ------------------------------

    def save_data(self, filepath):

        directory = os.path.dirname(filepath)
        os.makedirs(directory, exist_ok=True)
        self.data.to_excel(filepath, index=False, sheet_name="data")
        print(f"Data saved to {filepath}")










# --------------------------------------------------

# EXTRA
class FredIndicatorFetcher:

    def __init__(self, api_key):

        self.api_key = api_key
        self.fred = Fred(api_key=self.api_key)
        self.indicators = {
            "TLEI": "M16005USM358SNBR",
            "CPI": "CPIAUCSL",
            "GDP": "USALORSGPNOSTSAM",
            "CCI": "CSCICP03USM665S",
            "CEI": "USPHCI"
        }

    # ------------------------------

    def fetch_indicator(self, indicator_key, start_date='1960-01-31', end_date=None):

        if indicator_key not in self.indicators:
            raise ValueError(f"Indicator '{indicator_key}' not found in the predefined indicators list.")
        
        end_date = end_date if end_date else datetime.today().strftime('%Y-%m-%d')
        series_id = self.indicators[indicator_key]
        data = self.fred.get_series(series_id, start_date, end_date)
        return data

    # ------------------------------

    def save_to_excel(self, data_dict, file_name):

        save_directory = 'PAP-ERS/Data/'
        os.makedirs(save_directory, exist_ok=True)
        file_path = os.path.join(save_directory, file_name)
        combined_df = pd.DataFrame()

        for indicator, data in data_dict.items():
            df = pd.DataFrame(data, columns=[indicator])

            if combined_df.empty:
                combined_df = df

            else:
                combined_df = combined_df.join(df, how='outer')
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            combined_df.to_excel(writer, sheet_name='Combined_Indicators')
