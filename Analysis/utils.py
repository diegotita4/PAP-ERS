
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
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats

# ------------------------------

# 
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# --------------------------------------------------

# 
class EDA_comparison:

    def __init__(self, sp500_data, economic_indicators_data, date_column='Date', columns_to_analyze=None):
        self.sp500_data = sp500_data
        self.economic_indicators_data = economic_indicators_data
        self.date_column = date_column
        self.columns_to_analyze = columns_to_analyze
        self.sp500_data['Return'] = self.sp500_data['^GSPC CLOSE'].pct_change()

        if self.date_column in self.sp500_data.columns:
            self.sp500_data[self.date_column] = pd.to_datetime(self.sp500_data[self.date_column])
            self.sp500_data.set_index(self.date_column, inplace=True)

        if self.date_column in self.economic_indicators_data.columns:
            self.economic_indicators_data[self.date_column] = pd.to_datetime(self.economic_indicators_data[self.date_column])
            self.economic_indicators_data.set_index(self.date_column, inplace=True)

        if self.columns_to_analyze:
            self.sp500_data = self.sp500_data[self.columns_to_analyze]
            self.economic_indicators_data = self.economic_indicators_data[self.columns_to_analyze]

        self.merged_data = pd.merge(self.sp500_data, self.economic_indicators_data, left_index=True, right_index=True, how='inner')

        self.merged_data['Return (Norm)'] = 100 * (self.merged_data['Return'] + 1)
        self.merged_data['Return (SMA 3)'] = self.merged_data['Return (Norm)'].rolling(window=3).mean()
        self.merged_data['Return (SMA 6)'] = self.merged_data['Return (Norm)'].rolling(window=6).mean()

    # ------------------------------

    def data_summary(self):
        print("First few rows of the data:")
        print(self.merged_data.head(10))

        print("\nInformation:")
        print(self.merged_data.info())

        print("\nSummary statistics:")
        print(self.merged_data.describe())

        print("\nMissing values:")
        print(self.merged_data.isnull().sum())

        print("\nOutliers")
        print(np.where(np.abs(stats.zscore(self.merged_data)) > 3))

    # ------------------------------

    def data_statistics(self):
        numeric_data = self.merged_data.select_dtypes(include=[np.number])

        print("\nMean of each column:")
        print(numeric_data.mean())

        print("\nMedian of each column:")
        print(numeric_data.median())

        print("\nMode of each column:")
        print(numeric_data.mode().iloc[0])

        print("\nVariance of each column:")
        print(numeric_data.var())

        print("\nStandard deviation of each column:")
        print(numeric_data.std())

    # ------------------------------

    def plot_performance(self, sp500_column='Return (SMA 3)', indicators_columns=None):

        if indicators_columns is None:
            indicators_columns = self.economic_indicators_data.columns.tolist()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

        for indicator in indicators_columns:
            ax1.plot(self.merged_data.index, self.merged_data[indicator], label=indicator)

        ax1.set_title('Economic Indicators Performance')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend(loc='best')
        ax1.set_ylim([85, 110])
        ax1.axhline(y=100, color='red', linestyle='--', linewidth=2)

        ax2.plot(self.merged_data.index, self.merged_data[sp500_column], label=sp500_column, color='black')
        ax2.set_title('S&P 500 Performance')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Value')
        ax2.legend(loc='best')
        ax2.set_ylim([85, 110])
        ax2.axhline(y=100, color='red', linestyle='--', linewidth=2)

        plt.tight_layout()
        plt.show()

    # ------------------------------

    def plot_histograms(self, sp500_column='Return (SMA 3)', indicators_columns=None):

        if indicators_columns is None:
            indicators_columns = self.economic_indicators_data.columns.tolist()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

        for indicator in indicators_columns:
            ax1.hist(self.merged_data[indicator].dropna(), bins=30, alpha=0.5, label=indicator)

        ax1.axvline(x=100, color='red', linestyle='--', linewidth=2)
        ax1.set_xlim([85, 110])
        ax1.set_title('Economic Indicators Histograms')

        ax2.hist(self.merged_data[sp500_column].dropna(), bins=30, color='purple', alpha=0.5, label=sp500_column)

        ax2.axvline(x=100, color='red', linestyle='--', linewidth=2)
        ax2.set_xlim([85, 110])
        ax2.set_title('S&P 500 Histogram')

        plt.tight_layout()
        plt.show()

    # ------------------------------

    def plot_boxplots(self, sp500_column='Return (SMA 3)', indicators_columns=None):

        if indicators_columns is None:
            indicators_columns = self.economic_indicators_data.columns.tolist()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        ax1.boxplot([self.merged_data[indicator].dropna() for indicator in indicators_columns], vert=False, labels=indicators_columns)
        ax1.set_title('Economic Indicators Boxplots')
        ax1.set_xlabel('Value')

        ax2.boxplot(self.merged_data[sp500_column].dropna(), vert=False, labels=[sp500_column], patch_artist=True, boxprops=dict(facecolor='purple'))
        ax2.set_title('S&P 500 Boxplot')
        ax2.set_xlabel('Value')

        plt.tight_layout()
        plt.show()

    # ------------------------------

    def plot_correlation_matrix(self, indicators_columns=None):

        if indicators_columns is None:
            indicators_columns = self.economic_indicators_data.columns.tolist()

        plt.figure(figsize=(12, 8))
        sns.heatmap(self.merged_data[indicators_columns].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Economic Indicators Correlation Matrix')
        plt.show()
    
     # ------------------------------

    def plot_pairplot(self, indicators_columns=None):

        if indicators_columns is None:
            indicators_columns = self.economic_indicators_data.columns.tolist()
        
        data_to_plot = self.economic_indicators_data[indicators_columns].dropna()

        if data_to_plot.empty:
            print("No data available for pairplot.")
            return

        sns.pairplot(data_to_plot)
        plt.suptitle('Relaciones entre Indicadores Económicos', y=1.02)
        plt.show()

    # ------------------------------

    def perform_EDA_comparison(self):

        self.data_summary()
        self.data_statistics()
        self.plot_performance()
        self.plot_histograms()
        self.plot_boxplots()
        self.plot_correlation_matrix()
        self.plot_pairplot()

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
                df = yf.download(ticker, start=self.start_date, end=end_date, interval='1mo')[['Close']].copy()
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
