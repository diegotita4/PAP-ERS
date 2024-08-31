
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
    
    # ------------------------------
    
    def plot_indicators_with_sp500_dual_axis(self, indicators_columns, sp500_column='^GSPC CLOSE'):

        color_map = {
            'CLI': '#800080',  # Dark Magenta
            'BCI': '#B8860B',  # Dark Goldenrod
            'CCI': '#8B0000',  # Dark Red
            'GDP': '#2F4F4F'   # Dark Slate Gray
        }

        num_indicators = len(indicators_columns)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))  
        axs = axs.flatten()  

        for ax1, indicator_column in zip(axs, indicators_columns):
            indicator_color = color_map.get(indicator_column, 'blue')  

            # Economic Indicators
            ax1.set_xlabel('Year', fontsize=12)
            ax1.set_ylabel(indicator_column, fontsize=12, color='black')  
            ax1.plot(self.merged_data.index, self.merged_data[indicator_column], color=indicator_color, linewidth=1.5, label=indicator_column)
            ax1.tick_params(axis='y', labelcolor='black', labelsize=10)
            
            # S&P 500
            ax2 = ax1.twinx()
            ax2.set_ylabel('S&P 500', fontsize=12, color='black')  
            ax2.plot(self.merged_data.index, self.merged_data[sp500_column], color='black', linewidth=1.5, label='S&P 500')
            ax2.tick_params(axis='y', labelcolor='black', labelsize=10)

          
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

          
            ax1.grid(False)
            ax2.grid(False)
            ax1.set_facecolor('white')
            ax2.set_facecolor('white')

        fig.tight_layout()
        plt.suptitle('Economic Indicators and S&P 500', fontsize=16, y=1.02)
        plt.show()

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

    def plot_performance(self, sp500_column='^GSPC CLOSE', indicators_columns=None):

        if indicators_columns is None:
            indicators_columns = self.economic_indicators_data.columns.tolist()

        # Map color
        color_map = {
            'CLI': '#800080',  # Dark Magenta
            'BCI': '#B8860B',  # Dark Goldenrod
            'CCI': '#8B0000',  # Dark Red
            'GDP': '#2F4F4F'   # Dark Slate Gray
        }

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

        # Economic Indicators
        for indicator in indicators_columns:
            color = color_map.get(indicator, 'gray')  
            ax1.plot(self.merged_data.index, self.merged_data[indicator], label=indicator, color=color)

        ax1.set_title('Economic Indicators Performance')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend(loc='best')
        ax1.set_ylim([85, 110])
        ax1.axhline(y=100, color='blue', linestyle='--', linewidth=1)

        # S&P 500
        ax2.plot(self.merged_data.index, self.merged_data[sp500_column], label='S&P 500', color='black')
        ax2.set_title('S&P 500 Performance')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Value')

        mean_value = self.merged_data[sp500_column].mean()
        ax2.axhline(y=mean_value, color='blue', linestyle='--', linewidth=1)
        ax2.legend(loc='best')

        plt.tight_layout()
        plt.show()

    # ------------------------------

    def plot_histograms(self, sp500_column='^GSPC CLOSE', indicators_columns=None):

        if indicators_columns is None:
            indicators_columns = self.economic_indicators_data.columns.tolist()

        # Map color
        color_map = {
            'CLI': '#800080',  # Dark Magenta
            'BCI': '#B8860B',  # Dark Goldenrod
            'CCI': '#8B0000',  # Dark Red
            'GDP': '#2F4F4F'   # Dark Slate Gray
        }

        
        fig = plt.figure(figsize=(14, 10))

        # Economic Indicators
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)  
        for indicator in indicators_columns:
            color = color_map.get(indicator, 'gray')  
            ax1.hist(self.merged_data[indicator].dropna(), bins=30, alpha=0.5, label=indicator, color=color)

        ax1.axvline(x=100, color='blue', linestyle='--', linewidth=3)
        ax1.set_xlim([85, 110])
        ax1.set_title('Economic Indicators Histograms')
        ax1.legend(loc='best')

        # S&P 500
        ax2 = plt.subplot2grid((2, 2), (1, 0)) 
        ax2.hist(self.merged_data[sp500_column].dropna(), bins=30, color='black', alpha=0.7)
        mean_value = self.merged_data[sp500_column].mean()
        ax2.axvline(x=mean_value, color='blue', linestyle='--', linewidth=3)
        ax2.set_title('S&P 500 Histogram')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')

        # S&P 500 Pct change
        ax3 = plt.subplot2grid((2, 2), (1, 1))  
        sp500_pct_change = self.merged_data[sp500_column].pct_change().dropna() * 100  
        ax3.hist(sp500_pct_change, bins=30, color='green', alpha=0.7)
        mean_pct_change = sp500_pct_change.mean()
        ax3.axvline(x=mean_pct_change, color='blue', linestyle='--', linewidth=3)
        ax3.set_title('S&P 500 Pct Change Histogram')
        ax3.set_xlabel('Pct Change (%)')
        ax3.set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    # ------------------------------

    def plot_boxplots(self, sp500_column='^GSPC CLOSE', indicators_columns=None):

        if indicators_columns is None:
            indicators_columns = self.economic_indicators_data.columns.tolist()

        color_map = {
            'CLI': '#800080',  # Dark Magenta
            'BCI': '#B8860B',  # Dark Goldenrod
            'CCI': '#8B0000',  # Dark Red
            'GDP': '#2F4F4F'   # Dark Slate Gray
        }


        data = [self.merged_data[indicator].dropna() for indicator in indicators_columns]
        colors = [color_map.get(indicator, 'gray') for indicator in indicators_columns]


        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        box1 = ax1.boxplot(data, vert=False, patch_artist=True, labels=indicators_columns)
        for patch, color in zip(box1['boxes'], colors):
            patch.set_facecolor(color)

        ax1.set_title('Economic Indicators Boxplots')
        ax1.set_xlabel('Value')

        box2 = ax2.boxplot([self.merged_data[sp500_column].dropna()], vert=False, patch_artist=True, labels=['S&P 500'], boxprops=dict(facecolor='black'))
        
        ax2.set_title('S&P 500 Boxplot')
        ax2.set_xlabel('Value')

        plt.tight_layout()
        plt.show()


    # ------------------------------

    def plot_correlation_matrix(self, indicators_columns=None):

        if indicators_columns is None:
            indicators_columns = self.economic_indicators_data.columns.tolist()

        columns_to_corr = indicators_columns 

        plt.figure(figsize=(12, 8))
        sns.heatmap(self.merged_data[columns_to_corr].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
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

        color_map = {
            'CLI': '#800080',  # Dark Magenta
            'BCI': '#B8860B',  # Dark Goldenrod
            'CCI': '#8B0000',  # Dark Red
            'GDP': '#2F4F4F'   # Dark Slate Gray
        }

        sns.set(style="white")
        num_vars = len(indicators_columns)
        fig, axes = plt.subplots(num_vars, num_vars, figsize=(12, 12))

        for i, var1 in enumerate(indicators_columns):
            for j, var2 in enumerate(indicators_columns):
                ax = axes[i, j]
                if i == j:
                    sns.histplot(data_to_plot[var1], ax=ax, color=color_map.get(var1, 'gray'), kde=True)
                else:
                    sns.scatterplot(x=data_to_plot[var2], y=data_to_plot[var1], ax=ax, color=color_map.get(var1, 'gray'))

                if j == 0:
                    ax.set_ylabel(var1)
                else:
                    ax.set_ylabel('')
                    ax.set_yticks([])

                if i == num_vars - 1:
                    ax.set_xlabel(var2)
                else:
                    ax.set_xlabel('')
                    ax.set_xticks([])

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Ajustar la figura para que deje espacio para el título
        plt.suptitle('Economic Indicators Relation', y=1.03, fontsize=16)  # Ajustar la posición y el tamaño del título
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
