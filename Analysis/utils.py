
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from statsmodels.tsa.api import VAR

# ------------------------------

# 
sns.set(style="whitegrid")

# --------------------------------------------------

# 
class EDA_comparison:

    def __init__(self, sp500_data, economic_indicators_data, date_column='Date', columns_to_analyze=None):
        self.sp500_data = sp500_data
        self.economic_indicators_data = economic_indicators_data
        self.date_column = date_column
        self.columns_to_analyze = columns_to_analyze
        self.indicators_colors = {'CLI': '#800080','BCI': '#B8860B','CCI': '#8B0000','GDP': '#2F4F4F'}

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

    def plot_performances_indv(self, sp500_column='^GSPC CLOSE'):

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()

        for ax1, indicator in zip(axs, self.economic_indicators_data.columns.tolist()):
            ax1.set_xlabel('Year', fontsize=12)
            ax1.set_ylabel(indicator, fontsize=12, color='black')
            ax1.plot(self.merged_data.index, self.merged_data[indicator], color=self.indicators_colors.get(indicator, 'blue'), linewidth=1.5, label=indicator)
            ax1.tick_params(axis='y', labelcolor='black', labelsize=10)

            ax2 = ax1.twinx()
            ax2.set_ylabel('S&P 500', fontsize=12, color='black')
            ax2.plot(self.merged_data.index, self.merged_data[sp500_column], color='black', linewidth=1.5, label='S&P 500')
            ax2.tick_params(axis='y', labelcolor='black', labelsize=10)

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        fig.tight_layout()
        plt.suptitle('Economic Indicators and S&P 500', fontsize=16, y=1.02)
        plt.show()

    # ------------------------------

    def plot_performances_grpl(self, sp500_column='^GSPC CLOSE'):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

        for indicator in self.economic_indicators_data.columns.tolist():
            ax1.plot(self.merged_data.index, self.merged_data[indicator], label=indicator, color=self.indicators_colors.get(indicator, 'gary'),)

        ax1.set_title('Economic Indicators')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend(loc='best')
        ax1.set_ylim([85, 110])
        ax1.axhline(y=100, color='blue', linestyle='--', linewidth=1)

        ax2.plot(self.merged_data.index, self.merged_data[sp500_column], label='S&P 500', color='black')
        ax2.set_title('S&P 500')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Value')

        mean_value = self.merged_data[sp500_column].mean()
        ax2.axhline(y=mean_value, color='blue', linestyle='--', linewidth=1)
        ax2.legend(loc='best')

        plt.tight_layout()
        plt.show()

    # ------------------------------

    def plot_histograms(self, sp500_column='^GSPC CLOSE'):

        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)

        for indicator in self.economic_indicators_data.columns.tolist():
            ax1.hist(self.merged_data[indicator].dropna(), bins=30, alpha=0.5, label=indicator, color=self.indicators_colors.get(indicator, 'gray'))

        ax1.axvline(x=100, color='blue', linestyle='--', linewidth=3)
        ax1.set_xlim([85, 110])
        ax1.set_title('Economic Indicators')
        ax1.legend(loc='best')

        ax2 = plt.subplot2grid((2, 2), (1, 0))
        ax2.hist(self.merged_data[sp500_column].dropna(), bins=30, color='black', alpha=0.7)
        ax2.axvline(x=self.merged_data[sp500_column].mean(), color='blue', linestyle='--', linewidth=3)
        ax2.set_title('S&P 500')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')

        ax3 = plt.subplot2grid((2, 2), (1, 1))
        ax3.hist(self.merged_data[sp500_column].pct_change().dropna()*100, bins=30, color='green', alpha=0.7)
        ax3.axvline(x=(self.merged_data[sp500_column].pct_change().dropna()*100).mean(), color='blue', linestyle='--', linewidth=3)
        ax3.set_title('S&P 500 Pct Change')
        ax3.set_xlabel('Pct Change (%)')
        ax3.set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    # ------------------------------

    def plot_boxplots(self, sp500_column='^GSPC CLOSE'):

        data = [self.merged_data[indicator].dropna() for indicator in self.economic_indicators_data.columns.tolist()]
        colors = [self.indicators_colors.get(indicator, 'gray') for indicator in self.economic_indicators_data.columns.tolist()]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        box1 = ax1.boxplot(data, vert=False, patch_artist=True, labels=self.economic_indicators_data.columns.tolist())
        for patch, color in zip(box1['boxes'], colors):
            patch.set_facecolor(color)

        ax1.set_title('Economic Indicators')
        ax1.set_xlabel('Value')

        box2 = ax2.boxplot([self.merged_data[sp500_column].dropna()], vert=False, patch_artist=True, labels=['S&P 500'], boxprops=dict(facecolor='black'))
        
        ax2.set_title('S&P 500')
        ax2.set_xlabel('Value')

        plt.tight_layout()
        plt.show()

    # ------------------------------

    def plot_relations(self):

        sns.set(style="white")
        num_vars = len(self.economic_indicators_data.columns.tolist())
        fig, axes = plt.subplots(num_vars, num_vars, figsize=(12, 12))

        for i, var1 in enumerate(self.economic_indicators_data.columns.tolist()):
            for j, var2 in enumerate(self.economic_indicators_data.columns.tolist()):
                ax = axes[i, j]

                if i == j:
                    sns.histplot(self.economic_indicators_data[var1], ax=ax, color=self.indicators_colors.get(var1, 'gray'), kde=True)

                else:
                    sns.scatterplot(x=self.economic_indicators_data[var2], y=self.economic_indicators_data[var1], ax=ax, color=self.indicators_colors.get(var1, 'gray'))

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

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.suptitle('Economic Indicators', y=1.03, fontsize=16)
        plt.show()

    # ------------------------------

    def plot_correlation_matrix(self):

        plt.figure(figsize=(12, 8))
        sns.heatmap(self.economic_indicators_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Economic Indicators')
        plt.show()

    # ------------------------------

    def perform_EDA_comparison(self):

        self.data_summary()
        self.data_statistics()
        self.plot_performances_indv()
        self.plot_performances_grpl()
        self.plot_histograms()
        self.plot_boxplots()
        self.plot_relations()
        self.plot_correlation_matrix()

# --------------------------------------------------

# 
class HistoricalDataDownloader:

    def __init__(self, tickers, start_date):

        self.tickers = tickers
        self.start_date = start_date
        self.data = pd.DataFrame()
        self.beta_values = pd.DataFrame()
        self.cyclicality_labels = pd.DataFrame()

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

    def calculate_beta(self, market_ticker='^GSPC'):
        """
        Calculate beta values for each ticker based on the specified market index.
        """
        try:
            # Download market data (S&P 500)
            market_data = yf.download(market_ticker, start=self.start_date, interval='1mo')['Adj Close'].copy()
            market_data = market_data.pct_change().dropna()
            market_data.rename('Market', inplace=True)

            # Calculate log returns for each stock
            returns_data = pd.DataFrame()

            for ticker in self.tickers:
                stock_returns = self.data[f'{ticker} CLOSE'].pct_change().dropna()
                returns_data[ticker] = stock_returns

            # Align data with market data
            returns_data = returns_data.merge(market_data, left_index=True, right_index=True, how='inner')

            # Calculate rolling beta values
            for ticker in self.tickers:
                rolling_cov = returns_data[ticker].rolling(window=12).cov(returns_data['Market'])
                rolling_var_market = returns_data['Market'].rolling(window=12).var()
                self.beta_values[ticker] = rolling_cov / rolling_var_market

            self.beta_values.dropna(inplace=True)

        except Exception as e:
            print(f"Failed to calculate beta values: {e}")

    # ------------------------------

    def classify_cyclicality(self):
        """
        Classify each ticker as procyclical or anticyclical at each point in time based on beta values.
        A stock is classified as:
        - 'Anticyclical' if its beta at a specific point in time is between 0 and 0.7 (inclusive).
        - 'Procyclical' if its beta at a specific point in time is greater than 0.7.
        This classification is done for each time period individually.
        """

        try:
            # Define the classification function based on the beta for each time period
            def label_beta_for_each_time(beta_value):
                """
                Classify an asset based on its beta at a specific point in time:
                - 'Anticyclical' if the beta is between 0 and 0.7.
                - 'Procyclical' if the beta is greater than 0.7.
                """
                if 0 <= beta_value <= 0.7:
                    return 'Anticyclical'
                elif beta_value > 0.7:
                    return 'Procyclical'
                else:
                    return 'Unclassified'  # Catch negative or undefined beta values.

            # Apply the classification function to each beta value for each ticker at each time
            self.cyclicality_labels = self.beta_values.applymap(label_beta_for_each_time)

        except Exception as e:
            print(f"Failed to classify cyclicality: {e}")

    # ------------------------------

    def save_data(self, filepath):
        """
        Save the downloaded data, beta values, and cyclicality labels to an Excel file.
        """
        try:
            directory = os.path.dirname(filepath)
            os.makedirs(directory, exist_ok=True)

            with pd.ExcelWriter(filepath) as writer:
                self.data.to_excel(writer, index=False, sheet_name="Historical Data")
                self.beta_values.to_excel(writer, sheet_name="Beta Values")
                self.cyclicality_labels.to_excel(writer, sheet_name="Cyclicality Labels")

            print(f"Data saved to {filepath}")

        except Exception as e:
            print(f"Failed to save data: {e}")

# --------------------------------------------------

# Class to save different models to predict S&P 500 data
class Models:
    def __init__(self, indicators_df, sp500_df):
        # Shift economic indicators to reflect predictive nature
        indicators_df = self._shift_indicators(indicators_df)
        
        # Combine economic indicators with historical S&P 500 data
        self.data = self._merge_data(indicators_df, sp500_df)
        
    def _shift_indicators(self, indicators_df):
        # Shift indicators by 3 months to predict future S&P 500 movements
        indicators_df['CLI'] = indicators_df['CLI'].shift(3)
        indicators_df['BCI'] = indicators_df['BCI'].shift(3)
        indicators_df['CCI'] = indicators_df['CCI'].shift(3)
        return indicators_df
        
    def _merge_data(self, indicators_df, sp500_df):
        # Convert 'Date' columns to datetime format
        indicators_df['Date'] = pd.to_datetime(indicators_df['Date'])
        sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])
        
        # Merge both datasets based on 'Date'
        merged_data = pd.merge(indicators_df, sp500_df, on='Date', how='inner')
        
        # Create percentage change column for the S&P 500
        merged_data['SP500_Change'] = merged_data['^GSPC CLOSE'].pct_change()
        
        # Define the target column (1: overweight, 0: neutral, -1: underweight)
        merged_data['Target'] = merged_data['SP500_Change'].apply(lambda x: 1 if x > 0.02 
                                                                  else (-1 if x < -0.02 else 0))
        
        # Drop rows with any NaN values in the predictors or target columns
        merged_data.dropna(inplace=True)
        
        return merged_data
    
    def train_logistic_regression(self):
        # Define predictor variables (CLI, BCI, GDP, CCI, and S&P 500 historical data) and the target variable (Target)
        X = self.data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC CLOSE']]
        y = self.data['Target']
        
        # Split the dataset into training and testing sets (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the OneVsRest logistic regression model with adjusted hyperparameters
        self.lr_model = OneVsRestClassifier(LogisticRegression(
            solver='saga',   # You can experiment with 'saga' or other solvers
            C=0.3,                # Regularization strength (smaller values = stronger regularization)
            penalty='l2',          # Regularization type (l1, l2, or elasticnet)
            class_weight='balanced' # Handle class imbalance
        ))
        
        self.lr_model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = self.lr_model.predict(X_test)
        
        # Calculate accuracy and classification report
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Save test data and predictions
        self.X_test = X_test
        self.y_test = y_test
        self.lr_y_pred = y_pred
        
        return accuracy, report
    
    def train_var_model(self, maxlags=15):
        # Define predictor variables (CLI, BCI, GDP, CCI, and S&P 500 historical data)
        data = self.data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC CLOSE']]
        
        # Split the dataset into training and testing sets (80% training, 20% testing)
        train_size = int(len(data) * 0.8)
        train_data, test_data = data[:train_size], data[train_size:]
        
        # Train the VAR model
        model = VAR(train_data)
        
        # Fit the model using the specified number of lags
        self.var_model = model.fit(maxlags=maxlags, ic='aic')  # Select the best lag based on the Akaike criterion (AIC)
        
        # Save training and test data
        self.train_data = train_data
        self.test_data = test_data
    
    def predict_var(self, steps=10):
        # Use the trained VAR model to make predictions
        predictions = self.var_model.forecast(self.train_data.values[-self.var_model.k_ar:], steps=steps)
        
        # Convert predictions to a DataFrame with the same columns as the original data
        predictions_df = pd.DataFrame(predictions, columns=self.train_data.columns)
        
        return predictions_df
    
    def evaluate_var(self):
        # Generate predictions for the same length as the test set
        steps = len(self.test_data)
        var_predictions = self.var_model.forecast(self.train_data.values[-self.var_model.k_ar:], steps=steps)
        
        # Convert predictions to a DataFrame
        var_predictions_df = pd.DataFrame(var_predictions, columns=self.train_data.columns)
        
        # Evaluate the accuracy of the VAR model on the S&P 500 predictions
        actual_sp500 = self.test_data['^GSPC CLOSE'].values
        predicted_sp500 = var_predictions_df['^GSPC CLOSE'].values
        
        # Calculate accuracy based on direction of change
        actual_direction = np.sign(np.diff(actual_sp500))
        predicted_direction = np.sign(np.diff(predicted_sp500))
        accuracy = np.mean(actual_direction == predicted_direction)
        
        return accuracy

    def download_sp500_data(self):
        # Download historical data for S&P 500 from Yahoo Finance
        sp500_data = yf.download('^GSPC', start='2010-01-01', end='2024-07-01')
        sp500_data.reset_index(inplace=True)
        return sp500_data
    
    def download_economic_data(self):
        # Download economic indicators from appropriate API (to be implemented)
        pass




# --------------------------------------------------

# NOTAS
