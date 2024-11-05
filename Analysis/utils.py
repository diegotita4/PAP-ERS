
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
import random
import optuna
import joblib
from joblib import Parallel, delayed
from random import sample
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize


# ------------------------------

# SET THE STYLE FOR SEABORN VISUALIZATIONS
sns.set(style="whitegrid")

# --------------------------------------------------

# CLASS FOR PERFORMING EXPLORATOY DATA ANALYSIS (EDA) ON ECONOMIC INDICATORS AND S&P 500 DATA
class EDA_comparison:
    """
    Class for performing exploratory data analysis (EDA) on economic indicators 
    and S&P 500 data. Allows visualization and summarization of merged data.

    Attributes:
        sp500_data (DataFrame): S&P 500 data.
        economic_indicators_data (DataFrame): Economic indicators data.
        date_column (str): Name of the column containing dates.
        columns_to_analyze (list): List of columns to analyze. If None, all are analyzed.
        indicators_colors (dict): Mapping of colors for economic indicators.
        merged_data (DataFrame): Merged data of S&P 500 and economic indicators.
    """

    def __init__(self, sp500_data, economic_indicators_data, date_column='Date', columns_to_analyze=None):
        """
        Initializes the EDA_comparison class.

        Args:
            sp500_data (DataFrame): S&P 500 data.
            economic_indicators_data (DataFrame): Economic indicators data.
            date_column (str): Name of the column containing dates. Defaults to 'Date'.
            columns_to_analyze (list): List of columns to analyze. If None, all are analyzed.
        """

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
        """
        Displays a summary of the merged dataset, including:
        - The first and last few rows of the DataFrame.
        - General information about the DataFrame.
        - Descriptive statistics.
        - Count of missing values.
        - Identification of outliers using the Z-score method.
        """

        print("First few rows of the data:")
        print(self.merged_data.head(12))

        print("Last few rows of the data:")
        print(self.merged_data.tail(5))

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
        """
        Calculates and displays basic statistics of the numerical columns in the merged dataset, 
        including mean, median, mode, variance, and standard deviation.
        """

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

    def plot_performances_indv(self, sp500_column='^GSPC_AC'):
        """
        Plots the individual performance of economic indicators along with the S&P 500.

        Args:
            sp500_column (str): Name of the S&P 500 column in the merged data. Defaults to '^GSPC_AC'.
        """

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

    def plot_performances_grpl(self, sp500_column='^GSPC_AC'):
        """
        Plots the grouped performance of economic indicators and the S&P 500.

        Args:
            sp500_column (str): Name of the S&P 500 column in the merged data. Defaults to '^GSPC_AC'.
        """

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

    def plot_histograms(self, sp500_column='^GSPC_AC'):
        """
        Plots histograms for economic indicators and the S&P 500.

        Args:
            sp500_column (str): Name of the S&P 500 column in the merged data. Defaults to '^GSPC_AC'.
        """

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

    def plot_boxplots(self, sp500_column='^GSPC_AC'):
        """
        Plots boxplots for economic indicators and the S&P 500.

        Args:
            sp500_column (str): Name of the S&P 500 column in the merged data. Defaults to '^GSPC_AC'.
        """

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
        """
        Plots the relationships between economic indicators using scatter plots and histograms.
        """

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
        """
        Plots the correlation matrix between economic indicators.
        """

        plt.figure(figsize=(12, 8))
        sns.heatmap(self.economic_indicators_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Economic Indicators')
        plt.show()

    # ------------------------------

    def perform_EDA_comparison(self):
        """
        Performs a complete exploratory data analysis by executing all analysis and visualization methods.
        """

        self.data_summary()
        self.data_statistics()
        self.plot_performances_indv()
        self.plot_performances_grpl()
        self.plot_histograms()
        self.plot_boxplots()
        self.plot_relations()
        self.plot_correlation_matrix()

# --------------------------------------------------

# CLASS FOR DOWNLOADING HISTORICAL ADJUSTED CLOSE PRICES AND BETA VALUES FOR A LIST OF TICKERS
class HistoricalDataDownloader:
    """
    Class for downloading historical adjusted close prices and beta values for a list of tickers.

    Attributes:
        tickers (list): List of ticker symbols to download data for.
        start_date (str): Start date for downloading data.
        end_date (str): End date for downloading data.
        adj_close (DataFrame): DataFrame containing the adjusted close prices.
        beta (DataFrame): DataFrame containing the beta values.
    """

    def __init__(self, tickers):
        """
        Initializes the HistoricalDataDownloader class.

        Args:
            tickers (list): List of ticker symbols to download data for.
        """

        self.tickers = tickers
        self.start_date = '2000-01-01'
        self.end_date = '2024-06-01'
        self.adj_close = pd.DataFrame()
        self.beta = pd.DataFrame()
        self.rename_tickers()

    # ------------------------------

    def rename_tickers(self):
        """
        Renames specific tickers in the list to ensure compatibility with Yahoo Finance.
        """

        ticker_rename_map = {
            'BRK.B': 'BRK-B',
            'BF.B': 'BF-B'
        }
        self.tickers = [ticker_rename_map.get(ticker, ticker) for ticker in self.tickers]

    # ------------------------------

    def download_adj_close(self):
        """
        Downloads adjusted close prices for each ticker in the list.
        Merges the data into a single DataFrame.
        """

        for ticker in self.tickers:
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, interval='1mo')[['Adj Close']].copy()
                df.rename(columns={'Adj Close': f'{ticker}_AC'}, inplace=True)

                if df.empty:
                    print(f"No data found for {ticker} in the given range.")
                    continue

                if self.adj_close.empty:
                    self.adj_close = df

                else:
                    self.adj_close = self.adj_close.merge(df, left_index=True, right_index=True, how='outer')

            except Exception as e:
                print(f"Failed to download adj close values for {ticker}: {e}")

        self.adj_close.reset_index(inplace=True)

    # ------------------------------

    def download_beta(self):
        """
        Downloads the beta values for each ticker.
        If the beta value is not available, calculates it based on the adjusted close prices.
        """

        def calculate_beta(ticker):

            try:
                if f'{ticker}_AC' not in self.adj_close.columns:
                    print(f"{ticker} not in adj_close columns")
                    return None

                ticker_data = self.adj_close[['Date', f'{ticker}_AC']].dropna()
                ticker_data.set_index('Date', inplace=True)

                benchmark_ticker = '^GSPC'
                benchmark_data = yf.download(benchmark_ticker, start=self.start_date, end=self.end_date, interval='1mo')[['Adj Close']].dropna()
                benchmark_data.index.name = 'Date'
                benchmark_data.reset_index(inplace=True)

                if ticker_data.empty or benchmark_data.empty:
                    print(f"No data for {ticker} or benchmark")
                    return None

                merged_data = ticker_data.join(benchmark_data.set_index('Date'), how='inner', on='Date')
                ticker_returns = merged_data[f'{ticker}_AC'].pct_change().dropna()
                benchmark_returns = merged_data['Adj Close'].pct_change().dropna()

                if len(ticker_returns) < 2 or len(benchmark_returns) < 2:
                    print(f"Not enough data for {ticker}")
                    return None

                cov_value = ticker_returns.cov(benchmark_returns)
                benchmark_var = benchmark_returns.var()
                beta = cov_value / benchmark_var

                return beta if isinstance(beta, (int, float)) else beta.values[0]

            except Exception as e:
                print(f"Failed to calculate beta for {ticker}: {e}")
                return None

        # ----------

        def fetch_beta(ticker):
            """
            Fetches the beta value for a ticker from Yahoo Finance.
            If not available, calculates it using historical data.

            Args:
                ticker (str): The ticker symbol to fetch the beta for.

            Returns:
                dict: A dictionary containing the ticker and its beta value.
            """

            try:
                if f'{ticker}_AC' not in self.adj_close.columns:
                    return {"Ticker": ticker, "Beta": None}

                beta_info = yf.Ticker(ticker).info
                beta = beta_info.get("beta", None)

                if beta is None:
                    calculated_beta = calculate_beta(ticker)

                    if calculated_beta is not None:
                        return {"Ticker": ticker, "Beta": calculated_beta}

                    else:
                        return {"Ticker": ticker, "Beta": None}

                else:
                    return {"Ticker": ticker, "Beta": beta}

            except Exception as e:
                print(f"Failed to fetch beta for {ticker}: {e}")
                return {"Ticker": ticker, "Beta": None}

        # ----------

        with ThreadPoolExecutor(max_workers=10) as executor:
            betas = list(executor.map(fetch_beta, self.tickers))

        self.beta = pd.DataFrame(betas)
        
        valid_tickers = self.beta[self.beta['Beta'].notnull()]['Ticker'].tolist()

        for ticker in self.adj_close.columns:
            if ticker.endswith('_AC') and ticker[:-3] not in valid_tickers:
                self.adj_close.drop(columns=ticker, inplace=True)
        
        self.beta = self.beta[self.beta['Beta'].notnull()]

    # ------------------------------

    def classify_beta(self):
        """
        Classifies beta values into categories based on predefined bins.

        The bins are:
            - 0: Beta ≤ 0.7
            - 1: Beta > 0.7
        """

        bins = [-np.inf, 0.7, np.inf]
        labels = ['0', '1']

        self.beta['Nature'] = pd.cut(self.beta['Beta'], bins=bins, labels=labels)

    # ------------------------------

    def save_data(self, filepath):
        """
        Saves the adjusted close prices and beta values to an Excel file.

        Args:
            filepath (str): The path where the Excel file will be saved.
        """

        try:
            directory = os.path.dirname(filepath)
            os.makedirs(directory, exist_ok=True)

            with pd.ExcelWriter(filepath) as writer:
                self.adj_close.to_excel(writer, index=False, sheet_name="adj_close")

                if len(self.tickers) > 1:
                    self.beta.to_excel(writer, index=False, sheet_name="beta")

            print(f"Data saved to {filepath}")

        except Exception as e:
            print(f"Failed to save data: {e}")

# --------------------------------------------------

# CLASS FOR BUILDING, TRAINING, AND OPTIMIZING MACHINE LEARNING MODELS BASED ON ECONOMIC INDICATORS
# AND S&P 500 DATA, INCLUDING LOGISTIC REGRESSION, XGBOOST, AND MLP (MULTI-AYER PERCEPTRON).
class Models:
    """
    A class used to represent different machine learning models for predicting stock market performance
    based on economic indicators and the S&P 500.
    
    Attributes:
        sp500_data (DataFrame): The historical data for the S&P 500.
        economic_indicators_data (DataFrame): The economic indicators data.
        umbral (float): The threshold used to classify the target variable (Y).
        model_data (DataFrame): The processed and merged data for training the models.
    """

    def __init__(self, sp500_data, economic_indicators_data, umbral):
        """
        Initializes the Models class with the provided S&P 500 data, economic indicators, and threshold.

        Args:
            sp500_data (DataFrame): The historical S&P 500 data.
            economic_indicators_data (DataFrame): Data containing economic indicators.
            umbral (float): The threshold for classifying the target variable Y.

        Returns:
            None
        """

        self.economic_indicators_data = economic_indicators_data
        self.sp500_data = sp500_data
        self.umbral = umbral
        self.model_data = self.model_data_function(self.economic_indicators_data, self.sp500_data, self.umbral)

    # ------------------------------

    def model_data_function(self, economic_indicators_data, sp500_data, umbral):
        """
        Merges economic indicators with S&P 500 data, calculates percentage changes for S&P 500,
        and applies a threshold to create the target variable Y.

        Args:
            economic_indicators_data (DataFrame): Data containing economic indicators.
            sp500_data (DataFrame): The historical data for the S&P 500.
            umbral (float): The threshold used to classify Y.

        Returns:
            DataFrame: The merged and processed model data with the target variable Y.
        """

        model_data = pd.merge(economic_indicators_data, sp500_data, on='Date', how='inner')
        model_data['^GSPC_R'] = model_data['^GSPC_AC'].pct_change().dropna()
        model_data['Y'] = model_data['^GSPC_R'].apply(lambda x: 1 if x > umbral else (-1 if x < -umbral else 0))
        model_data = model_data.dropna()
        model_data = model_data.shift(-1).dropna()

        return model_data

    # ------------------------------

    def logistic_regression(self):
        """
        Trains a logistic regression model using the processed model data and evaluates its performance.
        The model handles multi-class classification using OneVsRestClassifier.

        Returns:
            LogisticRegression: The trained logistic regression model.
        """

        X = self.model_data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC_R']]
        y = self.model_data['Y']

        y = label_binarize(y, classes=np.unique(y))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

        self.lr_model = OneVsRestClassifier(LogisticRegression(max_iter=1000))

        self.lr_model.fit(X_train, y_train)

        y_pred = self.lr_model.predict(X_test)
        y_pred_proba = self.lr_model.predict_proba(X_test)  

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') 

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = y_test.shape[1]

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} ROC curve (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - Logistic Regression')
        plt.legend(loc="lower right")
        plt.show()

        self.save_best_model(self.lr_model, "logistic_regression", accuracy)

        # ----------
        print('\n-------------------')
        print('LOGISTIC REGRESSION')
        print('-------------------')
        print(f'\nParameters:\n')
        print(self.lr_model.get_params())
        print('\n----------\n')
        print(f'Accuracy: {accuracy:.4f}')
        print('\n----------\n')
        print('Classification Report:\n')
        print(report)

        return self.lr_model


    # ------------------------------

    def optimized_logistic_regression(self):
        """
        Optimizes the Logistic Regression model by searching for the best hyperparameters using RandomizedSearchCV.
        Evaluates the model using cross-validation and saves the best-performing model.

        Returns:
            LogisticRegression: The optimized logistic regression model.
        """

        X = self.model_data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC_R']]
        y = self.model_data['Y']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

        param_dist = [
            {'penalty': ['l2'], 
             'C': [0.01, 0.1, 0.3, 1, 3], 
             'solver': ['lbfgs'], 
             'max_iter': [1000, 3000], 
             'class_weight': [None, 'balanced']},
            {'penalty': ['l1'], 
             'C': [0.01, 0.1, 0.3, 1, 3], 
             'solver': ['liblinear'], 
             'max_iter': [1000, 3000], 
             'class_weight': [None, 'balanced']}
        ]
        random_search = RandomizedSearchCV(LogisticRegression(), param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1, random_state=0)
        random_search.fit(X_train, y_train)

        self.lr_model = random_search.best_estimator_

        cv_scores = cross_val_score(self.lr_model, X, y, cv=5)

        y_pred = self.lr_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        self.save_best_model(self.lr_model, "optimized_logistic_regression", accuracy)

        # ----------

        print('\n-----------------------------')
        print('OPTIMIZED LOGISTIC REGRESSION')
        print('-----------------------------')
        print(f'\nCross-Validation Scores: {cv_scores}')
        print(f'\nMean CV Accuracy: {cv_scores.mean():.4f}')
        print('\n----------\n')
        print(f'Parameters:\n')
        print(self.lr_model.get_params())
        print('\n----------\n')
        print(f'Accuracy: {accuracy:.4f}')
        print('\n----------\n')
        print('Classification Report:\n')
        print(report)

        return self.lr_model

    # ------------------------------

    def XGBoost(self):
        """
        Trains an XGBoost classifier using the processed model data. Performs RandomizedSearchCV to find the best hyperparameters.
        Evaluates the model on test data and saves the best-performing model.

        Returns:
            XGBClassifier: The trained XGBoost model.
        """

        X = self.model_data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC_R']]
        y = self.model_data['Y'] + 1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

        model = xgb.XGBClassifier()

        param_dist = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.5, 0.75, 1],
            'colsample_bytree': [0.5, 0.75, 1],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0],
            'min_child_weight': [1, 3, 5],
            'max_delta_step': [0, 1, 2]
        }

        random_search = RandomizedSearchCV(model, param_distributions=param_dist, cv=5, scoring='accuracy', verbose=1, n_jobs=-1, n_iter=100, random_state=0)
        random_search.fit(X_train, y_train)

        self.xgb_model = random_search.best_estimator_

        y_pred = self.xgb_model.predict(X_test)

        accuracy = accuracy_score(y_test - 1, y_pred - 1)
        report = classification_report(y_test - 1, y_pred - 1)

        self.save_best_model(self.xgb_model, "xgboost", accuracy)

        # ----------

        print('\n-------')
        print('XGBOOST')
        print('-------')
        print(f'\nParameters:\n')
        print(self.xgb_model.get_params())
        print('\n----------\n')
        print(f'Accuracy: {accuracy:.4f}')
        print('\n----------\n')
        print('Classification Report:\n')
        print(report)

        return self.xgb_model

    # ------------------------------

    def optimized_XGBoost(self):
        """
        Optimizes the XGBoost classifier using Optuna for hyperparameter tuning. 
        Evaluates the model on test data and saves the best-performing model.

        Returns:
            XGBClassifier: The optimized XGBoost model.
        """

        X = self.model_data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC_R']]
        y = self.model_data['Y'] + 1  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

        # ----------

        def objective(trial):
            """
            Defines the objective function for Optuna to optimize hyperparameters for the XGBoost model.
            
            Args:
                trial (Trial): Optuna trial object.

            Returns:
                float: The accuracy of the model on validation data.
            """

            param = {
                'objective': 'multi:softprob',  
                'num_class': 3,  
                'n_estimators': trial.suggest_int('n_estimators', 2, 4),  
                'learning_rate': trial.suggest_float('learning_rate', 0.010, 0.020), 
                'max_depth': trial.suggest_int('max_depth', 1, 2), 
                'subsample': trial.suggest_float('subsample', 0.2, 0.5), 
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.4),  
                'gamma': trial.suggest_float('gamma', 0, 0.3),  
                'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 2),  
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2),  
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),  
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 1)
            }
            model = xgb.XGBClassifier(**param)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            return accuracy

        # ----------

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        best_params = study.best_params

        self.xgb_model = xgb.XGBClassifier(**best_params)

        self.xgb_model.fit(X_train, y_train)

        y_pred = self.xgb_model.predict(X_test)
        y_pred_proba = self.xgb_model.predict_proba(X_test)  

        y_test_adjusted = y_test - 1
        y_pred_adjusted = y_pred - 1

        accuracy = accuracy_score(y_test_adjusted, y_pred_adjusted)
        report = classification_report(y_test_adjusted, y_pred_adjusted)

        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

        y_test_binarized = label_binarize(y_test_adjusted, classes=[-1, 0, 1])  # Asegurarse de usar las clases correctas

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(3):  
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} ROC curve (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - Optimized XGBoost')
        plt.legend(loc="lower right")
        plt.show()

        self.save_best_model(self.xgb_model, "optimized_xgboost", accuracy)

        # ----------

        print('\n-----------------')
        print('OPTIMIZED XGBOOST')
        print('-----------------')
        print(f'\nParameters:\n')
        print(self.xgb_model.get_params())
        print('\n----------\n')
        print(f'Accuracy: {accuracy:.4f}')
        print('\n----------\n')
        print('Classification Report:\n')
        print(report)

        return self.xgb_model

    # ------------------------------

    def MLP(self, activation='relu'):
        """
        Trains a Multi-Layer Perceptron (MLP) model using the processed model data.
        Evaluates the model on test data, plots the AUC-ROC curve for each class,
        and saves the trained model in a .pkl file inside the 'Models' folder with the activation function name in the filename.

        Args:
            activation (str): The activation function to use in the hidden layers (default is 'relu').

        Returns:
            tuple: A tuple containing the accuracy of the model and the classification report.
        """

        X = self.model_data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC_R']]
        y = self.model_data['Y']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=0)

        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=1500,
            activation=activation,
            solver='adam',
            alpha=0.005,
            learning_rate_init=0.0005,
            random_state=0
        )

        self.mlp_model.fit(X_train, y_train)

        y_pred = self.mlp_model.predict(X_test)
        y_pred_proba = self.mlp_model.predict_proba(X_test)  

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=1)

        y_test_binarized = label_binarize(y_test, classes=[-1, 0, 1])

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(3): 
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} ROC curve (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) - MLP (Activation: {activation})')
        plt.legend(loc="lower right")
        plt.show()

        # Create the Models directory if it doesn't exist
        models_dir = 'Models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Save the model to the 'Models' folder with the activation name
        model_filename = os.path.join(models_dir, f"mlp_{activation}.pkl")
        try:
            joblib.dump(self.mlp_model, model_filename)
            print(f"Model saved as '{model_filename}'")
        except Exception as e:
            print(f"Error saving the model: {e}")

        return accuracy, report
    # ------------------------------

    def optimized_MLP(self, n_trials=50):
        """
        Optimizes the Multi-Layer Perceptron (MLP) model using Optuna for hyperparameter tuning.
        Evaluates the model on test data and saves the best-performing model.

        Args:
            n_trials (int): The number of trials for Optuna to perform for hyperparameter tuning (default is 50).

        Returns:
            MLPClassifier: The optimized MLP model.
        """

        def objective(trial):
            """
            Defines the objective function for Optuna to optimize the MLP model's hyperparameters.

            Args:
                trial (Trial): Optuna trial object.

            Returns:
                float: The accuracy of the model on validation data.
            """
            hidden_layer_sizes = tuple([trial.suggest_int(f'n_units_l{i}', 16, 128) for i in range(3)])
            alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
            learning_rate_init = trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-1)

            X = self.model_data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC_R']]
            y = self.model_data['Y']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=0)

            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=1500,
                activation='relu',
                solver='adam',
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                random_state=0
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)

            return accuracy

        # ----------

        print("Running Optuna optimization for MLP...")
        self.study_mlp = optuna.create_study(direction='maximize')
        self.study_mlp.optimize(objective, n_trials=n_trials)

        best_params = self.study_mlp.best_params
        hidden_layer_sizes = tuple([best_params[f'n_units_l{i}'] for i in range(3)])

        X = self.model_data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC_R']]
        y = self.model_data['Y']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=0)

        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=1500,
            activation='relu',
            solver='adam',
            alpha=best_params['alpha'],
            learning_rate_init=best_params['learning_rate_init'],
            random_state=0
        )

        print("Training optimized MLP model...")
        self.mlp_model.fit(X_train, y_train)

        y_pred = self.mlp_model.predict(X_test)
        y_pred_proba = self.mlp_model.predict_proba(X_test)

        y_test_binarized = label_binarize(y_test, classes=[-1, 0, 1])

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} ROC curve (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) - Optimized MLP (Activation: relu)')
        plt.legend(loc="lower right")
        plt.show()

        # Save the model
        try:
            joblib.dump(self.mlp_model, 'best_mlp_model.pkl')
            print("Best MLP model saved as 'best_mlp_model.pkl'")
        except Exception as e:
            print(f"Error saving the model: {e}")

        return self.mlp_model


    # ------------------------------

    def save_data(self, filepath):
        """
        Saves the processed model data to an Excel file.

        Args:
            filepath (str): The file path where the data should be saved.

        Returns:
            None
        """
        
        try:
            directory = os.path.dirname(filepath)
            os.makedirs(directory, exist_ok=True)

            with pd.ExcelWriter(filepath) as writer:
                self.model_data.to_excel(writer, index=False, sheet_name="model_data")

            print(f"Data saved to {filepath}")

        except Exception as e:
            print(f"Failed to save data: {e}")

    # ------------------------------

    def save_best_model(self, model, model_name, accuracy):
        """
        Saves the model with the highest accuracy to a file. If a model with a higher accuracy already exists, 
        it skips saving.

        Args:
            model: The machine learning model to save.
            model_name (str): The name of the model.
            accuracy (float): The accuracy of the model.

        Returns:
            None
        """

        models_dir = 'Models'
        os.makedirs(models_dir, exist_ok=True)

        model_filename = f"Models/{model_name}.pkl"
        accuracy_filename = f"Models/{model_name}_best_accuracy.txt"

        if os.path.exists(accuracy_filename):
            with open(accuracy_filename, 'r') as f:
                saved_accuracy = float(f.read())

        else:
            saved_accuracy = 0

        if accuracy > saved_accuracy:

            joblib.dump(model, model_filename)

            with open(accuracy_filename, 'w') as f:
                f.write(str(accuracy))

            print(f"New model '{model_name}' saved with an accuracy of {accuracy * 100:.2f}%")

        else:
            print(f"The current model has better accuracy ({saved_accuracy * 100:.2f}%). The new model is not saved.")

# --------------------------------------------------

# # Let's proceed by creating a new class `DynamicBacktesting` that encapsulates the classes and methods identified in the notebook.

class DynamicBacktesting:
    """Class to encapsulate data handling, portfolio management, and backtesting functionality."""
    
    class DataHandler:
        """Handles loading asset and model data from Excel files."""
        def __init__(self, assets_path, model_path):
            self.assets_path = assets_path
            self.model_path = model_path
            self.adj_close = None
            self.beta_data = None
            self.model_data = None

        def load_assets_data(self):
            """Loads asset price data and beta classification data."""
            self.adj_close = pd.read_excel(self.assets_path, sheet_name='adj_close', index_col="Date")
            self.beta_data = pd.read_excel(self.assets_path, sheet_name='beta')

        def load_model_data(self):
            """Loads economic indicators and expected trend data."""
            self.model_data = pd.read_excel(self.model_path, index_col="Date")

    class Portfolio:
        """Manages portfolio initialization, optimization, and rebalancing."""
        def __init__(self, beta_data, adj_close, initial_portfolio_value=1_000_000, commission=0.01):
            self.beta_data = beta_data
            self.adj_close = adj_close
            self.initial_portfolio_value = initial_portfolio_value
            self.commission = commission
            self.num_shares = pd.Series(dtype=float)
            self.current_cash = initial_portfolio_value
            self.commission_history = []

        def build_portfolio(self, y_trend, current_prices):
            """Selects assets and initializes portfolio based on economic trend Y."""
            anticiclical = self.beta_data[self.beta_data['Nature'] == 0]['Ticker'].tolist()
            prociclical = self.beta_data[self.beta_data['Nature'] == 1]['Ticker'].tolist()

            if y_trend == -1:
                selected_anticiclical = sample(anticiclical, 15)
                selected_prociclical = sample(prociclical, 5)
            elif y_trend == 0:
                selected_anticiclical = sample(anticiclical, 10)
                selected_prociclical = sample(prociclical, 10)
            elif y_trend == 1:
                selected_anticiclical = sample(anticiclical, 5)
                selected_prociclical = sample(prociclical, 15)

            selected_assets = selected_anticiclical + selected_prociclical
            selected_assets = [asset + "_AC" for asset in selected_assets]
            available_assets = [asset for asset in selected_assets if asset in self.adj_close.columns]
            current_prices = current_prices[available_assets]

            investment_value_per_asset = self.initial_portfolio_value / len(available_assets)
            self.num_shares = (investment_value_per_asset / current_prices).apply(np.floor)
            self.current_cash = self.initial_portfolio_value - (self.num_shares * current_prices).sum()
            
            return available_assets

        def optimize_sharpe(self, returns):
            """Optimizes portfolio weights to maximize the Sharpe Ratio."""
            def sharpe_ratio(weights):
                portfolio_return = np.sum(returns.mean() * weights)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
                return -portfolio_return / portfolio_volatility

            num_assets = len(returns.columns)
            constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
            bounds = tuple((0, 1) for _ in range(num_assets))
            initial_guess = num_assets * [1. / num_assets,]
            optimized_result = minimize(sharpe_ratio, initial_guess, bounds=bounds, constraints=constraints)
            
            return optimized_result.x  # Optimal weights

        def rebalance_portfolio(self, y_trend, current_prices, rebalance_index):
            """Rebalances the portfolio based on new trend Y, considering commission."""
            pre_commission_value = (self.num_shares * current_prices).sum() + self.current_cash
            commission_cost = (self.num_shares * current_prices * self.commission).sum()
            self.commission_history.append(commission_cost)
            total_value_after_commission = pre_commission_value - commission_cost

            selected_assets = self.build_portfolio(y_trend, current_prices)
            current_prices = current_prices[selected_assets]

            returns = self.adj_close[selected_assets].iloc[rebalance_index-12:rebalance_index].pct_change().dropna()
            optimal_weights = self.optimize_sharpe(returns)

            investment_value_per_asset = total_value_after_commission * optimal_weights
            self.num_shares = (investment_value_per_asset / current_prices).apply(np.floor)
            self.current_cash = total_value_after_commission - (self.num_shares * current_prices).sum()

            return (self.num_shares * current_prices).sum() + self.current_cash, commission_cost

        def calculate_monthly_value(self, current_prices):
            """Calculates total portfolio value for the given month."""
            return (self.num_shares * current_prices).sum() + self.current_cash

    class Backtest:
        """Manages dynamic backtesting with periodic rebalancing."""
        def __init__(self, model_data, portfolio):
            self.model_data = model_data
            self.portfolio = portfolio
            self.portfolio_value = portfolio.initial_portfolio_value
            self.portfolio_value_history = []
            self.dates_history = []

        def run_backtest(self):
            dates = self.model_data.index
            initial_prices = self.portfolio.adj_close.loc[dates[0]]
            y_trend = self.model_data.loc[dates[0], 'Y']
            self.portfolio.build_portfolio(y_trend, initial_prices)

            for i, current_date in enumerate(dates):
                y_trend = self.model_data.loc[current_date, 'Y']
                current_prices = self.portfolio.adj_close.loc[current_date]

                if i % 12 == 0 and i > 0:
                    self.portfolio.rebalance_portfolio(y_trend, current_prices, i)

                portfolio_value = self.portfolio.calculate_monthly_value(current_prices)
                self.portfolio_value_history.append(portfolio_value)
                self.dates_history.append(current_date)

        def save_results_to_excel(self, filename):
            """Saves backtest results to an Excel file."""
            results_df = pd.DataFrame({
                'Date': self.dates_history,
                'Portfolio_Value': self.portfolio_value_history
            })
            results_df.to_excel(filename, index=False)
            print(f"Results saved to {filename}")
            
# The new DynamicBacktesting class now incorporates all functionality from the notebook.


