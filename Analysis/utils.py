
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

# 
class PortfolioManager:
    def __init__(self, excel_file_path, selected_model, model_data):
        """
        Initializes the PortfolioManager with the provided Excel file path, selected model, and model data.

        Args:
            excel_file_path (str): Path to the Excel file containing assets and historical prices.
            selected_model (str): The selected machine learning model for predictions.
            model_data (pd.DataFrame): Data to be used for model predictions.
        """
        self.excel_file_path = excel_file_path
        self.assets_data = self.load_assets_data()
        self.selected_model = selected_model
        self.model_data = model_data

    # ------------------------------

    def load_assets_data(self):
        """
        Loads asset data (Ticker, Beta, Nature) from the specified Excel file's 'beta' sheet.

        Returns:
            pd.DataFrame: DataFrame containing asset information (Ticker, Beta, Nature).
        """
        try:
            df = pd.read_excel(self.excel_file_path, sheet_name='beta')
            print("Structure of assets_data:", df.head())  # Debugging: Print first rows for inspection
            df.columns = ['Ticker', 'Beta', 'Nature']  # Ensure these columns exist in the Excel file
            return df
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return None

    # ------------------------------

    def classify_assets(self):
        """
        Classifies assets into pro-cyclical and anti-cyclical categories based on their Beta values.

        Returns:
            tuple: Two DataFrames, one for pro-cyclical and one for anti-cyclical assets.
        """
        self.assets_data['Ticker_AC'] = self.assets_data['Ticker'] + '_AC'
        pro_cyclics = self.assets_data[self.assets_data['Beta'] > 0.7]
        anti_cyclics = self.assets_data[(self.assets_data['Beta'] >= 0) & (self.assets_data['Beta'] <= 0.7)]
        return pro_cyclics, anti_cyclics

    # ------------------------------

    def predict_y(self):
        """
        Loads the mlp_relu model and predicts values based on the provided model data.

        Returns:
            np.array: Predicted values (y) from the model.
        """
        model_filename = 'Models/mlp_relu.pkl'  # Path to your model file
        try:
            model = joblib.load(model_filename)
            X = self.model_data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC_R']].values  # Use .values to remove feature names
            y_predicted = model.predict(X)
            return y_predicted
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    # ------------------------------

    def create_portfolio(self, y_pred):
        """
        Creates a portfolio based on the model's prediction. The allocation of assets depends on 
        the predicted y value (-1, 0, or 1).

        Args:
            y_pred (int): The predicted value of y (-1, 0, 1).

        Returns:
            dict: Portfolio allocation with Ticker and respective weights.
        """
        pro_cyclics, anti_cyclics = self.classify_assets()

        if y_pred == 1:
            pro_cyclic_weight = 0.75
            anti_cyclic_weight = 0.25
        elif y_pred == 0:
            pro_cyclic_weight = 0.50
            anti_cyclic_weight = 0.50
        elif y_pred == -1:
            pro_cyclic_weight = 0.25
            anti_cyclic_weight = 0.75
        else:
            raise ValueError("Invalid predicted value for y. Must be -1, 0, or 1.")

        portfolio = {}
        if not pro_cyclics.empty:
            pro_cyclic_assets = pro_cyclics['Ticker'].tolist()
            pro_cyclic_allocation = pro_cyclic_weight / len(pro_cyclic_assets)
            for asset in pro_cyclic_assets:
                portfolio[f"{asset}_AC"] = min(0.20, pro_cyclic_allocation)

        if not anti_cyclics.empty:
            anti_cyclic_assets = anti_cyclics['Ticker'].tolist()
            anti_cyclic_allocation = anti_cyclic_weight / len(anti_cyclic_assets)
            for asset in anti_cyclic_assets:
                portfolio[f"{asset}_AC"] = min(0.20, anti_cyclic_allocation)

        return portfolio

    # ------------------------------

    def calculate_portfolio_returns(self, portfolio, date_range, target_return=0.02):
        """
        Calculates portfolio returns for a given date range based on historical prices, with optimized weights
        to maximize the Omega Ratio.

        Args:
            portfolio (dict): Initial portfolio with asset tickers.
            date_range (list): List of dates over which to calculate the portfolio returns.
            target_return (float): Target return threshold for Omega Ratio optimization (default is 0.02).

        Returns:
            pd.Series: Series of portfolio returns for the given date range.
        """
        # Load historical prices and filter by the date range
        historical_prices = self.load_historical_prices(self.excel_file_path)
        historical_prices = historical_prices.loc[date_range]  # Filter the price data by date range
        
        # Extract the tickers and ensure they exist in the historical data
        portfolio_tickers = [f"{ticker}_AC" for ticker in portfolio.keys()]
        valid_tickers = [ticker for ticker in portfolio_tickers if ticker in historical_prices.columns]
        historical_prices = historical_prices[valid_tickers]  # Keep only valid tickers
        
        # Calculate daily returns for the valid tickers
        daily_returns = historical_prices.pct_change().dropna()

        # Define a function to calculate the Omega Ratio, which will serve as the objective function
        def omega_ratio_objective(weights, returns, target_return):
            """
            Objective function to minimize (negative Omega Ratio).

            Args:
                weights (np.array): Portfolio weights to optimize.
                returns (pd.DataFrame): Historical returns of the assets in the portfolio.
                target_return (float): Target return threshold for Omega Ratio.

            Returns:
                float: Negative of the Omega Ratio (since we're minimizing).
            """
            portfolio_returns = np.dot(returns, weights)  # Calculate portfolio returns
            excess_returns = portfolio_returns - target_return

            # Calculate gains and losses
            gains = excess_returns[excess_returns > 0].sum()
            losses = -excess_returns[excess_returns < 0].sum()

            # Avoid division by zero: if losses are zero, Omega is infinite (return a large negative value)
            if losses == 0:
                return -np.inf if gains > 0 else np.nan

            omega_ratio = gains / losses
            return -omega_ratio  # Return negative Omega Ratio for minimization

        # Initial guess for weights (equal weights for all assets)
        num_assets = len(valid_tickers)
        initial_weights = np.ones(num_assets) / num_assets

        # Set constraints: Weights must sum to 1
        constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

        # Set bounds for weights: All weights must be between 0 and 1
        bounds = [(0, 1) for _ in range(num_assets)]

        # Optimize the portfolio weights to maximize the Omega Ratio
        result = minimize(omega_ratio_objective, initial_weights, args=(daily_returns, target_return),
                        method='SLSQP', bounds=bounds, constraints=constraints)

        # Get the optimized weights
        if result.success:
            optimized_weights = result.x
        else:
            raise ValueError("Omega Ratio optimization failed.")

        # Calculate the optimized portfolio returns
        portfolio_returns = np.dot(daily_returns, optimized_weights)
        
        # Return the portfolio returns as a pandas Series, indexed by date
        return pd.Series(portfolio_returns, index=daily_returns.index)

    # ------------------------------

    def load_historical_prices(self, file_path):
        """
        Loads historical prices from an Excel file and ensures that the Date column is set as the index.

        Args:
            file_path (str): Path to the Excel file containing historical prices.

        Returns:
            pd.DataFrame: DataFrame with historical prices indexed by Date.
        """
        try:
            historical_prices = pd.read_excel(file_path, sheet_name='adj_close')
            historical_prices.set_index('Date', inplace=True)  # Ensure Date is the index
            return historical_prices.dropna()  # Optional: Drop rows with NaN
        except Exception as e:
            print(f"Error loading historical prices: {e}")
            return None
        
    # ------------------------------    

    def calculate_omega_ratio(self, portfolio_returns, target_return=0.20):
        """
        Calculates the Omega Ratio for the created portfolio.

        Args:
            portfolio_returns (pd.Series): The returns of the portfolio.
            target_return (float): Target return for calculating excess returns.

        Returns:
            float: Omega Ratio of the portfolio, or np.nan if invalid.
        """
        excess_returns = portfolio_returns - target_return

        gains = excess_returns[excess_returns > 0].sum()
        losses = -excess_returns[excess_returns < 0].sum()

        if losses == 0:
            if gains > 0:
                return np.inf  # Infinite Omega Ratio when there are no losses
            else:
                return np.nan  # Undefined Omega Ratio if both gains and losses are zero or negative

        omega_ratio = gains / losses
        return omega_ratio

# --------------------------------------------------

class DynamicBacktestingWithOmega:
    def __init__(self, initial_capital, portfolio_manager, com=0.001):
        """
        Initializes the dynamic backtesting class.
        
        Args:
            initial_capital (float): The initial capital for backtesting.
            portfolio_manager (PortfolioManager): An instance of the PortfolioManager class to manage the portfolio.
            com (float): Transaction commission (e.g., 0.001 for 0.1%).
        """
        self.initial_capital = initial_capital
        self.portfolio_manager = portfolio_manager
        self.com = com
        self.portfolio_value = []  # Portfolio values over time
        self.positions = {}  # Current positions in the portfolio
        self.cash = initial_capital  # Available cash for transactions

    def adjust_portfolio(self, trend_signal):
        """
        Adjusts the portfolio weights based on the trend signal (y), with random selection of assets from 
        pro-cyclical, anti-cyclical, and neutral categories.

        Args:
            trend_signal (int): The trend signal predicted by the model (-1, 0, 1).
        """
        # Get pro-cyclical and anti-cyclical assets
        pro_cyclics, anti_cyclics = self.portfolio_manager.classify_assets()

        # Define weight allocation based on the trend signal
        if trend_signal == 1:  # Bullish signal (favors pro-cyclical)
            pro_cyclic_weight = 0.75
            anti_cyclic_weight = 0.25
        elif trend_signal == 0:  # Neutral signal (balanced between pro- and anti-cyclical)
            pro_cyclic_weight = 0.50
            anti_cyclic_weight = 0.50
        elif trend_signal == -1:  # Bearish signal (favors anti-cyclical)
            pro_cyclic_weight = 0.25
            anti_cyclic_weight = 0.75
        else:
            raise ValueError("Invalid trend signal: must be -1, 0, or 1.")

        # Randomly select a subset of assets from pro-cyclical and anti-cyclical groups
        num_pro_cyclic_assets = len(pro_cyclics)
        num_anti_cyclic_assets = len(anti_cyclics)

        # Select a random sample of pro-cyclical assets
        if num_pro_cyclic_assets > 0:
            selected_pro_cyclic_assets = pro_cyclics.sample(frac=0.5)  # Select 50% of pro-cyclical assets randomly
            pro_cyclic_allocation = pro_cyclic_weight / len(selected_pro_cyclic_assets)
        else:
            selected_pro_cyclic_assets = pd.DataFrame()  # Empty DataFrame if no pro-cyclical assets exist
            pro_cyclic_allocation = 0

        # Select a random sample of anti-cyclical assets
        if num_anti_cyclic_assets > 0:
            selected_anti_cyclic_assets = anti_cyclics.sample(frac=0.5)  # Select 50% of anti-cyclical assets randomly
            anti_cyclic_allocation = anti_cyclic_weight / len(selected_anti_cyclic_assets)
        else:
            selected_anti_cyclic_assets = pd.DataFrame()  # Empty DataFrame if no anti-cyclical assets exist
            anti_cyclic_allocation = 0

        # Build the portfolio with selected assets and their corresponding weights
        portfolio = {}

        # Add selected pro-cyclical assets to the portfolio
        for asset in selected_pro_cyclic_assets['Ticker']:
            portfolio[f"{asset}_AC"] = min(0.20, pro_cyclic_allocation)

        # Add selected anti-cyclical assets to the portfolio
        for asset in selected_anti_cyclic_assets['Ticker']:
            portfolio[f"{asset}_AC"] = min(0.20, anti_cyclic_allocation)

        # Assign the portfolio to positions
        self.positions = portfolio
        print(f"Adjusted Portfolio: {self.positions}")  # Debugging: View the adjusted portfolio

    def calculate_portfolio_value(self, price_data, date):
        """
        Calculates the total portfolio value for a given date.

        Args:
            price_data (pd.DataFrame): DataFrame with historical prices.
            date (str or pd.Timestamp): The date for which to calculate the portfolio value.
        """
        total_value = 0
        valid_positions = {}

        # Filter only tickers that have valid prices (no NaN) on the current date
        for ticker, weight in self.positions.items():
            if ticker not in price_data.columns:
                print(f"Ticker {ticker} not found in price data, will be excluded.")  # Debugging
                continue  # Skip tickers that don't exist in the data

            if pd.notna(price_data.loc[date, ticker]):
                valid_positions[ticker] = weight
            else:
                print(f"Ticker {ticker} has no data for {date}, it will be excluded.")  # Debugging

        if valid_positions:
            # Recalculate the weights to ensure they sum to 100%
            total_weight = sum(valid_positions.values())
            for ticker, weight in valid_positions.items():
                # Normalize the weight with respect to total valid weight
                price = price_data.loc[date, ticker]
                total_value += (weight / total_weight) * price * (1 - self.com)  # Apply transaction commission
                print(f"Ticker: {ticker}, Adjusted Weight: {weight / total_weight}, Price: {price}")  # Debugging

        # Ensure that the portfolio value is updated
        if total_value > 0:
            self.portfolio_value.append(total_value)
            print(f"Total portfolio value on {date}: {total_value}")  # Debugging
        else:
            if self.portfolio_value:
                self.portfolio_value.append(self.portfolio_value[-1])
            else:
                self.portfolio_value.append(self.cash)
            print(f"No valid portfolio value on {date}, maintaining previous value.")  # Debugging

    def run_simulations(self, valid_tickers, assets_adj_close_data, model_data, sp500_data, num_simulations=2000):
        """
        Run multiple simulations using different sets of selected assets.

        Args:
            valid_tickers (list): List of valid tickers to choose from.
            assets_adj_close_data (pd.DataFrame): Historical price data for assets.
            model_data (pd.DataFrame): Data with model predictions.
            sp500_data (pd.Series): S&P 500 data.
            num_simulations (int): Number of simulations to run (default: 2000).
            
        Returns:
            history (list): List containing portfolio values for each simulation.
        """
        history = []

        for sim in range(num_simulations):
            try:
                # Randomly choose 10 assets from the valid tickers list
                selected_tickers = np.random.choice(valid_tickers, 10, replace=False)
                
                # Run backtesting with selected tickers
                portfolio_values = self.run_backtest(assets_adj_close_data[selected_tickers], model_data)
                
                # Ensure the portfolio values align with the S&P 500 dates
                portfolio_values_series = pd.Series(portfolio_values, index=model_data.index)  # Ensure it aligns with model dates
                
                # Append the simulation result to history
                history.append(portfolio_values_series)
            except Exception as e:
                print(f"Simulation {sim+1} failed due to: {e}")
                continue
        
        return history
    
    def run_backtest(self, price_data, model_data):
        """
        Runs the backtesting cycle, adjusting the portfolio based on the model's signal and calculating returns.
        Filters tickers that don't have valid price data.
        
        Args:
            price_data (pd.DataFrame): Historical price data indexed by date.
            model_data (pd.DataFrame): DataFrame with model predictions, including trend signal (y).
        
        Returns:
            list: Portfolio values over time.
        """
        def filter_valid_tickers(price_data, tickers):
            """
            Filters tickers that exist in the price data.
            
            Args:
                price_data (pd.DataFrame): DataFrame containing historical price data.
                tickers (list): List of tickers to check against the price data.
                
            Returns:
                list: A list of valid tickers that are found in the price data.
            """
            # Ensure the tickers in the list have the correct "_AC" suffix
            tickers_with_ac = [ticker + '_AC' if not ticker.endswith('_AC') else ticker for ticker in tickers]
            
            # Find tickers that exist in the price data
            valid_tickers = [ticker for ticker in tickers_with_ac if ticker in price_data.columns]
            
            if not valid_tickers:
                raise ValueError("No valid tickers found in the price data.")
            
            return valid_tickers
        
        all_tickers = [ticker for ticker in self.portfolio_manager.assets_data['Ticker']]
        valid_tickers = filter_valid_tickers(price_data, all_tickers)
        
        # Ensure the price data is aligned with valid tickers only
        price_data = price_data[valid_tickers]
        
        for i, row in model_data.iterrows():
            date = row.name  # Access the date from the index, not as a column
            trend_signal = row['Y']  # Trend signal from the model (-1, 0, 1)
            
            # Adjust the portfolio based on the model's trend signal (y)
            print(f"\nDate: {date}, Trend Signal: {trend_signal}")  # Debugging
            self.adjust_portfolio(trend_signal)
            
            # Calculate the portfolio value for the current date
            if date in price_data.index:
                self.calculate_portfolio_value(price_data, date)
        
        return self.portfolio_value

    def simulate_portfolios(self, price_data, model_data, num_simulations=2000):
        """
        Runs multiple portfolio simulations, storing the history of each simulation.

        Args:
            price_data (pd.DataFrame): Historical price data.
            model_data (pd.DataFrame): Model predictions with trend signals.
            num_simulations (int): Number of simulations to run.

        Returns:
            list: List containing the historical portfolio values for each simulation.
        """
        history = []

        for sim in range(num_simulations):
            print(f"\nRunning simulation {sim + 1}/{num_simulations}")

            # Reset the portfolio value for each simulation
            self.portfolio_value = []

            # Select random assets and run backtest
            temp = np.random.choice(price_data.columns, 10)  # Randomly choose 10 tickers
            selected_price_data = price_data[temp]  # Use only selected tickers for this simulation

            # Run backtest with selected tickers
            portfolio_values = self.run_backtest(selected_price_data, model_data)

            # Append portfolio values to history
            history.append(portfolio_values)

        return history
    
    def calculate_performance_metrics(self, history):
        """
        Calculate performance metrics such as average final return and average Omega ratio for all simulations.

        Args:
            history (list of pd.Series): List containing portfolio values for each simulation.

        Returns:
            dict: Performance metrics including average final return and average Omega ratio.
        """
        final_returns = []
        omega_ratios = []

        for simulation in history:
            # Calculate final return as a percentage
            final_return = (simulation.iloc[-1] - simulation.iloc[0]) / simulation.iloc[0] * 100
            final_returns.append(final_return)
            
            # Calculate daily returns for Omega Ratio
            daily_returns = simulation.pct_change().dropna()
            
            # Calculate Omega ratio for the portfolio
            omega_ratio = self.portfolio_manager.calculate_omega_ratio(daily_returns, target_return=0.02)
            omega_ratios.append(omega_ratio)

        # Calculate average metrics
        avg_final_return = np.mean(final_returns)
        avg_omega_ratio = np.mean(omega_ratios)

        return {
            "Average Final Return (%)": avg_final_return,
            "Average Omega Ratio": avg_omega_ratio
        }


    def plot_performance(self, sp500_data, dates):
        """
        Plots the portfolio returns versus the S&P 500 benchmark returns, with final returns and Omega Ratio in the title.

        Args:
            sp500_data (pd.Series): The S&P 500 index values as a pandas Series.
            dates (list): List of dates corresponding to the portfolio values.
        """
        # Ensure that the number of dates matches the portfolio values
        if len(dates) != len(self.portfolio_value):
            print(f"Warning: Number of dates ({len(dates)}) and portfolio values ({len(self.portfolio_value)}) do not match.")
            dates = dates[:len(self.portfolio_value)]  # Align the dates with the portfolio values

        # Calculate returns for the portfolio
        portfolio_returns = pd.Series(self.portfolio_value, index=dates).pct_change().fillna(0)  # Daily returns of the portfolio
        cumulative_portfolio_returns = (1 + portfolio_returns).cumprod() - 1  # Cumulative returns

        # Calculate returns for the S&P 500 as a pandas Series
        sp500_returns = sp500_data.pct_change().fillna(0)  # Daily returns of the S&P 500
        cumulative_sp500_returns = (1 + sp500_returns[:len(self.portfolio_value)]).cumprod() - 1  # Cumulative returns

        # Calculate final returns for the portfolio and S&P 500 in percentage
        portfolio_return_percentage = cumulative_portfolio_returns.iloc[-1] * 100
        sp500_return_percentage = cumulative_sp500_returns.iloc[-1] * 100

        # Calculate Omega Ratio for the portfolio
        omega_ratio_portfolio = self.portfolio_manager.calculate_omega_ratio(portfolio_returns, target_return=0.02)

        # Plot the performance
        plt.figure(figsize=(10, 6))

        # Plot the cumulative returns of the portfolio
        plt.plot(dates, cumulative_portfolio_returns, label='Dynamic Portfolio', color='blue')

        # Plot the cumulative returns of the S&P 500 (benchmark)
        plt.plot(dates, cumulative_sp500_returns, label='S&P 500 (Benchmark)', color='green')

        # Add title with final returns and Omega Ratio
        plt.title(f"Final Returns: Portfolio {portfolio_return_percentage:.2f}% vs S&P 500 {sp500_return_percentage:.2f}% | Omega Ratio: {omega_ratio_portfolio:.2f}")
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns (%)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_simulation_performance(self, history, sp500_data):
        """
        Plots the average portfolio performance across all simulations and compares it with the S&P 500.
        
        Args:
            history (list of lists): List containing portfolio values for each simulation.
            sp500_data (pd.Series): The S&P 500 index values as a pandas Series.
        """
        # Convert history (list of lists) into a DataFrame for easier manipulation
        simulations_df = pd.DataFrame(history)

        # Ensure we have matching dates and values for all simulations
        if len(simulations_df) != len(sp500_data):
            print(f"Warning: Number of dates ({len(simulations_df)}) and S&P 500 values ({len(sp500_data)}) do not match.")
            min_length = min(len(simulations_df), len(sp500_data))
            simulations_df = simulations_df.iloc[:min_length]
            sp500_data = sp500_data.iloc[:min_length]

        # Calculate the average performance across all simulations
        avg_portfolio_performance = simulations_df.mean(axis=1)

        # Calculate the cumulative returns for the average portfolio performance
        avg_portfolio_returns = avg_portfolio_performance.pct_change().fillna(0)
        cumulative_avg_portfolio_returns = (1 + avg_portfolio_returns).cumprod() - 1

        # Calcular los rendimientos acumulados solo una vez para el S&P 500
        sp500_returns = sp500_data.pct_change().fillna(0)
        cumulative_sp500_returns = (1 + sp500_returns).cumprod() - 1

        # Ensure the lengths of the data match
        min_length = min(len(cumulative_avg_portfolio_returns), len(cumulative_sp500_returns))
        cumulative_avg_portfolio_returns = cumulative_avg_portfolio_returns[:min_length]
        cumulative_sp500_returns = cumulative_sp500_returns[:min_length]
        aligned_dates = sp500_data.index[:min_length]

        # Plot the performance
        plt.figure(figsize=(10, 6))

        # Plot the cumulative returns of the average portfolio performance
        plt.plot(aligned_dates, cumulative_avg_portfolio_returns, label='Average Portfolio Performance (Simulations)', color='blue')

        # Plot the cumulative returns of the S&P 500
        plt.plot(aligned_dates, cumulative_sp500_returns, label='S&P 500 (Benchmark)', color='green')

        # Add title, labels, and grid
        plt.title("Average Portfolio Performance (Simulations) vs S&P 500")
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns (%)')
        plt.legend()
        plt.grid(True)
        plt.show()


    def calculate_omega_ratio(self, portfolio_returns, target_return=0.20):
        """
        Calculates the Omega Ratio for the created portfolio.

        Args:
            portfolio_returns (pd.Series): The returns of the portfolio.
            target_return (float): Target return for calculating excess returns.

        Returns:
            float: Omega Ratio of the portfolio, or np.nan if invalid.
        """
        excess_returns = portfolio_returns - target_return

        gains = excess_returns[excess_returns > 0].sum()
        losses = -excess_returns[excess_returns < 0].sum()

        if losses == 0:
            if gains > 0:
                return np.inf  # Infinite Omega Ratio when there are no losses
            else:
                return np.nan  # Undefined Omega Ratio if both gains and losses are zero or negative

        omega_ratio = gains / losses
        return omega_ratio
# --------------------------------------------------


