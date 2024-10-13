
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
import optuna
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.multiclass import OneVsRestClassifier

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

    def __init__(self, sp500_data, economic_indicators_data, umbral):

        self.economic_indicators_data = economic_indicators_data
        self.sp500_data = sp500_data
        self.umbral = umbral
        self.model_data = self.model_data_function(self.economic_indicators_data, self.sp500_data, self.umbral)

    # ------------------------------

    def model_data_function(self, economic_indicators_data, sp500_data, umbral):

        model_data = pd.merge(economic_indicators_data, sp500_data, on='Date', how='inner')
        model_data['^GSPC_R'] = model_data['^GSPC_AC'].pct_change().dropna()
        model_data['Y'] = model_data['^GSPC_R'].apply(lambda x: 1 if x > umbral else (-1 if x < -umbral else 0))
        model_data = model_data.dropna()
        model_data = model_data.shift(-1).dropna()

        return model_data

    # ------------------------------

    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt

    def logistic_regression(self):

        X = self.model_data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC_R']]
        y = self.model_data['Y']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

        # Utilizamos OneVsRestClassifier para manejar correctamente la clasificación multiclase
        self.lr_model = OneVsRestClassifier(LogisticRegression(max_iter=1000))

        self.lr_model.fit(X_train, y_train)

        y_pred = self.lr_model.predict(X_test)
        y_pred_proba = self.lr_model.predict_proba(X_test)  # Obtiene las probabilidades para cada clase

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Calcular la curva ROC y el AUC para clasificación multiclase
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')  # Aquí calculamos AUC directamente

        # Graficar la curva AUC-ROC
        fpr = dict()
        tpr = dict()

        # Graficamos la curva ROC para cada clase
        for i in range(len(self.lr_model.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} ROC curve (AUC = {roc_auc:.2f})')

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

        X = self.model_data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC_R']]
        y = self.model_data['Y'] + 1  # Ajuste del objetivo para evitar valores negativos

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

        # ----------

        def objective(trial):

            param = {
                'objective': 'multi:softprob',  # Softprob para multiclase
                'num_class': 3,  # Número de clases
                'n_estimators': trial.suggest_int('n_estimators', 2, 4),  # Reducción en el número de árboles
                'learning_rate': trial.suggest_float('learning_rate', 0.010, 0.020),  # Learning rate más bajo
                'max_depth': trial.suggest_int('max_depth', 1, 2),  # Reducción en la profundidad máxima
                'subsample': trial.suggest_float('subsample', 0.2, 0.5),  # Subsample más bajo
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.4),  # Reducción en el muestreo de columnas
                'gamma': trial.suggest_float('gamma', 0, 0.3),  # Gamma más bajo
                'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 2),  # Aumento de la regularización L1
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2),  # Aumento de la regularización L2
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),  # Aumentar min_child_weight para evitar overfitting
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
        y_pred_proba = self.xgb_model.predict_proba(X_test)  # Obtiene las probabilidades para cada clase

        # Ajustamos y_test y y_pred de nuevo para evitar problemas de desplazamiento
        y_test_adjusted = y_test - 1
        y_pred_adjusted = y_pred - 1

        accuracy = accuracy_score(y_test_adjusted, y_pred_adjusted)
        report = classification_report(y_test_adjusted, y_pred_adjusted)

        # Calcular la curva ROC y el AUC para clasificación multiclase usando 'ovr'
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

        # Graficar la curva AUC-ROC para cada clase
        fpr = dict()
        tpr = dict()

        for i in range(3):  # Como tenemos 3 clases, generamos la curva para cada una
            fpr[i], tpr[i], _ = roc_curve(y_test_adjusted, y_pred_proba[:, i], pos_label=i)
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} ROC curve (AUC = {roc_auc:.2f})')

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
        y_pred_proba = self.mlp_model.predict_proba(X_test)  # Probabilidades para calcular el AUC-ROC

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=1)

        # Calcular la curva ROC y el AUC para clasificación multiclase usando 'ovr'
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

        # Graficar la curva AUC-ROC para cada clase
        fpr = dict()
        tpr = dict()

        for i in range(len(self.mlp_model.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} ROC curve (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) - MLP ({activation})')
        plt.legend(loc="lower right")
        plt.show()

        return accuracy, report


    # ------------------------------

    def optimized_MLP(self, n_trials=50):

        def objective(trial):

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

        self.mlp_model.fit(X_train, y_train)

        y_pred = self.mlp_model.predict(X_test)
        y_pred_proba = self.mlp_model.predict_proba(X_test)  # Probabilidades para calcular el AUC-ROC

        # Calcular la curva ROC y el AUC para clasificación multiclase usando 'ovr'
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

        # Graficar la curva AUC-ROC para cada clase
        fpr = dict()
        tpr = dict()

        for i in range(len(self.mlp_model.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} ROC curve (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) - Optimized MLP')
        plt.legend(loc="lower right")
        plt.show()

        joblib.dump(self.mlp_model, 'best_mlp_model.pkl')
        print("Best MLP model saved as 'best_mlp_model.pkl'")


    # ------------------------------

    def save_data(self, filepath):

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

        self.excel_file_path = excel_file_path
        self.assets_data = self.load_assets_data()
        self.selected_model = selected_model
        self.model_data = model_data

    # ------------------------------

    def load_assets_data(self):

        try:
            df = pd.read_excel(self.excel_file_path, sheet_name='beta')
            df.columns = ['Ticker', 'Beta', 'Nature']
            return df

        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return None

    # ------------------------------

    def classify_assets(self):

        self.assets_data['Ticker_AC'] = self.assets_data['Ticker'] + '_AC'

        pro_ciclics = self.assets_data[self.assets_data['Beta'] > 0.7]
        anti_ciclics = self.assets_data[(self.assets_data['Beta'] >= 0) & (self.assets_data['Beta'] <= 0.7)]

        return pro_ciclics, anti_ciclics

    # ------------------------------
    
    def predict_y(self):

        model_filename = f'Models/{self.selected_model}.pkl'
        
        try:
            model = joblib.load(model_filename)
            y_predicted = model.predict(self.model_data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC_R']])
            
            return y_predicted

        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    # ------------------------------

    def create_portfolio(self):

        y_predicted = self.predict_y()[-1]

        pro_ciclics, anti_ciclics = self.classify_assets()

        if y_predicted == 1:
            pro_ciclic_weight = 0.75
            anti_ciclic_weight = 0.25

        elif y_predicted == 0:
            pro_ciclic_weight = 0.50
            anti_ciclic_weight = 0.50

        elif y_predicted == -1:
            pro_ciclic_weight = 0.25
            anti_ciclic_weight = 0.75

        else:
            raise ValueError("Invalid predicted value for y. Must be -1, 0, or 1.")

        portfolio = {}

        # ----------

        if not pro_ciclics.empty:
            pro_ciclic_assets = pro_ciclics['Ticker'].tolist()
            pro_ciclic_allocation = pro_ciclic_weight / len(pro_ciclic_assets)

            for asset in pro_ciclic_assets:
                portfolio[f"{asset}_AC"] = pro_ciclic_allocation

        # ----------

        if not anti_ciclics.empty:
            anti_ciclic_assets = anti_ciclics['Ticker'].tolist()
            anti_ciclic_allocation = anti_ciclic_weight / len(anti_ciclic_assets)

            for asset in anti_ciclic_assets:
                portfolio[f"{asset}_AC"] = anti_ciclic_allocation

        return portfolio

    # ------------------------------

    def load_historical_prices(self):

        try:
            adj_close_data = pd.read_excel(self.excel_file_path, sheet_name='adj_close', index_col=0, parse_dates=True)
            return adj_close_data

        except Exception as e:
            print(f"Error loading historical prices: {e}")
            return None

    # ------------------------------

    def calculate_portfolio_returns(self, portfolio, target_return=0):

        historical_prices = self.load_historical_prices()

        portfolio_tickers = [f"{ticker}_AC" for ticker in portfolio.keys()]
        valid_tickers = [ticker for ticker in portfolio_tickers if ticker in historical_prices.columns]
        historical_prices = historical_prices[valid_tickers]

        daily_returns = historical_prices.pct_change().dropna()

        portfolio_weights = np.array([portfolio[ticker.replace('_AC', '')] for ticker in valid_tickers])
        portfolio_returns = daily_returns.dot(portfolio_weights)

        return portfolio_returns

    # ------------------------------

    def calculate_omega_ratio(self, target_return=0):

        portfolio = self.create_portfolio()
        portfolio_returns = self.calculate_portfolio_returns(portfolio)

        excess_returns = portfolio_returns - target_return

        gains = excess_returns[excess_returns > 0].sum()
        losses = -excess_returns[excess_returns < 0].sum()

        if losses == 0:
            return np.inf

        omega_ratio = gains / losses

        return omega_ratio

    # ------------------------------

    def print_portfolio_with_omega(self, target_return=0):

        portfolio = self.create_portfolio()
        omega_ratio = self.calculate_omega_ratio(target_return)

        print(f"Portfolio:")
        for ticker, weight in portfolio.items():
            print(f"{ticker}: {weight*100:.2f}%")

        print(f"\nOmega Ratio: {omega_ratio:.4f} (Target return: {target_return})")

# --------------------------------------------------










# --------------------------------------------------

# 
class DynamicBacktesting:

    def __init__(self):

        return

    # ------------------------------

    def first_function(self):

        return

 # ------------------------------

    def second_function(self):

        return

 # ------------------------------

    def third_function(self):

        return
