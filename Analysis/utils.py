
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
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV

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

class HistoricalDataDownloader:

    def __init__(self, tickers):
        self.tickers = tickers
        self.start_date = '2000-01-01'
        self.end_date = '2024-06-01'
        self.adj_close = pd.DataFrame()
        self.beta = pd.DataFrame()
        self.rename_tickers()

    # ------------------------------

    def rename_tickers(self):
        ticker_rename_map = {
            'BRK.B': 'BRK-B',
            'BF.B': 'BF-B'
        }
        self.tickers = [ticker_rename_map.get(ticker, ticker) for ticker in self.tickers]

    # ------------------------------

    def download_adj_close(self):

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

        bins = [-np.inf, 0.7, np.inf]
        labels = ['0', '1']

        self.beta['Nature'] = pd.cut(self.beta['Beta'], bins=bins, labels=labels)

    # ------------------------------

    def save_data(self, filepath):

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

# 
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

    def train_logistic_regression(self):

        X = self.model_data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC_R']]
        y = self.model_data['Y']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        self.lr_model = LogisticRegression(
            solver='saga',
            C=0.3,
            penalty='l2',
            class_weight='balanced'
        )
        
        self.lr_model.fit(X_train, y_train)
        
        y_pred = self.lr_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        self.X_test = X_test
        self.y_test = y_test
        self.lr_y_pred = y_pred
        
        return accuracy, report

    # ------------------------------

    def train_xgboost(self):

        X = self.model_data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC_R']]
        y = self.model_data['Y'] + 1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

        param_distributions = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.5]
        }

        randomized_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, cv=5, scoring='accuracy', verbose=1, n_jobs=-1, n_iter=100)
        randomized_search.fit(X_train, y_train)

        self.xgb_model = randomized_search.best_estimator_

        y_pred = self.xgb_model.predict(X_test)

        y_pred_original = y_pred - 1

        accuracy = accuracy_score(y_test, y_pred_original)
        report = classification_report(y_test - 1, y_pred_original)

        self.X_test = X_test
        self.y_test = y_test
        self.xgb_y_pred = y_pred

        print(f"Best hyperparameters: {randomized_search.best_params_}")

        return accuracy, report

    # ------------------------------

    def train_mlp(self, activation='relu'):

        X = self.model_data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC_R']]
        y = self.model_data['Y']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
        
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
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=1)
        
        self.X_test = X_test
        self.y_test = y_test
        self.mlp_y_pred = y_pred
        
        return accuracy, report

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
