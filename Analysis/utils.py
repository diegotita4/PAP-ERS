
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
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

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
                    return None

                ticker_data = self.adj_close[[f'{ticker}_AC']].dropna()
                
                benchmark_ticker = '^GSPC'
                benchmark_data = yf.download(benchmark_ticker, start=self.start_date, end=self.end_date, interval='1mo')[['Adj Close']].dropna()

                if ticker_data.empty or benchmark_data.empty:
                    return None

                merged_data = ticker_data.join(benchmark_data, how='inner')
                ticker_returns = merged_data[f'{ticker}_AC'].pct_change().dropna()
                benchmark_returns = merged_data['Adj Close'].pct_change().dropna()

                cov_matrix = ticker_returns.cov(benchmark_returns)
                benchmark_var = benchmark_returns.var()

                beta = cov_matrix / benchmark_var
                return beta if isinstance(beta, (int, float)) else beta.values[0]

            except Exception as e:
                print(f"Failed to calculate beta for {ticker}: {e}")
                return None

        def fetch_beta(ticker):
            try:
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
                print(f"Failed to download beta for {ticker}: {e}")
                return {"Ticker": ticker, "Beta": None}

        with ThreadPoolExecutor(max_workers=10) as executor:
            betas = list(executor.map(fetch_beta, self.tickers))

        self.beta = pd.DataFrame(betas)
        self.beta = self.beta[self.beta['Beta'].notnull()]

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

    def __init__(self, indicators_df, sp500_df):

        # Shift economic indicators to reflect predictive nature
        indicators_df = self._shift_indicators(indicators_df)
        
        # Combine economic indicators with historical S&P 500 data
        self.data = self._merge_data(indicators_df, sp500_df)

    # ------------------------------

    def _shift_indicators(self, indicators_df):

        # Shift indicators by 3 months to predict future S&P 500 movements
        indicators_df['CLI'] = indicators_df['CLI'].shift(3)
        indicators_df['BCI'] = indicators_df['BCI'].shift(3)
        indicators_df['CCI'] = indicators_df['CCI'].shift(3)
        return indicators_df

    # ------------------------------

    def _merge_data(self, indicators_df, sp500_df):

        # Convert 'Date' columns to datetime format
        indicators_df['Date'] = pd.to_datetime(indicators_df['Date'])
        sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])
        
        # Merge both datasets based on 'Date'
        merged_data = pd.merge(indicators_df, sp500_df, on='Date', how='inner')
        
        # Create percentage change column for the S&P 500
        merged_data['SP500_Change'] = merged_data['^GSPC_AC'].pct_change()
        
        # Define the target column (1: overweight, 0: neutral, -1: underweight)
        merged_data['Target'] = merged_data['SP500_Change'].apply(lambda x: 1 if x > 0.02 else (-1 if x < -0.02 else 0))
        
        # Drop rows with any NaN values in the predictors or target columns
        merged_data.dropna(inplace=True)
        
        return merged_data

    # ------------------------------

    def train_logistic_regression(self):

        # Define predictor variables (CLI, BCI, GDP, CCI, and S&P 500 historical data) and the target variable (Target)
        X = self.data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC_AC']]
        y = self.data['Target']
        
        # Split the dataset into training and testing sets (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the logistic regression model with adjusted hyperparameters
        self.lr_model = LogisticRegression(
            solver='saga',   # You can experiment with 'saga' or other solvers
            C=0.3,                # Regularization strength (smaller values = stronger regularization)
            penalty='l2',          # Regularization type (l1, l2, or elasticnet)
            class_weight='balanced' # Handle class imbalance
        )
        
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
    

    # ------------------------------

    def train_xgboost(self):
        # Define features and target
        X = self.data[['CLI', 'BCI', 'CCI', '^GSPC_AC', 'SP500_Change']]
        y = self.data['Target']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the XGBoost classifier with default parameters
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

        # Define the grid of hyperparameters to search
        param_grid = {
            'n_estimators': [100, 200, 300],      # Number of trees
            'learning_rate': [0.01, 0.1, 0.3],    # Learning rate
            'max_depth': [3, 5, 7],               # Maximum depth of trees
            'subsample': [0.6, 0.8, 1.0],         # Fraction of samples to use for each tree
            'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of features to use for each tree
            'gamma': [0, 0.1, 0.5]                # Minimum loss reduction for making a split
        }

        # Use GridSearchCV to find the best hyperparameters
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)

        # Save the best model
        self.xgb_model = grid_search.best_estimator_

        # Predict using the best model found by GridSearchCV
        y_pred = self.xgb_model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Save test data and predictions
        self.X_test = X_test
        self.y_test = y_test
        self.xgb_y_pred = y_pred

        print(f"Best hyperparameters: {grid_search.best_params_}")
        return accuracy, report

    # ------------------------------

    def train_mlp(self, activation='relu'):

        # Define predictor variables (CLI, BCI, GDP, CCI, and S&P 500 historical data) and the target variable (Target)
        X = self.data[['CLI', 'BCI', 'GDP', 'CCI', '^GSPC_AC']]
        y = self.data['Target']
        
        # Scale data using StandardScaler for better neural network performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split the dataset into training and testing sets (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train a Multi-Layer Perceptron (MLP) Neural Network
        # Modificación en el constructor del MLPClassifier para mejorar el rendimiento
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),  # Aumento en el número de neuronas en las capas ocultas
            max_iter=1500,                     # Incrementar el número de iteraciones para mayor convergencia
            activation='relu',                 # Mantener ReLU como función de activación
            solver='adam',                     # Adam es un buen optimizador por defecto
            alpha=0.005,                       # Aumentar ligeramente la regularización para mejorar la generalización
            learning_rate_init=0.0005,         # Reducir la tasa de aprendizaje para ajustes más finos
            random_state=42                    # Fijar la semilla para replicabilidad
        )
        
        self.mlp_model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = self.mlp_model.predict(X_test)
        
        # Calculate accuracy and classification report
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Save test data and predictions
        self.X_test = X_test
        self.y_test = y_test
        self.mlp_y_pred = y_pred
        
        return accuracy, report

    # ------------------------------

    def download_sp500_data(self):

        # Download historical data for S&P 500 from Yahoo Finance
        sp500_data = yf.download('^GSPC', start='2010-01-01', end='2024-07-01')
        sp500_data.reset_index(inplace=True)
        return sp500_data
