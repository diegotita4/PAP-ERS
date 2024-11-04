
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: Estrategias de Rotación Sectorial (ERS)                                                    -- #
# -- script: Analysis.py - Python script with the main functionality                                     -- #
# -- authors: diegotita4 - Antonio-IF - JoAlfonso - Oscar148                                             -- #
# -- license: GNU GENERAL PUBLIC LICENSE - Version 3, 29 June 2007                                       -- #
# -- repository: https://github.com/diegotita4/PAP-ERS                                                   -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# --------------------------------------------------

# LIBRARIES
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from utils import Models as M
from utils import EDA_comparison as EDA
from utils import PortfolioManager as PM
from utils import HistoricalDataDownloader as HDD
from utils import DynamicBacktesting as DBT

# --------------------------------------------------

# TICKER FOR THE BENCHMARK
benchmark = ['^GSPC']

# ----------

# INITIALIZE HDD FOR THE BENCHMARK
# HDD_sp500 = HDD(tickers=benchmark)
# HDD_sp500.download_adj_close()
# HDD_sp500.save_data(filepath="Data/sp500_data.xlsx")

# ------------------------------

# READ ECONOMIC INDICATORS AND S&P 500 DATA
economic_indicators_data = pd.read_excel("Data/economic_indicators_data.xlsx")
sp500_data = pd.read_excel("Data/sp500_data.xlsx")

# --------------------------------------------------

# PERFORM EDA COMPARISON
# EDA_comparison = EDA(sp500_data, economic_indicators_data)
# EDA_comparison.perform_EDA_comparison()

# --------------------------------------------------

# READ S&P 500 ASSETS LIST
assets = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]["Symbol"].tolist()

# ----------

# INITIALIZE HDD FOR THE S&P 500 ASSETS
# HDD_assets = HDD(tickers=assets)
# HDD_assets.download_adj_close()
# HDD_assets.download_beta()
# HDD_assets.classify_beta()
# HDD_assets.save_data(filepath="Data/assets_data.xlsx")

# ------------------------------

# READ ASSETS DATA
assets_data = pd.read_excel("Data/assets_data.xlsx")
print("Columnas en assets_data:", assets_data.columns)  # Verificar los nombres de las columnas
assets_adj_close_data = pd.read_excel("Data/assets_data.xlsx", sheet_name="adj_close")
assets_beta_data = pd.read_excel("Data/assets_data.xlsx", sheet_name="beta")

# --------------------------------------------------

# Asegurarse de que la columna 'Date' sea el índice en los datos de precios
assets_adj_close_data.set_index('Date', inplace=True)
sp500_data.set_index('Date', inplace=True)

# UMBRAL
umbral = 0.02

# ----------

# INITIALIZE THE MODELS CLASS WITH ECONOMIC INDICATORS AND S&P 500 DATA
M_model = M(sp500_data, economic_indicators_data, umbral)
#M_model.save_data('Data/model_data.xlsx')

# ------------------------------

# READ MODEL DATA
model_data = pd.read_excel("Data/model_data.xlsx")
model_data.set_index('Date', inplace=True)

# ------------------------------

# TRAIN LOGISTIC REGRESSION (LR) MODEL
#lr_model = M_model.logistic_regression()

# ----------

# TRAIN OPTIMIZED LOGISTIC REGRESSION (LR) MODEL
#optimized_lr_model = M_model.optimized_logistic_regression()

# --------------------

# TRAIN MULTI-LAYER PERCEPTRON (MLP) NEURAL NETWORK WITH RELU ACTIVATION
#print("\n--- Training MLP with ReLU Activation ---")
#mlp_accuracy_relu, mlp_report_relu = M_model.MLP(activation='relu')
#print("MLP Neural Network (ReLU) Accuracy:", mlp_accuracy_relu)
#print(mlp_report_relu)

# ----------

# TRAIN MULTI-LAYER PERCEPTRON (MLP) NEURAL NETWORK WITH TANH ACTIVATION
#mlp_accuracy_tanh, mlp_report_tanh = M_model.MLP(activation='tanh')
#print("MLP Neural Network (tanh) Accuracy:", mlp_accuracy_tanh)
#print(mlp_report_tanh)

# ----------

# TRAIN MULTI-LAYER PERCEPTRON (MLP) NEURAL NETWORK WITH LOGISTIC ACTIVATION
#mlp_accuracy_logistic, mlp_report_logistic = M_model.MLP(activation='logistic')
#print("MLP Neural Network (logistic) Accuracy:", mlp_accuracy_logistic)
#print(mlp_report_logistic)

# ----------

# TRAIN OPTIMIZED MULTI-LAYER PERCEPTRON (MLP)
# M_model.optimize_mlp_with_optuna()

# --------------------

# TRAIN XGBOOST MODEL
#xgb_model = M_model.XGBoost()

# ----------

# TRAIN OPTIMIZED XGBOOST MODEL
# optimized_xgb_model = M_model.optimized_XGBoost()

# ----------

# --------------------------------------------------
# RUN DYNAMIC BACKTESTING

# --------------------------------------------------
# LIBRARIES
import pandas as pd
import numpy as np
from utils import PortfolioManager as PM
from utils import DynamicBacktesting as DBT
import matplotlib.pyplot as plt

# --------------------------------------------------
# CLASS TO LOAD AND HANDLE DATA

# --------------------------------------------------
# CLASS TO LOAD AND HANDLE DATA

class DataHandler:
    """Class to handle data loading from Excel files."""
    def __init__(self, assets_path, model_path):
        self.assets_path = assets_path
        self.model_path = model_path
        self.adj_close = None
        self.beta_data = None
        self.model_data = None

    def load_assets_data(self):
        """Loads asset price data and beta classification data."""
        # Load adjusted close prices and ensure Date is parsed as datetime
        self.adj_close = pd.read_excel(self.assets_path, sheet_name='adj_close')
        self.adj_close['Date'] = pd.to_datetime(self.adj_close['Date'], errors='coerce')
        self.adj_close.set_index('Date', inplace=True)
        
        # Load beta data
        self.beta_data = pd.read_excel(self.assets_path, sheet_name='beta')

    def load_model_data(self):
        """Loads economic indicators and expected trend data."""
        # Load model data and ensure Date is parsed as datetime
        self.model_data = pd.read_excel(self.model_path)
        self.model_data['Date'] = pd.to_datetime(self.model_data['Date'], errors='coerce')
        self.model_data.set_index('Date', inplace=True)

    def align_data(self):
        """Aligns adj_close and model_data to have common dates only starting from the year 2000."""
        # Filter dates from 2000 onwards
        self.adj_close = self.adj_close[self.adj_close.index >= "2000-01-01"]
        self.model_data = self.model_data[self.model_data.index >= "2000-01-01"]

        # Get common dates between adj_close and model_data
        common_dates = self.adj_close.index.intersection(self.model_data.index)
        self.adj_close = self.adj_close.loc[common_dates]
        self.model_data = self.model_data.loc[common_dates]

        print(f"Data aligned to {len(common_dates)} common dates, starting from 2000.")




# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
# LOAD AND PROCESS DATA

# Initialize DataHandler and load data
data_handler = DataHandler("Data/assets_data.xlsx", "Data/model_data.xlsx")
data_handler.load_assets_data()
data_handler.load_model_data()

# Align data to ensure dates are consistent and start from 2000
data_handler.align_data()

# --------------------------------------------------
# RUN DYNAMIC BACKTESTING

# Initialize PortfolioManager with aligned data
portfolio_manager = PM(data_handler.assets_path, data_handler.model_data)

# Define initial capital for backtesting
initial_capital = 1_000_000

# Initialize DynamicBacktesting with initial capital and PortfolioManager
dynamic_backtesting = DBT(initial_capital, portfolio_manager)

# Run backtesting cycle
dynamic_backtesting.run_backtest()

# Display portfolio value history
print("Portfolio Value History:", dynamic_backtesting.portfolio_value)

# Optional: Plot the results of the backtesting
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(dynamic_backtesting.portfolio_value, label="Portfolio Value")
plt.xlabel("Rebalance Date")
plt.ylabel("Portfolio Value (USD)")
plt.title("Backtesting Results: Portfolio Value Over Time")
plt.legend()
plt.show()



# --------------------------------------------------







# --------------------------------------------------

# INITIALIZE THE DYNAMIC BACKTESTING CLASS
#DBT_dynback = DBT()
#DBT_dynback.first_function()
#DBT_dynback.second_function()
#DBT_dynback.third_function()
