
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Estrategias de Rotación Sectorial (ERS)                                                    -- #
# -- script: Analysis.py - Python script with the main functionality                                     -- #
# -- authors: diegotita4 - Antonio-IF - JoAlfonso - Oscar148                                             -- #
# -- license: GNU GENERAL PUBLIC LICENSE - Version 3, 29 June 2007                                       -- #
# -- repository: https://github.com/diegotita4/PAP-ERS                                                   -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# --------------------------------------------------

# LIBRARIES
import pandas as pd
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
assets_adj_close_data = pd.read_excel("Data/assets_data.xlsx", sheet_name="adj_close")
assets_beta_data = pd.read_excel("Data/assets_data.xlsx", sheet_name="beta")

# --------------------------------------------------

# UMBRAL
umbral = 0.02

# ----------

# INITIALIZE THE MODELS CLASS WITH ECONOMIC INDICATORS AND S&P 500 DATA
M_model = M(sp500_data, economic_indicators_data, umbral)
#M_model.save_data('Data/model_data.xlsx')

# ------------------------------

# READ MODEL DATA
model_data = pd.read_excel("Data/model_data.xlsx")

# ------------------------------

# TRAIN LOGISTIC REGRESSION (LR) MODEL
#lr_model = M_model.logistic_regression()

# ----------

# TRAIN OPTIMIZED LOGISTIC REGRESSION (LR) MODEL
#optimized_lr_model = M_model.optimized_logistic_regression()

# --------------------

# TRAIN MULTI-LAYER PERCEPTRON (MLP) NEURAL NETWORK WITH RELU ACTIVATION
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
#M_model.optimize_mlp_with_optuna()

# --------------------

# TRAIN XGBOOST MODEL
#xgb_model = M_model.XGBoost()

# ----------

# TRAIN OPTIMIZED XGBOOST MODEL
#optimized_xgb_model = M_model.optimized_XGBoost()

# --------------------------------------------------

# SELECTED MODEL
selected_model = "logistic_regression"

# ----------

# 
PM_portfolio = PM(
    excel_file_path='Data/assets_data.xlsx',
    selected_model=selected_model,
    model_data=model_data
)

# ----------

# 
PM_portfolio.print_portfolio_with_omega(target_return=0)

# --------------------------------------------------










# --------------------------------------------------

# INITIALIZE THE DYNAMIC BACKTESTING CLASS
#DBT_dynback = DBT()
#DBT_dynback.first_function()
#DBT_dynback.second_function()
#DBT_dynback.third_function()
