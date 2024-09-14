
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
from utils import Models
from utils import EDA_comparison as EDAC
from utils import HistoricalDataDownloader as HDD

# --------------------------------------------------

# INITIALIZE HDD FOR THE BENCHMARK
benchmark = ['^GSPC']
HDD_sp500 = HDD(tickers=benchmark)
HDD_sp500.download_adj_close()
HDD_sp500.save_data(filepath="Data/sp500_data.xlsx")

# ------------------------------

# READ ECONOMIC INDICATORS AND S&P 500 DATA
economic_indicators_data = pd.read_excel("Data/economic_indicators_data.xlsx")
sp500_data = pd.read_excel("Data/sp500_data.xlsx")

# --------------------------------------------------

# PERFORM EDA COMPARISON
# eda_comparison = EDAC(sp500_data, economic_indicators_data)
# eda_comparison.perform_EDA_comparison()

# --------------------------------------------------

# READ S&P 500 ASSETS LIST
assets = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]["Symbol"].tolist()

# ------------------------------

# INITIALIZE HDD FOR THE S&P 500 ASSETS
HDD_assets = HDD(tickers=assets)
HDD_assets.download_adj_close()
HDD_assets.download_beta()
HDD_assets.save_data(filepath="Data/assets_data.xlsx")

# ------------------------------

# READ ASSETS DATA
assets_adj_close_data = pd.read_excel("Data/assets_data.xlsx", sheet_name="adj_close")
assets_beta_data = pd.read_excel("Data/assets_data.xlsx", sheet_name="beta")

# --------------------------------------------------

# INITIALIZE THE MODELS CLASS WITH ECONOMIC INDICATORS AND S&P 500 DATA
model = Models(economic_indicators_data, sp500_data)

# ------------------------------

# TRAIN LOGISTIC REGRESSION MODEL
lr_accuracy, lr_report = model.train_logistic_regression()
print("Logistic Regression Accuracy:", lr_accuracy)
print(lr_report)

# ----------

# TRAIN MULTILAYER PERCEPTRON (MLP) NEURAL NETWORK WITH ReLU ACTIVATION
mlp_accuracy_relu, mlp_report_relu = model.train_mlp(activation='relu')
print("MLP Neural Network (ReLU) Accuracy:", mlp_accuracy_relu)
print(mlp_report_relu)

# ----------

# TRAIN MULTILAYER PERCEPTRON (MLP) NEURAL NETWORK WITH tanh ACTIVATION
mlp_accuracy_tanh, mlp_report_tanh = model.train_mlp(activation='tanh')
print("MLP Neural Network (tanh) Accuracy:", mlp_accuracy_tanh)
print(mlp_report_tanh)

# ----------

# TRAIN MULTILAYER PERCEPTRON (MLP) NEURAL NETWORK WITH LOGISTIC ACTIVATION
mlp_accuracy_logistic, mlp_report_logistic = model.train_mlp(activation='logistic')
print("MLP Neural Network (logistic) Accuracy:", mlp_accuracy_logistic)
print(mlp_report_logistic)

# --------------------------------------------------
