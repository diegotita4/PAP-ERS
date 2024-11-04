
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
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from utils import Models as M
from utils import EDA_comparison as EDA
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

# Inicializa DynamicBacktesting
# Función para ejecutar una simulación única
# def run_single_backtest(i, data_handler):
#     portfolio = DBT.Portfolio(data_handler.beta_data, data_handler.adj_close)
#     backtest = DBT.Backtest(data_handler.model_data, portfolio)
#     backtest.run_backtest()
#     return backtest.portfolio_value_history[0], backtest.portfolio_value_history[-1]

# Inicializar DataHandler una vez
# data_handler = DBT.DataHandler("Data/assets_data.xlsx", "Data/model_data.xlsx")
# data_handler.load_assets_data()
# data_handler.load_model_data()

# Ejecutar las simulaciones en paralelo
# results = Parallel(n_jobs=12)(delayed(run_single_backtest)(i, data_handler) for i in range(2000))

# Guardar resultados
# results_df = pd.DataFrame(results, columns=['Valor Inicial', 'Valor Final'])
# results_df.to_excel("backtest_results_summary.xlsx", index=False)
# print("Resultados del backtesting guardados en 'backtest_results_summary.xlsx'")


# Checar metricas de las simulaciones: 
# Cargar los datos de resultados del backtesting
results_df = pd.read_excel("backtest_results.xlsx")

# Calcular rendimientos para cada simulación
results_df['Rendimiento'] = (results_df['Valor Final'] - results_df['Valor Inicial']) / results_df['Valor Inicial']

# Parámetros para el cálculo de métricas
risk_free_rate = 0.02  # Ejemplo de tasa libre de riesgo anual
benchmark_return = 0.07  # Supuesto retorno del índice de referencia (S&P 500)

# 1. Calcular el Ratio de Sharpe
mean_return = results_df['Rendimiento'].mean()
std_dev_return = results_df['Rendimiento'].std()
sharpe_ratio = (mean_return - risk_free_rate) / std_dev_return

# 2. Calcular el Omega Ratio
# Umbral de retorno mínimo (puede ser la tasa libre de riesgo)
threshold_return = risk_free_rate
excess_returns = results_df['Rendimiento'] - threshold_return

# Omega ratio: suma de retornos por encima del umbral / suma de retornos por debajo del umbral
omega_ratio = excess_returns[excess_returns > 0].sum() / abs(excess_returns[excess_returns < 0].sum())

# 3. Calcular el Alpha de Jensen
# Usamos el supuesto rendimiento de referencia (benchmark_return)
# Retorno promedio del portafolio - [Tasa libre de riesgo + Beta * (Rendimiento del benchmark - Tasa libre de riesgo)]
# Para simplificar, asumimos beta = 1 si el portafolio replica al mercado.
jensen_alpha = mean_return - (risk_free_rate + 1 * (benchmark_return - risk_free_rate))

# Resultados
print(f"Ratio de Sharpe: {sharpe_ratio}")
print(f"Omega Ratio: {omega_ratio}")
print(f"Alpha de Jensen: {jensen_alpha}")