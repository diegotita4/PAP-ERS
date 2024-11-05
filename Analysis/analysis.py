
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
#def run_single_backtest(i, data_handler):
#     portfolio = DBT.Portfolio(data_handler.beta_data, data_handler.adj_close)
#     backtest = DBT.Backtest(data_handler.model_data, portfolio)
#     backtest.run_backtest()
#     return backtest.portfolio_value_history

# Inicializar DataHandler una vez
#data_handler = DBT.DataHandler("Data/assets_data.xlsx", "Data/model_data.xlsx")
#data_handler.load_assets_data()
#data_handler.load_model_data()

# Ejecutar las simulaciones en paralelo
#results = Parallel(n_jobs=12)(delayed(run_single_backtest)(i, data_handler) for i in range(2000))

# Guardar resultados
#results_df = pd.DataFrame(results)
#results_df.to_excel("backtest_results_historicaldata.xlsx", index=False)
#print("Resultados del backtesting guardados en 'backtest_results_historicaldata.xlsx'")


# Checar metricas de las simulaciones: 

# Load backtesting results data for the portfolio
results_df = pd.read_excel("backtest_results_historicaldata.xlsx")

# Calculate returns for each simulation
initial_values = results_df.iloc[0]  # Initial values (first row)
final_values = results_df.iloc[-1]   # Final values (last row)
performance_df = pd.DataFrame({
    'Initial Value': initial_values,
    'Final Value': final_values
})
performance_df['Return'] = (performance_df['Final Value'] - performance_df['Initial Value']) / performance_df['Initial Value']

# Parameters for calculating metrics
risk_free_rate = 0.1050  # Annual risk-free rate
target_return = 0.05     # Target return for Omega Ratio calculation

# ------------------------------
# 1. Calculate Sharpe Ratio for the portfolio
mean_return = performance_df['Return'].mean()
std_dev_return = performance_df['Return'].std()
sharpe_ratio_portfolio = (mean_return - risk_free_rate) / std_dev_return if std_dev_return != 0 else float('nan')

# Identify the best and worst scenarios based on final value
best_scenario = performance_df['Final Value'].idxmax()
worst_scenario = performance_df['Final Value'].idxmin()

# ------------------------------
# 2. Calculate Jensen's Alpha and Omega Ratio for the portfolio

# Load S&P 500 data for comparison and calculate Jensen's Alpha
sp500_data = pd.read_excel("Data/sp500_data.xlsx")
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
sp500_data.set_index('Date', inplace=True)
sp500_data['Return'] = sp500_data['^GSPC_AC'].pct_change()

# Calculate average return and beta of the portfolio with respect to S&P 500
mean_return_sp500 = sp500_data['Return'].mean() * 252  # Annualized return
covariance = performance_df['Return'].cov(sp500_data['Return'])
variance_sp500 = sp500_data['Return'].var()
beta_portfolio = covariance / variance_sp500

# Calculate Jensen's Alpha for the portfolio
alpha_jensen_portfolio = mean_return - (risk_free_rate + beta_portfolio * (mean_return_sp500 - risk_free_rate))

# ------------------------------
# 3. Calculate the Omega Ratio for the portfolio
excess_returns_portfolio = performance_df['Return'] - target_return
omega_ratio_portfolio = (excess_returns_portfolio[excess_returns_portfolio > 0].sum() /
                         abs(excess_returns_portfolio[excess_returns_portfolio < 0].sum())) if excess_returns_portfolio[excess_returns_portfolio < 0].sum() != 0 else float('inf')

# ------------------------------
# 4. Calculate the Sharpe and Omega Ratio for S&P 500
std_dev_return_sp500 = sp500_data['Return'].std() * (252 ** 0.5)  # Annualized standard deviation
sharpe_ratio_sp500 = (mean_return_sp500 - risk_free_rate) / std_dev_return_sp500 if std_dev_return_sp500 != 0 else float('nan')

# Calculate the Omega Ratio for S&P 500
excess_returns_sp500 = sp500_data['Return'] - (target_return / 252)  # Daily adjustment for annual target
omega_ratio_sp500 = (excess_returns_sp500[excess_returns_sp500 > 0].sum() /
                     abs(excess_returns_sp500[excess_returns_sp500 < 0].sum())) if excess_returns_sp500[excess_returns_sp500 < 0].sum() != 0 else float('inf')

# ------------------------------
# Calculate average and median of all portfolio final values
average_final_value = performance_df['Final Value'].mean()
median_final_value = performance_df['Final Value'].median()

# ------------------------------
# Display Results
print("Performance Metrics:")

print("\nMetrics for S&P 500:")
print(f"  - Sharpe Ratio: {sharpe_ratio_sp500:.2f}")
print(f"  - Omega Ratio: {omega_ratio_sp500:.2f}")
print(f"  - Jensen's Alpha: 0 (as the benchmark)")

print("\nGeneral Metrics for Portfolio (all simulations):")
print(f"  - Sharpe Ratio: {sharpe_ratio_portfolio:.2f}")
print(f"  - Omega Ratio: {omega_ratio_portfolio:.2f}")

print(f"\nBest Scenario (Simulation {best_scenario}):")
print(f"  - Sharpe Ratio: {(performance_df.loc[best_scenario, 'Return'] - risk_free_rate) / std_dev_return:.2f}")
print(f"  - Omega Ratio: {omega_ratio_portfolio:.2f}")
print(f"  - Return: {performance_df.loc[best_scenario, 'Return'] * 100:.2f}%")

print(f"\nWorst Scenario (Simulation {worst_scenario}):")
print(f"  - Sharpe Ratio: {(performance_df.loc[worst_scenario, 'Return'] - risk_free_rate) / std_dev_return:.2f}")
print(f"  - Omega Ratio: {omega_ratio_portfolio:.2f}")
print(f"  - Return: {performance_df.loc[worst_scenario, 'Return'] * 100:.2f}%")

print("\nAverage and Median Final Values for all portfolios:")
print(f"  - Average Final Value: {average_final_value:.2f}")
print(f"  - Median Final Value: {median_final_value:.2f}")
