
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
from utils import Models as M
from utils import EDA_comparison as EDA
from utils import PortfolioManager as PM
from utils import HistoricalDataDownloader as HDD
from utils import DynamicBacktestingWithOmega as DBT

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
# PREDICTIONS AND PORTFOLIO MANAGEMENT
# --------------------------------------------------

# Select the model (MLP with ReLU activation)
selected_model = "mlp_relu"

# Initialize the PortfolioManager with the necessary data
portfolio_manager = PM(excel_file_path='Data/assets_data.xlsx', selected_model='mlp_relu', model_data=model_data)

# Initialize the dynamic backtesting class with the PortfolioManager and initial capital
backtest = DBT(1000000, portfolio_manager)

# Run dynamic backtesting using the asset price data and model predictions
portfolio_values = backtest.run_backtest(assets_adj_close_data, model_data)

# Display backtesting results
print("Portfolio Values: ", portfolio_values)

# --------------------------------------------------
# Calculate the returns of the portfolio and the S&P 500
# --------------------------------------------------

# Create a series of the portfolio values to calculate returns
portfolio_values_series = pd.Series(portfolio_values, index=model_data.index)

# Calculate daily returns for the portfolio
portfolio_returns = portfolio_values_series.pct_change().dropna()

# Calculate daily returns for the S&P 500
if '^GSPC_AC' in sp500_data.columns:
    sp500_returns = sp500_data['^GSPC_AC'].pct_change().dropna()
else:
    print("The column '^GSPC_AC' is not present in the S&P 500 data.")
    sp500_returns = None

# --------------------------------------------------
# Calculate the Omega Ratio for the portfolio and compare with the S&P 500
# --------------------------------------------------

# Calculate the Omega Ratio of the portfolio using the PortfolioManager function
omega_ratio_portfolio = portfolio_manager.calculate_omega_ratio(portfolio_returns, target_return=0.02)
print(f"Portfolio Omega Ratio: {omega_ratio_portfolio}")

# Calculate the Omega Ratio for the S&P 500, if data is available
if sp500_returns is not None:
    omega_ratio_sp500 = portfolio_manager.calculate_omega_ratio(sp500_returns, target_return=0.02)
    print(f"S&P 500 Omega Ratio: {omega_ratio_sp500}")

# --------------------------------------------------
# Plot portfolio performance versus the S&P 500
# --------------------------------------------------

if '^GSPC_AC' in sp500_data.columns:
    backtest.plot_performance(sp500_data['^GSPC_AC'], sp500_data.index)
else:
    print("The column '^GSPC_AC' is not present in the S&P 500 data.")

# --------------------------------------------------
# Define the function to get valid tickers
# --------------------------------------------------

def get_valid_tickers(historical_prices, min_required_dates):
    """
    Filters tickers that have valid historical data for at least the minimum required number of dates.
    
    Args:
        historical_prices (pd.DataFrame): DataFrame containing historical price data for all assets.
        min_required_dates (int): Minimum number of dates for which valid price data should be available.
    
    Returns:
        list: List of valid tickers with sufficient data.
    """
    valid_tickers = []
    
    for ticker in historical_prices.columns:
        # Check if the ticker has enough non-NaN data
        if historical_prices[ticker].count() >= min_required_dates:
            valid_tickers.append(ticker)
    
    return valid_tickers

# --------------------------------------------------
# Get the list of valid tickers based on historical data
# --------------------------------------------------

min_required_dates = len(assets_adj_close_data)  # Set to the number of dates in your historical data
valid_tickers = get_valid_tickers(assets_adj_close_data, min_required_dates)

print(f"Valid Tickers for Backtesting: {valid_tickers}")  # Debugging: Print the valid tickers

# --------------------------------------------------
# Run Simulations
# --------------------------------------------------

# Set number of simulations
num_simulations = 5
history = []  # Store the results of simulations

# Run simulations
for sim in range(num_simulations):
    try:
        # Randomly choose 10 assets from the valid tickers list
        selected_tickers = np.random.choice(valid_tickers, 10, replace=False)
        
        try:
            portfolio_values = backtest.run_backtest(assets_adj_close_data, model_data)
        except ValueError as e:
            print(f"Error: {e}")
        
        # Append the simulation result to history
        history.append(portfolio_values)
    except Exception as e:
        print(f"Simulation {sim+1} failed due to: {e}")
        continue

# --------------------------------------------------
# Calculate and Display Performance Metrics
# --------------------------------------------------

# Calculate the performance metrics (e.g., average final return and Omega ratio) for all simulations
performance_metrics = backtest.calculate_performance_metrics(history)

# Display the performance metrics
print("Performance Metrics from Simulations:")
print(f"Average Final Return (%): {performance_metrics['Average Final Return (%)']:.2f}")
print(f"Average Omega Ratio: {performance_metrics['Average Omega Ratio']:.2f}")

# --------------------------------------------------
# Plot the Average Performance across Simulations
# --------------------------------------------------

# Plot the average portfolio performance across simulations and compare it with the S&P 500
if '^GSPC_AC' in sp500_data.columns:
    backtest.plot_simulation_performance(history, sp500_data['^GSPC_AC'])
else:
    print("The column '^GSPC_AC' is not present in the S&P 500 data.")

# --------------------------------------------------







# --------------------------------------------------

# INITIALIZE THE DYNAMIC BACKTESTING CLASS
#DBT_dynback = DBT()
#DBT_dynback.first_function()
#DBT_dynback.second_function()
#DBT_dynback.third_function()
