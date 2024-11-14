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
# Cargar resultados de backtesting del portafolio
results_df = pd.read_excel("backtest_results_historicaldata.xlsx")

def calculate_portfolio_metrics(row):
    """Calcula métricas para una trayectoria individual del portafolio."""
    # Filtrar valores no nulos y ceros
    values = row[row != 0].dropna().values
    
    if len(values) < 2:
        return None
    
    initial_value = values[0]
    final_value = values[-1]
    
    # Verificar valores válidos
    if initial_value <= 0 or final_value <= 0:
        return None
        
    # Calcular retornos
    total_return = (final_value / initial_value) - 1
    num_periods = len(values) - 1
    semi_annual_return = (1 + total_return) ** (1 / num_periods) - 1
    
    return {
        'Initial_Value': initial_value,
        'Final_Value': final_value,
        'Total_Return': total_return,
        'Semi_Annual_Return': semi_annual_return,
        'Periods': num_periods
    }

# Procesar cada escenario
scenarios = []
for i, row in results_df.iterrows():
    metrics = calculate_portfolio_metrics(row)
    if metrics is not None:
        metrics['Scenario'] = i
        scenarios.append(metrics)

performance_df = pd.DataFrame(scenarios)

# Remover outliers usando el método IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Limpiar outliers de retornos semestrales
clean_performance_df = remove_outliers(performance_df, 'Semi_Annual_Return')

# Parámetros
risk_free_rate = 0.1050
target_return = 0.50

# Calcular métricas con datos limpios
mean_semi_annual_return = clean_performance_df['Semi_Annual_Return'].mean()
median_semi_annual_return = clean_performance_df['Semi_Annual_Return'].median()
std_semi_annual_return = clean_performance_df['Semi_Annual_Return'].std()

# Sharpe Ratio (ajustado para frecuencia semestral)
annualized_sharpe_ratio = (mean_semi_annual_return - risk_free_rate) / std_semi_annual_return * np.sqrt(2)

# Cargar datos del S&P 500
sp500_data = pd.read_excel("Data/sp500_data.xlsx")
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
sp500_data.set_index('Date', inplace=True)

# Calcular retorno anual del S&P 500
first_sp500 = sp500_data['^GSPC_AC'].iloc[0]
last_sp500 = sp500_data['^GSPC_AC'].iloc[-1]
total_years = (sp500_data.index[-1] - sp500_data.index[0]).days / 365.25
sp500_annual_return = ((last_sp500 / first_sp500) ** (1 / total_years)) - 1
sp500_annual_std = sp500_data['^GSPC_AC'].pct_change().std() * (252 ** 0.5)

# Calcular beta y alpha de Jensen (ajustado para frecuencia semestral)
sp500_returns = sp500_data['^GSPC_AC'].pct_change().dropna()
portfolio_returns = clean_performance_df['Semi_Annual_Return']

# Modificar esta parte para manejar los índices correctamente
if len(portfolio_returns) > 0 and len(sp500_returns) > 0:
    # Convertir los retornos del portafolio a Series con índice numérico
    portfolio_returns = pd.Series(portfolio_returns.values)
    # Resamplear los retornos del S&P 500 a frecuencia semestral
    semi_annual_sp500_returns = sp500_returns.resample('6ME').mean()
    # Asegurar que tengamos la misma cantidad de datos
    min_len = min(len(portfolio_returns), len(semi_annual_sp500_returns))
    portfolio_returns = portfolio_returns[:min_len]
    semi_annual_sp500_returns = semi_annual_sp500_returns[:min_len]
    
    beta_portfolio = portfolio_returns.cov(semi_annual_sp500_returns) / semi_annual_sp500_returns.var()
    alpha_jensen_portfolio = mean_semi_annual_return - (risk_free_rate + beta_portfolio * (sp500_annual_return - risk_free_rate))
else:
    beta_portfolio = np.nan
    alpha_jensen_portfolio = np.nan

# Omega Ratio
excess_returns = clean_performance_df['Semi_Annual_Return'] - target_return
positive_returns = excess_returns[excess_returns > 0].sum()
negative_returns = abs(excess_returns[excess_returns < 0].sum())
omega_ratio = positive_returns / negative_returns if negative_returns != 0 else np.nan

# Mostrar resultados
print("\nMetrics for S&P 500:")
print(f"  - Annual Return: {sp500_annual_return * 100:.2f}%")
print(f"  - Volatility: {sp500_annual_std * 100:.2f}%")

print("\nGeneral Metrics for Portfolio (all simulations):")
print(f"  - Average Semi-Annual Return: {mean_semi_annual_return * 100:.2f}%")
print(f"  - Median Semi-Annual Return: {median_semi_annual_return * 100:.2f}%")
print(f"  - Volatility (Semi-Annual): {std_semi_annual_return * 100:.2f}%")
print(f"  - Annualized Sharpe Ratio: {annualized_sharpe_ratio:.2f}")
print(f"  - Beta: {beta_portfolio:.2f}")
print(f"  - Jensen's Alpha: {alpha_jensen_portfolio:.4f}")
print(f"  - Omega Ratio: {omega_ratio:.2f}")

# Estadísticas de valores finales (sin outliers), en millones
clean_performance_df.loc[:, 'Final_Value_Millions'] = clean_performance_df['Final_Value'] / 1_000_000
clean_final_values = remove_outliers(clean_performance_df, 'Final_Value_Millions')
print("\nDistribution of Final Values (without outliers, in millions):")
print(f"  - Average: {clean_final_values['Final_Value_Millions'].mean():,.2f} M")
print(f"  - Median: {clean_final_values['Final_Value_Millions'].median():,.2f} M")
print(f"  - Min: {clean_final_values['Final_Value_Millions'].min():,.2f} M")
print(f"  - Max: {clean_final_values['Final_Value_Millions'].max():,.2f} M")

# Estadísticas adicionales
print("\nPortfolio Statistics:")
print(f"  - Number of valid scenarios: {len(clean_performance_df)}")
print(f"  - Percentage of winning scenarios: {(clean_performance_df['Semi_Annual_Return'] > target_return).mean() * 100:.2f}%")

# Obtener los 5 mejores y 5 peores escenarios basados en el rendimiento semestral
top_5_scenarios = clean_performance_df.nlargest(5, 'Semi_Annual_Return')
bottom_5_scenarios = clean_performance_df.nsmallest(5, 'Semi_Annual_Return')

# Mostrar los mejores y peores escenarios
print("\nTop 5 Best Scenarios:")
print(top_5_scenarios[['Scenario', 'Initial_Value', 'Final_Value_Millions', 'Total_Return', 'Semi_Annual_Return', 'Periods']])

print("\nBottom 5 Worst Scenarios:")
print(bottom_5_scenarios[['Scenario', 'Initial_Value', 'Final_Value_Millions', 'Total_Return', 'Semi_Annual_Return', 'Periods']])
