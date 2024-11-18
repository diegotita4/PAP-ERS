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



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_sp500_vs_predictions(sp500_data, predictions):
    """
    Grafica el comportamiento del S&P 500 junto con las predicciones del modelo.
    
    Args:
        sp500_data (DataFrame): Datos históricos del S&P 500 con índice como fecha y columna ['^GSPC_AC'].
        predictions (DataFrame): Predicciones del modelo con índice como fecha y columna ['Prediction'] (-1, 0, 1).
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Graficar el SP500
    axs[0].plot(sp500_data.index, sp500_data['^GSPC_AC'], color='blue', label='S&P 500')
    axs[0].set_title("S&P 500 Historical Data")
    axs[0].set_ylabel("S&P 500 Value")
    axs[0].legend()
    axs[0].grid(True)

    # Graficar las predicciones del modelo
    axs[1].scatter(predictions.index, predictions['Prediction'], c=predictions['Prediction'], 
                   cmap='coolwarm', label='Model Predictions')
    axs[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.8, label='Neutral Prediction (0)')
    axs[1].set_title("Model Predictions")
    axs[1].set_ylabel("Prediction (-1, 0, 1)")
    axs[1].set_xlabel("Date")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# def plot_boxplot_with_benchmark(data, benchmark_mean):
#     """
#     Genera un boxplot para las simulaciones con una línea promedio del benchmark.

#     Args:
#         data (DataFrame): DataFrame con una columna 'Semi_Annual_Return' que contiene los rendimientos.
#         benchmark_mean (float): Promedio del benchmark (por ejemplo, retorno medio del S&P 500).
#     """
#     plt.figure(figsize=(8, 6))
#     sns.boxplot(x=data['Semi_Annual_Return'], color='skyblue', width=0.6)
#     plt.axvline(x=benchmark_mean, color='red', linestyle='--', label=f'Benchmark Mean: {benchmark_mean:.2%}')
#     plt.title("Boxplot of Semi-Annual Returns with Benchmark Mean")
#     plt.xlabel("Semi-Annual Return")
#     plt.legend()
#     plt.grid(axis='x', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.show()

# Generar 'predictions' desde model_data
predictions = model_data[['Y']].rename(columns={'Y': 'Prediction'})

# Ejecutar las funciones de las gráficas
plot_sp500_vs_predictions(sp500_data, predictions)
# plot_boxplot_with_benchmark(clean_performance_df, sp500_annual_return)

def plot_sp500_vs_predictions_adjusted(sp500_data, predictions):
    """
    Grafica el comportamiento del S&P 500 junto con las predicciones del modelo, incluyendo líneas horizontales para los ciclos.
    
    Args:
        sp500_data (DataFrame): Datos históricos del S&P 500 con índice como fecha y columna ['^GSPC_AC'].
        predictions (DataFrame): Predicciones del modelo con índice como fecha y columna ['Prediction'] (-1, 0, 1).
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Graficar el SP500
    axs[0].plot(sp500_data.index, sp500_data['^GSPC_AC'], color='blue', label='S&P 500')
    axs[0].set_title("S&P 500 Historical Data")
    axs[0].set_ylabel("S&P 500 Value")
    axs[0].legend()
    axs[0].grid(True)

    # Graficar las predicciones del modelo con líneas horizontales
    axs[1].plot(predictions.index, predictions['Prediction'], color='black', label='Model Predictions', alpha=0.8)
    axs[1].axhline(y=-1, color='blue', linestyle='--', linewidth=1, label='Bearish Cycle (-1)')
    axs[1].axhline(y=0, color='gray', linestyle='--', linewidth=1, label='Neutral Cycle (0)')
    axs[1].axhline(y=1, color='red', linestyle='--', linewidth=1, label='Bullish Cycle (1)')
    axs[1].set_title("Model Predictions with Cycles")
    axs[1].set_ylabel("Prediction (-1, 0, 1)")
    axs[1].set_xlabel("Date")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

plot_sp500_vs_predictions_adjusted(sp500_data, predictions)


def classify_sp500(sp500_data, threshold=0.02):
    """
    Clasifica el rendimiento del S&P 500 en alto, neutral y bajo.
    
    Args:
        sp500_data (DataFrame): Datos históricos del S&P 500 con índice como fecha y columna ['^GSPC_AC'].
        threshold (float): Umbral para definir categorías (por ejemplo, 0.02 para 2%).
        
    Returns:
        DataFrame: DataFrame original con una nueva columna ['SP500_Category'].
    """
    # Calcular rendimiento mensual
    sp500_data['SP500_Return'] = sp500_data['^GSPC_AC'].pct_change()

    # Clasificar en tres categorías
    sp500_data['SP500_Category'] = sp500_data['SP500_Return'].apply(
        lambda x: 1 if x > threshold else (-1 if x < -threshold else 0)
    )
    return sp500_data
def merge_model_with_sp500(sp500_data, predictions):
    """
    Combina las predicciones del modelo con las categorías del S&P 500.
    
    Args:
        sp500_data (DataFrame): Datos del S&P 500 con la columna ['SP500_Category'].
        predictions (DataFrame): Predicciones del modelo con índice como fecha y columna ['Prediction'].
        
    Returns:
        DataFrame: DataFrame combinado.
    """
    # Asegurarse de que los índices coincidan
    combined_data = sp500_data[['SP500_Category']].merge(predictions, left_index=True, right_index=True, how='inner')
    return combined_data

def plot_predictions_vs_sp500(sp500_data, combined_data):
    """
    Grafica las predicciones del modelo junto con la clasificación del S&P 500.
    
    Args:
        sp500_data (DataFrame): Datos del S&P 500 con rendimiento y clasificación.
        combined_data (DataFrame): DataFrame combinado con predicciones y categorías.
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # # Graficar las predicciones del modelo
    # axs[0].plot(combined_data.index, combined_data['Prediction'], color='black', label='Model Predictions', alpha=0.8)
    # axs[0].axhline(y=-1, color='blue', linestyle='--', linewidth=1, label='Bearish Prediction (-1)')
    # axs[0].axhline(y=0, color='gray', linestyle='--', linewidth=1, label='Neutral Prediction (0)')
    # axs[0].axhline(y=1, color='red', linestyle='--', linewidth=1, label='Bullish Prediction (1)')
    # axs[0].set_ylabel("Model Predictions (-1, 0, 1)")
    # axs[0].legend()
    # axs[0].grid(True)

    # Graficar las predicciones del modelo
    axs[0].scatter(predictions.index, predictions['Prediction'], c=predictions['Prediction'], 
                   cmap='coolwarm', label='Model Predictions')
    axs[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8, label='Neutral Prediction (0)')
    axs[0].set_title("Model Predictions")
    axs[0].set_ylabel("Prediction (-1, 0, 1)")
    axs[0].set_xlabel("Date")
    axs[0].legend()
    axs[0].grid(True)

    # Graficar la clasificación del S&P 500
    axs[1].scatter(combined_data.index, combined_data['SP500_Category'], c=combined_data['SP500_Category'], 
                   cmap='coolwarm', label='S&P 500 Classification')
    axs[1].axhline(y=-1, color='blue', linestyle='--', linewidth=1, label='Bajo (-1)')
    axs[1].axhline(y=0, color='gray', linestyle='--', linewidth=1, label='Neutral (0)')
    axs[1].axhline(y=1, color='red', linestyle='--', linewidth=1, label='Alto (1)')
    axs[1].set_ylabel("SP500 Categories (-1, 0, 1)")
    axs[1].set_xlabel("Date")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

classify_sp500_data = classify_sp500(sp500_data)
combined_data = merge_model_with_sp500(classify_sp500_data, predictions)
plot_predictions_vs_sp500(sp500_data, combined_data)


