
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
from utils import EDA_comparison
from utils import EDA
from utils import HistoricalDataDownloader as HDD

# --------------------------------------------------

# 
HDD_sp500 = HDD(
    tickers=['^GSPC'],
    start_date="2000-01-01")
HDD_sp500.download_data()
HDD_sp500.save_data(filepath="Data/sp500_data.xlsx")

# ------------------------------

# 
economic_indicators_data = pd.read_excel("Data/economic_indicators_data.xlsx", sheet_name="data")
sp500_data = pd.read_excel("Data/sp500_data.xlsx", sheet_name="data")

# --------------------------------------------------

# 
EDA_economic_indicators = EDA(
    file_path="Data/economic_indicators_data.xlsx", 
    date_column='Date',
    columns_to_analyze=['CLI', 'BCI', 'GDP', 'CCI'],
    sheet_name='data'
)

# ------------------------------

# 
EDA_economic_indicators.perform_EDA()

# --------------------------------------------------

# 
EDA_sp500 = EDA(
    file_path="Data/sp500_data.xlsx", 
    date_column='Date',
    columns_to_analyze=['^GSPC CLOSE'],
    sheet_name='data'
)

# ------------------------------

# 
EDA_sp500.perform_EDA()

# --------------------------------------------------

# 
EDA_comparison_economic_indicators_sp500 = EDA_comparison(
    sp500_data=sp500_data,
    economic_indicators_data=economic_indicators_data
)

# ------------------------------

# 
EDA_comparison_economic_indicators_sp500.perform_EDA_comparison()










# --------------------------------------------------

# EXTRA

# from utils import FredIndicatorFetcher as FIF
# api_key = '2df41cfada1473ef26fa8dede4c9bef5 '
# fetcher = FIF(api_key)
# data_tlei = fetcher.fetch_indicator('TLEI')
# data_cpi = fetcher.fetch_indicator('CPI')
# data_gdp = fetcher.fetch_indicator('GDP')
# data_cci = fetcher.fetch_indicator('CCI')
# data_cei = fetcher.fetch_indicator('CEI')
# fetcher.save_to_excel({'TLEI': data_tlei,
#                        'CPI': data_cpi, 
#                        'GDP': data_gdp,
#                        'CCI': data_cci,
#                        'CEI': data_cei}, 'economic_indicatorsv2.xlsx')

# --------------------------------------------------

# NOTAS

a = ['^GSPC', 'A', 'AAL', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE',
     'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ',
     'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME',
     'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD',
     'APH', 'APTV', 'ARE', 'ARM', 'ASML', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK',
     'AXON', 'AXP', 'AZN', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY',
     'BDX', 'BEN', 'BF.B', 'BG', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLDR',
     'BLK', 'BMY', 'BR', 'BRK.B', 'BRO', 'BSX', 'BWA', 'BX', 'BXP', 'C', 'CAG',
     'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCEP', 'CCI', 'CCL', 'CDNS',
     'CDW', 'CE', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF',
     'CL', 'CLX', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF',
     'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM',
     'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CTVA',
     'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DASH', 'DAY', 'DD', 'DDOG', 'DE', 'DECK',
     'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOV',
     'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EA', 'EBAY',
     'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'ENPH',
     'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'ETSY',
     'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCX',
     'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FMC', 'FOX',
     'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GDDY', 'GE', 'GEHC', 'GEN',
     'GEV', 'GFS', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL',
     'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD',
     'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC',
     'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF',
     'ILMN', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM',
     'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ',
     'JNPR', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KKR', 'KLAC',
     'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX',
     'LIN', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS',
     'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK',
     'MCO', 'MDB', 'MDLZ', 'MDT', 'MELI', 'MET', 'META', 'MGM', 'MHK', 'MKC',
     'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR',
     'MRK', 'MRNA', 'MRO', 'MRVL', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH',
     'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE',
     'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS',
     'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS',
     'OXY', 'PANW', 'PARA', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PDD', 'PEG', 'PEP',
     'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PM', 'PNC', 'PNR',
     'PNW', 'PODD', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR',
     'PYPL', 'QCOM', 'QRVO', 'RCL', 'REG', 'REGN', 'RF', 'RJF', 'RL', 'RMD',
     'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW',
     'SHW', 'SJM', 'SLB', 'SMCI', 'SNA', 'SNPS', 'SO', 'SOLV', 'SPG', 'SPGI',
     'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SW', 'SWK', 'SWKS', 'SYF',
     'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TEAM', 'TECH', 'TEL', 'TER',
     'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW',
     'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTD', 'TTWO', 'TXN', 'TXT', 'TYL',
     'UAL', 'UBER', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB',
     'V', 'VICI', 'VLO', 'VLTO', 'VMC', 'VRSK', 'VRSN', 'VRTX', 'VST', 'VTR',
     'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDAY', 'WDC', 'WEC', 'WELL',
     'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WST', 'WTW', 'WY', 'WYNN', 'XEL',
     'XOM', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZS', 'ZTS']
