
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
import matplotlib.pyplot as plt
from utils import SP500ComparisonEDA
from utils import EDAAnalysis as EDA
from utils import FredIndicatorFetcher as FIF
from utils import HistoricalDataDownloader as HDD

# --------------------------------------------------

# Download historic index and companies of USA MARKET
if __name__ == "__main__":
    #companies = ['^GSPC']
    companies = a = ['^GSPC', 'A', 'AAL', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE',
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
    downloader = HDD(companies)
    downloader.download_data(start_date="2000-01-01")
    downloader.save_data(filepath="Data/historical_data.xlsx")

# --------------------------------------------------

# Instantiate the class with the correct file path and parameters
EDA = EDA(
    file_path="Data/economic_indicators_data.xlsx", 
    date_column='Date',
    columns_to_analyze=['CLI', 'BCI', 'GDP', 'CCI'],
    sheet_name='data'
)

EDA.perform_eda()

# --------------------------------------------------

# Load the data (you probably already have this in your code)
economic_df = pd.read_excel("Data/economic_indicators_data.xlsx", sheet_name="data")
historical_df = pd.read_excel("Data/historical_data.xlsx", sheet_name="Historical Data")

# --------------------------------------------------

# Instantiate the comparison and EDA class
comparator_eda = SP500ComparisonEDA(sp500_data=historical_df, indicators_data=economic_df)

# Plot comparison of indicators vs S&P 500
comparator_eda.plot_comparison()

# Plot histograms of indicators vs S&P 500
comparator_eda.plot_histograms()

# Plot boxplots of indicators vs S&P 500
comparator_eda.plot_boxplots()

# --------------------------------------------------

# EXTRA

# api_key = '2df41cfada1473ef26fa8dede4c9bef5 '
# fetcher = FIF(api_key)
# data_tlei = fetcher.fetch_indicator('TLEI')
# data_cpi = fetcher.fetch_indicator('CPI')
# data_gdp = fetcher.fetch_indicator('GDP')
# data_cci = fetcher.fetch_indicator('CCI')
# data_cei = fetcher.fetch_indicator('CEI')
# fetcher.save_to_excel({'TLEI': data_tlei, 'CPI': data_cpi, 
#                        'GDP': data_gdp, 'CCI': data_cci,
#                        'CEI': data_cei}, 'economic_indicatorsv2.xlsx')
