from fredapi import Fred
import pandas as pd
from datetime import datetime
import openpyxl


# Your FRED API key
api_key = '2df41cfada1473ef26fa8dede4c9bef5 '  # Replace with your FRED API key

# Initialize the Fred API with your key
fred = Fred(api_key=api_key)

# Define the FRED series IDs for the indicators
indicators = {
    #"CLI": "USARECDM",  # Proxy for CLI
    #"BCI": "USABCI",  # Proxy for Business Confidence Indicator
    "GDP": "GDP",  # Gross Domestic Product
    "CCI": "CSCICP03USM665S"  # Consumer Confidence Index
}
# Specify the start date
start_date = '1960-01-31'

# Fetch the data from FRED
data_frames = []
for indicator, series_id in indicators.items():
    data = fred.get_series(series_id, observation_start=start_date)
    data = data.rename(indicator)
    data_frames.append(data)

# Merge all data frames on Date
merged_data = pd.concat(data_frames, axis=1)

# Convert the index to a DateTime index (if not already)
merged_data.index = pd.to_datetime(merged_data.index)

# Resample the data monthly (if necessary)
merged_data = merged_data.resample('M').last()

# Define the output file path
output_file = "PAP-ERS/Data/economic_indicators_dataAPI's.xlsx"

# Export to Excel
merged_data.to_excel(output_file, sheet_name='data')

print(f"Data has been successfully saved to {output_file}")
