# Describe utils.py 
# Import libraries.....
from fredapi import Fred
import pandas as pd
from datetime import datetime
import openpyxl
import os

# Class to handle FRED API requests
class FredIndicatorFetcher:
    """
        FredIndicatorFetcher is a class for interacting with the FRED API to fetch economic 
        indicator data and save it to an Excel file.

        Attributes:
            api_key (str): The API key for accessing the FRED API.
            fred (Fred): An instance of the Fred class initialized with the provided API key.
            indicators (dict): A dictionary mapping human-readable indicator names to their 
                            corresponding FRED series IDs.
        """
    def __init__(self, api_key):
        self.api_key = api_key
        self.fred = Fred(api_key=self.api_key)
        self.indicators = {
            "TLEI": "M16005USM358SNBR",  # Composite Index of Three Lagging Indicators, Amplitude-Adjusted, Weighted for United States (Monthly)
            #"LEI": "USSLIND",  # Leading Economic Index for United States (Monthly)
            "CPI": "CPIAUCSL",  # Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
            "GDP":  "USALORSGPNOSTSAM",  # Gross Domestic Product for United states (Monthly)
            "CCI": "CSCICP03USM665S",  # Consumer Confidence Index (Monthly)
            "CEI": "USPHCI" # Coincident Economic Activity Index for the United States  (Monthly)
            #"BCI": "USABCI",  # Proxy for Business Confidence Indicator (Monthly)
        }

    def fetch_indicator(self, indicator_key, start_date='1960-01-31', end_date=None):
        if indicator_key not in self.indicators:
            raise ValueError(f"Indicator '{indicator_key}' not found in the predefined indicators list.")
        
        end_date = end_date if end_date else datetime.today().strftime('%Y-%m-%d')
        series_id = self.indicators[indicator_key]
        data = self.fred.get_series(series_id, start_date, end_date)
        return data

    def save_to_excel(self, data_dict, file_name):
        # Define the directory where the file will be saved
        save_directory = 'PAP-ERS/Data/'
        # Ensure the directory exists; if not, create it
        os.makedirs(save_directory, exist_ok=True)
        # Combine the directory and file name to get the full path
        file_path = os.path.join(save_directory, file_name)
        # Initialize an empty DataFrame to combine all indicators
        combined_df = pd.DataFrame()
        # Iterate over each indicator and its data
        for indicator, data in data_dict.items():
            # Convert the Series to a DataFrame
            df = pd.DataFrame(data, columns=[indicator])
            # Combine the data by joining on the index (date)
            if combined_df.empty:
                combined_df = df
            else:
                combined_df = combined_df.join(df, how='outer')
        
        # Save the combined DataFrame to a single sheet in the Excel file
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            combined_df.to_excel(writer, sheet_name='Combined_Indicators')


# Example usage:
# api_key = '2df41cfada1473ef26fa8dede4c9bef5 '
# fetcher = FredIndicatorFetcher(api_key)
# data_cli = fetcher.fetch_indicator('CLI')
# data_lei = fetcher.fetch_indicator('LEI')
# fetcher.save_to_excel({'CLI': data_cli, 'LEI': data_lei}, 'economic_indicators.xlsx')






