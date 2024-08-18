# The central code for PAP-II ! =) 
# EDA FOR ECONOMIC INDICATORS. 
# import libraries.....

import pandas as pd

# Load the Excel file
file_path = '/Users/juan.mugica/Desktop/PAP-II./PAP-ERS/Data/economic_indicators_data.xlsx'
excel_data = pd.ExcelFile(file_path)

# Display sheet names to understand the structure
sheet_names = excel_data.sheet_names
sheet_names
