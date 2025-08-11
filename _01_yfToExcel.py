# Using python to retrieve Yahoo! Finance Data

import yfinance as yf

data = yf.download('GLD', start='2015-06-30', end='2025-06-30')
data.to_excel('./excels/GLD.xlsx')