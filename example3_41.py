# Calculating Sharpe Ratio for Buy-and-Hold Strategies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import yfinance as yf

# data = yf.download('QQQ', start='2015-06-30', end='2025-06-30')
# data.to_excel('./excels/QQQ.xlsx', engine='openpyxl')


df = pd.read_excel('./excels/QQQ.xlsx')
df.sort_values(by='Date', inplace=True)
dailyret = df.loc[:, 'Close'].pct_change() # Daily returns
excessRet = dailyret - 0.04/252 # Excess returns = Strategy returns - risk free rate (financing cost)
sharpeRatio = np.sqrt(252) * np.mean(excessRet) / np.std(excessRet)
print(sharpeRatio)
