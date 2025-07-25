# Sharpe ratio of a long-short market-neutral strategy

import yfinance as yf
import pandas as pd
import numpy as np

# data = yf.download('SPY', start='2015-01-01', end='2025-06-30')
# data.to_excel('./excels/SPY.xlsx', engine='openpyxl')

# --- above code to save SPY.xlsx file ---

df_ige = pd.read_excel('./excels/IGE.xlsx')
df_spy = pd.read_excel('./excels/SPY.xlsx')
df = pd.merge(df_ige, df_spy, on='Date', suffixes=('_IGE', '_SPY'))
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)
dailyret = df[['Close_IGE', 'Close_SPY']].pct_change()
dailyret.rename(columns={"Close_IGE": "IGE", "Close_SPY": "SPY"}, inplace=True)
netRet = (dailyret['IGE'] - dailyret['SPY']) / 2
sharpeRatio = np.sqrt(252) * np.mean(netRet) / np.std(netRet)
print(sharpeRatio)