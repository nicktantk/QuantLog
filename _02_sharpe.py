# Sharpe ratio of a long-short market-neutral strategy

import yfinance as yf
import pandas as pd
import numpy as np

df_ige = pd.read_excel('./excels/IGE.xlsx')
df_spy = pd.read_excel('./excels/SPY.xlsx')
df = pd.merge(df_ige, df_spy, on='Date', suffixes=('_IGE', '_SPY'))
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
dailyret = df[['Close_IGE', 'Close_SPY']].pct_change()
dailyret.rename(columns={"Close_IGE": "IGE", "Close_SPY": "SPY"}, inplace=True)
netRet = (dailyret['IGE'] - dailyret['SPY']) / 2
sharpeRatio = np.sqrt(252) * np.mean(netRet) / np.std(netRet)
print(sharpeRatio)