# Calculating Sharpe Ratio for Long-Only Versus Market-Neutral Strategies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_excel('./excels/IGE.xls')
df.sort_values(by='Date', inplace=True)
dailyret = df.loc[:, 'Adj Close'].pct_change() # Daily returns
excessRet = dailyret - 0.04/252 # Excess returns = Strategy returns - risk free rate (financing cost)
sharpeRatio = np.sqrt(252) * np.mean(excessRet) / np.std(excessRet)
print(sharpeRatio)