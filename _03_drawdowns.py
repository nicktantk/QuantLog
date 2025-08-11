import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculateMaxDD(cumret):
    # Calculation of maximum drawdown and maximum drawdown duration based on 
    # cumulative COMPOUNDED returns. cumret must be a compounded cumulative return.
    # i is the index of the day with maxDD.

    highwatermark = np.zeros(cumret.shape)
    drawdown = np.zeros(cumret.shape)
    drawdownduration = np.zeros(cumret.shape)
    for t in np.arange(1, cumret.shape[0]):
        highwatermark[t] = np.maximum(highwatermark[t-1], cumret[t])
        drawdown[t] = (1+cumret[t])/(1+highwatermark[t]) - 1
        if drawdown[t] == 0:
            drawdownduration[t] = 0
        else:
            drawdownduration[t] = drawdownduration[t-1] + 1

    maxDD, i = np.min(drawdown), np.argmin(drawdown)
    maxDDD = np.max(drawdownduration)
    return maxDD, maxDDD, i

# Calculating daily net returns
df = pd.read_excel('./excels/SPY.xlsx')
df.sort_values(by='Date', inplace=True)
netRet = df['Close'].pct_change()

# Calculating the maxDD and maxDDD
cumret = np.cumprod(1+netRet) - 1
plt.plot(cumret)
plt.show()
maxDD, maxDDD, startDDDay = calculateMaxDD(cumret.values)
print("maxDD:", maxDD)
print("maxDDD:", maxDDD)
print("Start of DD:", startDDDay)

        