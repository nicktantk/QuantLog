# Pair trading of GLD and GDX
# This illustrates how to separate the data into a training set and test set
# Involves optimising of paraters on the training set and looking at its 
# effect on a test set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df1 = pd.read_excel('./excels/GLD.xlsx')
df2 = pd.read_excel('./excels/GDX.xlsx')
df = pd.merge(df1, df2, on='Date', suffixes=('_GLD', '_GDX'))
df.sort_values(by='Date', inplace=True)
trainset = np.arange(0, 900) # np array
testset = np.arange(trainset.shape[0], df.shape[0]) # np array

model = sm.OLS(df['Close_GLD'].iloc[trainset],
             df['Close_GDX'].iloc[trainset]) # ordinary least squares 
results = model.fit()
hedgeRatio = results.params # gradient of GLD against GDX
print("hedgeRatio:", hedgeRatio.iloc[0])

spread = df['Close_GLD'] - hedgeRatio.iloc[0]*df['Close_GDX']

spreadMean = np.mean(spread.iloc[trainset])
print("spreadMean: ", spreadMean)
spreadStd = np.std(spread.iloc[trainset])
print("spreadStd:", spreadStd)
df['zscore'] = (spread-spreadMean)/spreadStd
df['positions_GLD_Long'] = 0
df['positions_GDX_Long'] = 0
df['positions_GLD_Short'] = 0
df['positions_GDX_Short'] = 0
df.loc[df.zscore >= 0.4, ('positions_GLD_Short', 'positions_GDX_Short')] = [-1, 1]  # Short spread
df.loc[df.zscore <= -0.4, ('positions_GLD_Long', 'positions_GDX_Long')] = [1, -1]  # Buy spread
df.loc[df.zscore < 0, ('positions_GLD_Short', 'positions_GDX_Short')] = 0  # Exit short spread
df.loc[df.zscore > 0, ('positions_GLD_Long', 'positions_GDX_Long')] = 0  # Exit long spread
df.ffill(inplace=True)  # ensure existing positions are carried forward unless there is an exit signal

positions_Long = df.loc[:, ('positions_GLD_Long', 'positions_GDX_Long')]
positions_Short = df.loc[:, ('positions_GLD_Short', 'positions_GDX_Short')]
positions = np.array(positions_Long) + np.array(positions_Short)
positions = pd.DataFrame(positions)
dailyret = df.loc[:, ('Close_GLD', 'Close_GDX')].pct_change()
pnl = (np.array(positions.shift()) * np.array(dailyret)).sum(axis=1)

sharpeTrainset = np.sqrt(252) * np.mean(pnl[trainset[1:]]) / np.std(pnl[trainset[1:]])
print("sharpeTrainset:", sharpeTrainset)
sharpeTestset = np.sqrt(252) * np.mean(pnl[testset]) / np.std(pnl[testset])
print("sharpeTestset:", sharpeTestset)

print(df[['positions_GLD_Long', 'positions_GDX_Long',
          'positions_GLD_Short', 'positions_GDX_Short']].tail(100))
plt.plot(np.cumsum(pnl[testset]))
plt.show()
