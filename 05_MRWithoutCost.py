import numpy as np
import pandas as pd

df = pd.read_excel('./excels/SPY.xlsx')
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)
df['returns'] = df['Close'].pct_change()

# Parameters
lookback = 20
long_entry_threshold = -0.5  # zscore threshold for going long
short_entry_threshold = 0.5  # zscore threshold for going short

# Calculate rolling mean, std dev, and zscore for returns
df['roll_mean'] = df['returns'].rolling(window=lookback).mean()
df['roll_std'] = df['returns'].rolling(window=lookback).std()
df['zscore'] = (df['returns'] - df['roll_mean']) / df['roll_std']

# Generate discrete long and short entry signals
df['long_entry'] = 0
df['short_entry'] = 0
df.loc[df['zscore'] < long_entry_threshold, 'long_entry'] = 1      # Long entry signal (+1)
df.loc[df['zscore'] > short_entry_threshold, 'short_entry'] = 1    # Short entry signal (+1)

# Generate exit signals that clear positions
df['long_exit'] = 0
df['short_exit'] = 0
df.loc[df['zscore'] >= 0, 'long_exit'] = 1     # Clear all longs when zscore >= 0
df.loc[df['zscore'] <= 0, 'short_exit'] = 1    # Clear all shorts when zscore <= 0

# Cumulative sums of long and short entries (incremental additions)
df['long_pos'] = df['long_entry'].cumsum()
df['short_pos'] = df['short_entry'].cumsum()

# Apply resets to long positions on long exits
long_exit_indices = df.index[df['long_exit'] == 1]
df.loc[long_exit_indices, 'long_pos'] = 0

# To propagate the reset effect forward in time for longs:
for i in range(1, len(df)):
    if df['long_exit'].iloc[i] == 1:
        df.at[df.index[i], 'long_pos'] = 0
    else:
        # If no exit today, position is previous day's position plus today's entry
        if i > 0:
            df.at[df.index[i], 'long_pos'] = df.at[df.index[i-1], 'long_pos'] + df['long_entry'].iloc[i]

# Apply resets to short positions on short exits
short_exit_indices = df.index[df['short_exit'] == 1]
df.loc[short_exit_indices, 'short_pos'] = 0

# Propagate the reset effect forward in time for shorts:
for i in range(1, len(df)):
    if df['short_exit'].iloc[i] == 1:
        df.at[df.index[i], 'short_pos'] = 0
    else:
        if i > 0:
            df.at[df.index[i], 'short_pos'] = df.at[df.index[i-1], 'short_pos'] + df['short_entry'].iloc[i]

# Net position = long positions - short positions
df['net_position'] = df['long_pos'] - df['short_pos']

# Lag position by one day to avoid lookahead bias
df['net_position_lagged'] = df['net_position'].shift()

# Calculate strategy daily returns
df['strategy_return'] = df['net_position_lagged'] * df['returns']

# Sharpe ratio
annual_factor = np.sqrt(252)
sharpe = annual_factor * df['strategy_return'].mean() / df['strategy_return'].std()
print("Sharpe Ratio:", sharpe)

# Check the signals and positions
print(df[['returns', 'zscore', 'long_entry', 'short_entry', 'long_exit', 'short_exit',
          'long_pos', 'short_pos', 'net_position', 'net_position_lagged', 'strategy_return']].tail(20))
