import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def calculate_ichimoku(df):
    """Calculate Ichimoku Cloud components"""
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['conversion_line'] = (high_9 + low_9) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low) / 2
    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['base_line'] = (high_26 + low_26) / 2
    
    # Leading Span A (Senkou Span A): (Conversion Line + Base Line) / 2, plotted 26 periods ahead
    df['leading_span_a'] = ((df['conversion_line'] + df['base_line']) / 2).shift(26)
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['leading_span_b'] = ((high_52 + low_52) / 2).shift(26)
        
    # Cloud boundaries - use np.maximum/minimum to avoid alignment issues
    df['cloud_top'] = np.maximum(df['leading_span_a'].values, df['leading_span_b'].values)
    df['cloud_bottom'] = np.minimum(df['leading_span_a'].values, df['leading_span_b'].values)
    df['cloud_thickness'] = df['cloud_top'] - df['cloud_bottom']
    
    return df

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(period).mean()
    return df

def calculate_signals(df):
    """
    Generate trading signals based on enhanced Ichimoku criteria
    1. Lagging confirmation
    2. Thin cloud condition
    3. Span-a-b relationship
    4. Price-cloud relationship
    5. TK cross
    6. Volume condition
    """

    # Lagging confirmation (Last close > 26 periods ago's close)
    df['chikou_long_ok'] = df['Close'] > df['Close'].shift(26)
    df['chikou_short_ok'] = df['Close'] < df['Close'].shift(26)

    # Cloud thickness condition
    df['thin_cloud'] = df['cloud_thickness'] < (0.5 * df['atr'])

    # Span-a-b relationship
    df['span_a_above_b'] = df['leading_span_a'] > df['leading_span_b']
    df['span_a_below_b'] = df['leading_span_a'] < df['leading_span_b']
    
    # Price-cloud relationship
    df['above_cloud'] = df['Close'] > df['cloud_top']
    df['below_cloud'] = df['Close'] < df['cloud_bottom']

    # TK cross
    df['breakout_long'] = (df['conversion_line'] > df['base_line']) & (df['conversion_line'].shift() <= df['base_line'].shift())
    df['breakout_short'] = (df['conversion_line'] < df['base_line']) & (df['conversion_line'].shift() >= df['base_line'].shift())
        
    # Volume condition
    df['adv_20'] = df['Volume'].rolling(20).mean()
    df['volume_ok'] = df['Volume'].values > (1.2 * df['adv_20'].values)
        
    # All conditions ✓
    df['long_signal'] = (
        # df['chikou_long_ok'] 
        # & df['thin_cloud'] 
        # df['span_a_above_b'] 
        df['above_cloud'] 
        & df['breakout_long']
        # & df['volume_ok']
    )
    
    df['short_signal'] = (
        # df['chikou_short_ok'] 
        # & df['thin_cloud'] 
        # df['span_a_below_b'] 
        df['below_cloud']
        & df['breakout_short']
        # & df['volume_ok']
    )
    
    df['signal'] = np.where(df['long_signal'], 1, 
                   np.where(df['short_signal'], -1, 0))
    
    # Shift signals to avoid lookahead bias
    df['signal'] = df['signal'].shift(1)
    
    return df

def backtest_strategy(df, atr_stop_multiplier, atr_target_multiplier):
    """Backtest the strategy with ATR-based stops and targets"""
    df['position'] = 0
    df['entry_price'] = np.nan
    df['stop_loss'] = np.nan
    df['take_profit'] = np.nan
    df['pnl'] = 0.0
    df['equity'] = 100000.0  # Starting capital
    
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    equity = 100000.0
    trades = []
    
    for i in range(1, len(df)):
        if i == len(df) - 1 and position != 0:
            # Close any open position at the end of the backtest
            current_price = df['Close'].iloc[i]
            pnl = (current_price - entry_price) * position * (equity * 0.95 / entry_price)
            equity += pnl
            
            trades.append({
                'entry_date': df.index[i-1],
                'exit_date': df.index[i],
                'direction': 'Long' if position == 1 else 'Short',
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl': pnl,
                'exit_reason': 'End of Backtest'
            })
            
            df.loc[df.index[i], 'pnl'] = pnl
            df.loc[df.index[i], 'equity'] = equity
            position = 0
            break

        current_price = df['Close'].iloc[i]
        current_atr = df['atr'].iloc[i]
        signal = df['signal'].iloc[i]
        
        # Check exit conditions for existing position
        if position != 0:
            exit_triggered = False
            exit_reason = ''
            
            if position == 1:  # Long position
                if current_price <= stop_loss:
                    exit_triggered = True
                    exit_reason = 'Stop Loss'
                    df.loc[df.index[i], 'stop_loss'] = stop_loss
                elif current_price >= take_profit:
                    exit_triggered = True
                    exit_reason = 'Take Profit'
                    df.loc[df.index[i], 'take_profit'] = take_profit
                elif signal == -1:  # Opposite signal
                    exit_triggered = True
                    exit_reason = 'Opposite Signal'
            
            elif position == -1:  # Short position
                if current_price >= stop_loss:
                    exit_triggered = True
                    exit_reason = 'Stop Loss'
                    df.loc[df.index[i], 'stop_loss'] = stop_loss
                elif current_price <= take_profit:
                    exit_triggered = True
                    exit_reason = 'Take Profit'
                    df.loc[df.index[i], 'take_profit'] = take_profit
                elif signal == 1:  # Opposite signal
                    exit_triggered = True
                    exit_reason = 'Opposite Signal'
            
            if exit_triggered:
                pnl = (current_price - entry_price) * position * (equity * 0.95/ entry_price)
                equity += pnl
                
                trades.append({
                    'entry_date': df.index[i-1],
                    'exit_date': df.index[i],
                    'direction': 'Long' if position == 1 else 'Short',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })
                
                df.loc[df.index[i], 'pnl'] = pnl
                position = 0
        
        # Enter new position on signal
        if position == 0 and signal != 0 and not pd.isna(current_atr):
            position = signal
            entry_price = current_price
            
            if position == 1:  # Long
                stop_loss = entry_price - (atr_stop_multiplier * current_atr)
                take_profit = entry_price + (atr_target_multiplier * current_atr)
            else:  # Short
                stop_loss = entry_price + (atr_stop_multiplier * current_atr)
                take_profit = entry_price - (atr_target_multiplier * current_atr)
        
        df.loc[df.index[i], 'position'] = position
        df.loc[df.index[i], 'equity'] = equity
    
    return df, trades

def plot_results(df, trades, ticker):
    """Plot strategy results"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot 1: Price, Ichimoku Cloud, and Signals
    ax1.plot(df.index, df['Close'], label='Close Price', color='lightgray', linewidth=1.5)
    ax1.plot(df.index, df['conversion_line'], label='Conversion Line', color='lightblue', alpha=0.7)
    ax1.plot(df.index, df['base_line'], label='Base Line', color='lightcoral', alpha=0.7)
    
    # Cloud
    ax1.fill_between(df.index, df['leading_span_a'], df['leading_span_b'],
                     where=df['leading_span_a'] >= df['leading_span_b'],
                     color='lightgreen', alpha=0.3, label='Bullish Cloud')
    ax1.fill_between(df.index, df['leading_span_a'], df['leading_span_b'],
                     where=df['leading_span_a'] < df['leading_span_b'],
                     color='lightcoral', alpha=0.3, label='Bearish Cloud')
    
    # Entry signals
    longs = df[df['signal'] == 1]
    shorts = df[df['signal'] == -1]
    ax1.scatter(longs.index, longs['Close'], color='green', marker='^', s=100, 
               label='Long Signal', zorder=5)
    ax1.scatter(shorts.index, shorts['Close'], color='red', marker='v', s=100,
               label='Short Signal', zorder=5)
    ax1.scatter(df.index, df['stop_loss'], color='orange', marker='o', s=50,
               label='Stop Loss', zorder=5)
    ax1.scatter(df.index, df['take_profit'], color='purple', marker='o', s=50,
               label='Take Profit', zorder=5)
    
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{ticker} - Enhanced Ichimoku Cloud Breakout Strategy')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Position and Equity
    ax2.plot(df.index, df['equity'], label='Equity Curve', color='purple', linewidth=2)
    ax2.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Starting Capital')
    ax2.set_ylabel('Equity ($)')
    ax2.set_title('Equity Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cloud Thickness vs ATR Threshold
    ax3.plot(df.index, df['cloud_thickness'], label='Cloud Thickness', color='orange')
    ax3.plot(df.index, 0.5 * df['atr'], label='0.5 × ATR (Threshold)', 
            color='red', linestyle='--')
    ax3.fill_between(df.index, 0, 0.5 * df['atr'], alpha=0.2, color='red',
                     label='Thin Cloud Zone')
    ax3.set_ylabel('Thickness')
    ax3.set_xlabel('Date')
    ax3.set_title('Cloud Thickness Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{ticker}_ichimoku_strategy.png")
    
    return fig

def print_performance_metrics(df, trades):
    """Print strategy performance metrics"""
    print("\n" + "="*70)
    print("STRATEGY PERFORMANCE METRICS")
    print("="*70)
    
    total_return = ((df['equity'].iloc[-1] - df['equity'].iloc[0]) / df['equity'].iloc[0]) * 100
    buy_hold_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
    
    print(f"\nTotal Return: {total_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Final Equity: ${df['equity'].iloc[-1]:,.2f}")
    print(f"Starting Equity: ${df['equity'].iloc[0]:,.2f}")
    
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        print(f"\nTotal Trades: {len(trades)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Win: ${avg_win:,.2f}")
        print(f"Average Loss: ${avg_loss:,.2f}")
        
        if avg_loss != 0:
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum())
            print(f"Profit Factor: {profit_factor:.2f}")
        
        print("\n" + "-"*70)
        print("RECENT TRADES:")
        print("-"*70)
        print(trades_df.to_string(index=False))
    else:
        print("\nNo trades executed during the backtest period.")
    
    print("\n" + "="*70)

# Main execution
if __name__ == "__main__":
    # Download SPY data
    ticker = "AGG"
    end_date = datetime(2026,1,1)
    start_date = end_date - timedelta(days=365*3)  # 10 years of data
    
    print(f"Downloading {ticker} data from {start_date.date()} to {end_date.date()}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Reset index to make sure we have clean DatetimeIndex
    df = df.copy()
    
    # Calculate indicators
    print("Calculating Ichimoku indicators...")
    df = calculate_ichimoku(df)
    df = calculate_atr(df)
    
    # Generate signals
    print("Generating trading signals...")
    df = calculate_signals(df)
    
    # Run backtest
    print("Running backtest...")
    df, trades = backtest_strategy(df, atr_stop_multiplier=8.0, atr_target_multiplier=16.0)
    
    # Print performance
    print_performance_metrics(df, trades)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(df, trades, ticker)