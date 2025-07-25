import yfinance as yf

def test_yfinance():
    for symbol in ['AAPL', 'MSFT', 'BTC-USD']:
        print(">>", symbol, end='...\n')
        data = yf.download(symbol, start='2020-01-01', end='2020-12-31')
        print(data)

if __name__ == "__main__":
    test_yfinance()