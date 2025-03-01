import yfinance as yf
import pandas as pd
import mplfinance as mpf


symbol = "^GSPC" # S&P500
start_date = "2024-12-01"
end_date = "2024-12-31"

data = yf.download(symbol, start=start_date, end=end_date, interval="1d")

if data.empty:
    print("failed")
else:
    print("Success")

data.to_csv("SPX500.csv")
data.reset_index(inplace=True)
print(data.columns)

#mpf.plot(data, type="candle", title="S&P 500 Daily Candlestick", ylabel="Price (USD)", style="charles", figsize=(10, 6))
