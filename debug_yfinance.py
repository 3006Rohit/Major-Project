import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

symbol = "AAPL"
lookback = 60
end_date = datetime.now()
start_date = end_date - timedelta(days=lookback)

print(f"Downloading {symbol} from {start_date} to {end_date}...")
data = yf.download(symbol, start=start_date, end=end_date, progress=False)

print("\nShape:", data.shape)
print("\nColumns:", data.columns)
print("\nHead:\n", data.head())

if isinstance(data.columns, pd.MultiIndex):
    print("\nDetected MultiIndex columns. Flattening...")
    data.columns = data.columns.get_level_values(0)
    print("New Columns:", data.columns)

data.reset_index(inplace=True)
print("\nAfter reset_index head:\n", data.head())

if len(data) == 0:
    print("\nERROR: Data is empty!")
else:
    print("\nSUCCESS: Data found.")
