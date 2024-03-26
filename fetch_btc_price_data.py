import ccxt
import pandas as pd

# Initialize the Binance exchange API
binance = ccxt.binance()

# Define the symbol (BTC/USDT for Bitcoin prices in USDT)
symbol = 'BTC/USDT'

# Define the timeframe (e.g., 1h, 1d etc)
timeframe = '1d'

# Define the filename
filename = 'bitcoin_price_data_day.csv'

# Fetch historical price data iteratively
print("Fetching Bitcoin price data...")

# Fetch historical price data with automatic pagination
ohlcv = binance.fetch_ohlcv(symbol, timeframe, params={"paginate": True})

# Convert the data to a pandas DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Save the DataFrame to a CSV file
df.to_csv(filename, index=False)

print(f"Bitcoin price data saved to '{filename}'")