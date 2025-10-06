import pandas as pd
from pybit.unified_trading import HTTP
from datetime import datetime, timedelta
import pyarrow.parquet as pq
import pyarrow as pa

# Connect to Bybit (no API key needed for public data)
session = HTTP(testnet=False, timeout=30)  # Add timeout for better reliability

def fetch_ohlcv(symbol="BTCUSDT", interval="1", limit=200):
    """Fetch OHLCV (kline) data"""
    resp = session.get_kline(
        category="linear",   # for USDT perpetuals
        symbol=symbol,
        interval=interval,
        limit=limit
    )
    return resp["result"]["list"]

def fetch_mark_price(symbol="BTCUSDT", start_time=None, end_time=None, interval=1):
    """Fetch mark price data with 1-minute intervals (one record per minute)"""
    all_data = []
    
    if start_time and end_time:
        # Convert to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        print(f"  Fetching mark price data: {interval}-minute intervals from {start_time} to {end_time}")
        
        # For a 1-hour range, we can fetch all data in one request
        resp = session.get_mark_price_kline(
            category="linear",
            symbol=symbol,
            interval=interval,  # 1-minute intervals
            start=start_ms,
            end=end_ms,
            limit=1000  # Should be enough for 1 hour (60 records)
        )
        
        data = resp["result"]["list"]
        if data:
            all_data.extend(data)
            print(f"  Mark price: Retrieved {len(all_data)} records (1 per minute)")
        else:
            print("  Mark price: No data found for the specified time range")
    else:
        # Fetch latest data if no date range specified
        resp = session.get_mark_price_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=200
        )
        all_data = resp["result"]["list"]
    
    return all_data

def fetch_index_price(symbol="BTCUSDT", start_time=None, end_time=None, interval=1):
    """Fetch index price data with 1-minute intervals (one record per minute)"""
    all_data = []
    
    if start_time and end_time:
        # Convert to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        print(f"  Fetching index price data: {interval}-minute intervals from {start_time} to {end_time}")
        
        # For a 1-hour range, we can fetch all data in one request
        resp = session.get_index_price_kline(
            category="linear",
            symbol=symbol,
            interval=interval,  # 1-minute intervals
            start=start_ms,
            end=end_ms,
            limit=1000  # Should be enough for 1 hour (60 records)
        )
        
        data = resp["result"]["list"]
        if data:
            all_data.extend(data)
            print(f"  Index price: Retrieved {len(all_data)} records (1 per minute)")
        else:
            print("  Index price: No data found for the specified time range")
    else:
        # Fetch latest data if no date range specified
        resp = session.get_index_price_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=200
        )
        all_data = resp["result"]["list"]
    
    return all_data

# Configuration: Fetch data from 3 PM to 4 PM on June 1, 2025
start_date = datetime(2025, 6, 1, 15, 0, 0)  # 3:00 PM
end_date = datetime(2025, 6, 1, 16, 0, 0)    # 4:00 PM
print("Mode: Fetching data from 3 PM to 4 PM on June 1, 2025")

print(f"Fetching historical data from {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
print("Fetching 1-minute interval data (one record per minute)...")

# Fetch historical mark price and index price data
print("Fetching mark price data...")
mark = fetch_mark_price(start_time=start_date, end_time=end_date)

print("Fetching index price data...")
index = fetch_index_price(start_time=start_date, end_time=end_date)

print(f"Retrieved {len(mark)} mark price records and {len(index)} index price records")

# Convert to DataFrames
df_mark = pd.DataFrame(mark, columns=[
    "start_time", "mark_open", "mark_high", "mark_low", "mark_close"
])
df_index = pd.DataFrame(index, columns=[
    "start_time", "index_open", "index_high", "index_low", "index_close"
])

# Convert timestamps (Bybit returns timestamps in milliseconds)
for df in [df_mark, df_index]:
    df["start_time"] = pd.to_datetime(pd.to_numeric(df["start_time"]), unit="ms")

# Merge mark price and index price data on timestamp
df = df_mark.merge(df_index, on="start_time", how="outer")
df = df.sort_values("start_time").reset_index(drop=True)

# Save to Parquet
filename = f"bybit_mark_index_prices_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.parquet"
table = pa.Table.from_pandas(df)
pq.write_table(table, filename)

print(f"Saved historical data to {filename}")
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame columns: {df.columns.tolist()}")
print(f"Date range: {df['start_time'].min()} to {df['start_time'].max()}")
print("\nFirst few rows:")
print(df.head())
print("\nLast few rows:")
print(df.tail())
