#!/usr/bin/env python3
"""
Incremental Bybit ETHUSDT fetcher (15m OHLCV + mark + index, funding)
- Saves 2 parquet files:
    1) 15m_eth_market.parquet -> OHLCV + mark + index price
    2) funding_eth.parquet -> funding rate
- Appends only new data to existing files
- Historical fetch starts from 2025-09-30 00:00 UTC if file doesn't exist
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# ---------- CONFIG ----------
SYMBOL = "ETHUSDT"
CATEGORY = "linear"
INTERVAL = "15"  # 15 minute
START_DT = datetime(2025, 9, 30, 0, 0, 0, tzinfo=timezone.utc)
BASE = "https://api.bybit.com/v5/market"
KLINE_URL = f"{BASE}/kline"
MARK_KLINE_URL = f"{BASE}/mark-price-kline"
INDEX_KLINE_URL = f"{BASE}/index-price-kline"
FUNDING_URL = f"{BASE}/funding/history"
FEAR_GREED_URL = "https://api.alternative.me/fng/"

LIMIT = 200
SLEEP_BETWEEN = 0.12
RETRY_SLEEP = 1.0
MAX_RETRIES = 3

OUT_FILE = "15m_eth.parquet"


# ---------- helpers ----------
def safe_get(url, params=None, max_retries=MAX_RETRIES):
    attempt = 0
    while attempt < max_retries:
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            attempt += 1
            print(f"HTTP error ({attempt}/{max_retries}) for {url}: {e}")
            time.sleep(RETRY_SLEEP)
    raise RuntimeError(f"Failed to GET {url} after {max_retries} attempts")


def ms(dt):
    return int(dt.timestamp() * 1000)


def ms_to_dt(ms_ts):
    return datetime.fromtimestamp(ms_ts / 1000.0, tz=timezone.utc)


# ---------- fetch functions ----------
def fetch_klines(url, start_ms):
    params = {"category": CATEGORY, "symbol": SYMBOL, "interval": INTERVAL, "start": start_ms, "limit": LIMIT}
    js = safe_get(url, params=params)
    return js.get("result", {}).get("list", [])


def fetch_funding(start_ms, end_ms):
    params = {"category": CATEGORY, "symbol": SYMBOL, "limit": LIMIT, "startTime": start_ms, "endTime": end_ms}
    js = safe_get(FUNDING_URL, params=params)
    return js.get("result", {}).get("list", [])


def fetch_fear_greed_for_date_range(start_date, end_date):
    """Fetch Fear & Greed data for a specific date range"""
    try:
        # Calculate how many days we need
        days_needed = (end_date - start_date).days + 1
        limit = min(days_needed, 365)  # Limit to reasonable number
        
        print(f"Fetching Fear & Greed data for {days_needed} days (limit: {limit})")
        js = safe_get(f"{FEAR_GREED_URL}?limit={limit}")
        
        if js and "data" in js and len(js["data"]) > 0:
            fear_greed_by_date = {}
            for item in js["data"]:
                timestamp = int(item["timestamp"])
                # Convert to daily timestamp (start of day in UTC)
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                daily_timestamp = int(dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
                
                fear_greed_by_date[daily_timestamp] = {
                    "fear_greed_value": int(item["value"]),
                    "fear_greed_classification": item["value_classification"]
                }
            print(f"Fetched Fear & Greed data for {len(fear_greed_by_date)} days")
            return fear_greed_by_date
        else:
            print("No Fear & Greed data available")
            return {}
    except Exception as e:
        print(f"Error fetching Fear & Greed data: {e}")
        return {}


# ---------- parse functions ----------
def parse_kline_row(k):
    return {
        "start_time": ms_to_dt(int(k[0])),
        "open": float(k[1]),
        "high": float(k[2]),
        "low": float(k[3]),
        "close": float(k[4]),
        "volume": float(k[5]),
        "turnover": float(k[6])
    }


def parse_mark_idx_row(k, col_prefix):
    return {
        "start_time": ms_to_dt(int(k[0])),
        f"{col_prefix}_open": float(k[1]),
        f"{col_prefix}_high": float(k[2]),
        f"{col_prefix}_low": float(k[3]),
        f"{col_prefix}_close": float(k[4])
    }


# Note: parse_funding_row removed - funding data now uses OHLCV start_time


# ---------- incremental fetch ----------
def incremental_fetch_all_data():
    """Fetch OHLCV + mark + index price + funding + fear & greed and merge into one DataFrame"""
    interval_ms = 60 * 1000

    # Load existing file if exists
    if os.path.exists(OUT_FILE):
        df_existing = pd.read_parquet(OUT_FILE)
        last_start = pd.to_datetime(df_existing["start_time"].max()).tz_convert(timezone.utc)
        start_ms = int(last_start.timestamp() * 1000) + interval_ms
        print(f"Existing data file, fetching after {last_start}")
    else:
        df_existing = None
        start_ms = ms(START_DT)
        print(f"No data file, starting from {START_DT}")

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_rows = []
    seen_timestamps = set()

    # Fetch Fear & Greed data for the entire date range once
    start_date = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).date()
    end_date = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).date()
    fear_greed_data = fetch_fear_greed_for_date_range(start_date, end_date)

    cur = start_ms
    while cur < end_ms:
        try:
            # Fetch OHLCV, mark, index, and funding
            ohlcv_batch = fetch_klines(KLINE_URL, cur)
            mark_batch = fetch_klines(MARK_KLINE_URL, cur)
            index_batch = fetch_klines(INDEX_KLINE_URL, cur)
            
            # Fetch funding data for this time range (funding is updated every 8 hours)
            # For a 200-minute batch, we should get 0 or 1 funding data points
            batch_start_ms = cur
            batch_end_ms = cur + (200 * 60 * 1000)  # 200 minutes later
            funding_batch = fetch_funding(batch_start_ms, batch_end_ms)

            if not ohlcv_batch:
                break
                
            # Debug: show what we actually got from API
            print(f"API returned {len(ohlcv_batch)} OHLCV candles, {len(mark_batch)} mark candles, {len(index_batch)} index candles, {len(funding_batch)} funding records")
            
            # If we didn't get 200 candles, we might be at the end of available data
            if len(ohlcv_batch) < 200:
                print(f"Warning: Only got {len(ohlcv_batch)} candles instead of 200. This might be the end of available data.")
                # If we're getting very few candles consistently, we might be at the end
                if len(ohlcv_batch) < 10:
                    print("Very few candles returned, stopping fetch.")
                    break

            # Fear & Greed data will be fetched per timestamp as needed
            
            # Create a mapping of funding rates by timestamp for quick lookup
            funding_by_timestamp = {}
            for funding_item in funding_batch:
                funding_ts = int(funding_item.get("fundingRateTimestamp") or 0)
                if funding_ts > 0:
                    funding_by_timestamp[funding_ts] = float(funding_item.get("fundingRate") or 0)
            print(f"Funding data points: {len(funding_by_timestamp)} (expected: 0-1 for 200min batch)")
            
            # Get the most recent funding rate for this batch (since funding is updated every 8 hours)
            current_funding_rate = None
            if funding_by_timestamp:
                # Use the most recent funding rate in this batch
                latest_funding_ts = max(funding_by_timestamp.keys())
                current_funding_rate = funding_by_timestamp[latest_funding_ts]
                print(f"Using funding rate {current_funding_rate} from timestamp {latest_funding_ts}")
            
            # Process all candles in the batch (should be 200 candles = 200 minutes)
            new_rows = []
            for i, k in enumerate(ohlcv_batch):
                timestamp_ms = int(k[0])
                if timestamp_ms not in seen_timestamps:
                    row = parse_kline_row(k)  # This sets start_time as the primary timestamp
                    if i < len(mark_batch):
                        row.update(parse_mark_idx_row(mark_batch[i], "mark"))
                    if i < len(index_batch):
                        row.update(parse_mark_idx_row(index_batch[i], "index"))
                    
                    # Add Fear & Greed data to each row (using OHLCV start_time)
                    # Convert OHLCV timestamp to daily timestamp (start of day)
                    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                    daily_timestamp = int(dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
                    
                    if daily_timestamp in fear_greed_data:
                        row.update(fear_greed_data[daily_timestamp])
                    else:
                        # Find the closest previous daily Fear & Greed data
                        fear_greed_value = None
                        for fg_timestamp in sorted(fear_greed_data.keys()):
                            if fg_timestamp <= daily_timestamp:
                                fear_greed_value = fear_greed_data[fg_timestamp]
                            else:
                                break
                        
                        if fear_greed_value:
                            row.update(fear_greed_value)
                        else:
                            # Set default values if no Fear & Greed data found
                            row["fear_greed_value"] = None
                            row["fear_greed_classification"] = None
                    
                    # Add funding rate data (use the current funding rate for this batch)
                    # Since funding is updated every 8 hours, use the most recent rate for all candles in this batch
                    row["funding_rate"] = current_funding_rate
                    
                    new_rows.append(row)
                    seen_timestamps.add(timestamp_ms)

            all_rows.extend(new_rows)
            
            # Move to next batch: advance by 200 minutes (200 * 60 * 1000 ms)
            if ohlcv_batch:
                batch_start_time = ms_to_dt(cur)
                batch_end_time = ms_to_dt(int(ohlcv_batch[-1][0]))
                # Calculate expected end time (200 minutes = 3 hours 20 minutes)
                expected_end_time = ms_to_dt(cur + (200 * 60 * 1000))
                # Advance by 200 minutes for next batch
                cur = cur + (200 * 60 * 1000)  # Next batch starts 200 minutes later
                print(f"Fetched {len(new_rows)} new rows from {batch_start_time} to {batch_end_time} (expected: {expected_end_time})")
            else:
                break
                
            time.sleep(SLEEP_BETWEEN)

        except Exception as e:
            print("Error fetching batch:", e)
            time.sleep(RETRY_SLEEP)

    if not all_rows:
        print("No new market data.")
        return

    df_new = pd.DataFrame(all_rows)
    df_new.sort_values("start_time", inplace=True)

    if df_existing is not None and not df_existing.empty:
        combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=["start_time"], keep="last")
    else:
        combined = df_new

    combined.sort_values("start_time", inplace=True)
    
    # Forward-fill funding rates to ensure all rows have values
    combined['funding_rate'] = combined['funding_rate'].ffill()
    
    # Also backward-fill any remaining nulls at the beginning
    combined['funding_rate'] = combined['funding_rate'].bfill()
    
    combined.to_parquet(OUT_FILE, index=False)
    print(f"Saved all data -> {OUT_FILE} (new {len(df_new)}, total {len(combined)})")




# ---------- main ----------
def main():
    print("=== Incremental Bybit ETHUSDT fetch with Fear & Greed ===")
    incremental_fetch_all_data()
    print("=== Done ===")


if __name__ == "__main__":
    main()
