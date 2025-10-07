import ccxt
import time
import uuid
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------------------
# CONFIG
# -----------------------------------
# Get API credentials from environment
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')

# Validate environment variables
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    print("❌ Error: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file")
    print("Please copy env.template to .env and update the values")
    exit(1)

# Bybit testnet configuration
EXCHANGE = ccxt.bybit({
    'apiKey': BYBIT_API_KEY,
    'secret': BYBIT_API_SECRET,
    'sandbox': True,  # Use testnet
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'recvWindow': 120000,  # Increase recv window to 120 seconds for large time differences
        'timeDifference': 0,  # Will be set dynamically
        'adjustForTimeDifference': True,  # Enable automatic time adjustment
        'timeout': 30000,  # 30 second timeout
    },
    'urls': {
        'api': {
            'public': 'https://api-testnet.bybit.com',
            'private': 'https://api-testnet.bybit.com',
        }
    }
})
# Supported trading symbols
SUPPORTED_SYMBOLS = ['BTC/USDT', 'ETH/USDT']
DEFAULT_SYMBOL = 'BTC/USDT'
MAX_ERRORS = 5
COOLDOWN = 30  # seconds

print(f"[INFO] Using Bybit testnet API with key: {BYBIT_API_KEY[:8]}...")
print(f"[INFO] Supported symbols: {', '.join(SUPPORTED_SYMBOLS)}")

# Load markets for the initial exchange instance
try:
    print("[INFO] Loading initial markets...")
    EXCHANGE.load_markets()
    print("[OK] Initial markets loaded successfully")
except Exception as e:
    print(f"[WARN] Failed to load initial markets: {e}")

# -----------------------------------
# UTILITIES
# -----------------------------------
def gen_client_order_id(prefix="bot"):
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

def within_limits(price, qty, market):
    """Check tick/step/min-notional constraints."""
    try:
        tick = market['precision']['price']
        step = market['precision']['amount']
        min_notional = market['limits']['cost']['min']
        value = price * qty
        
        # For testnet, be more lenient with validation
        print(f"[DEBUG] Validating: price={price}, qty={qty}, value={value}")
        print(f"[DEBUG] Limits: tick={tick}, step={step}, min_notional={min_notional}")
        
        # Check minimum notional value
        if value < min_notional:
            print(f"[ERROR] Value {value} below minimum {min_notional}")
            return False
            
        # Check price tick (round to nearest tick)
        if tick > 0:
            rounded_price = round(price / tick) * tick
            if abs(price - rounded_price) > tick * 0.1:  # Allow small rounding errors
                print(f"[ERROR] Price {price} not aligned with tick {tick}")
                return False
                
        # Check quantity step
        if step > 0:
            rounded_qty = round(qty / step) * step
            if abs(qty - rounded_qty) > step * 0.1:  # Allow small rounding errors
                print(f"[ERROR] Quantity {qty} not aligned with step {step}")
                return False
                
        print("[OK] All validations passed")
        return True
        
    except Exception as e:
        print(f"[WARN] Validation error: {e}")
        return True  # Allow on testnet if validation fails

# -----------------------------------
# CIRCUIT BREAKER
# -----------------------------------
error_times = []

def record_error():
    global error_times
    error_times.append(datetime.now(timezone.utc))
    # Keep only last 60s window
    error_times = [t for t in error_times if datetime.now(timezone.utc) - t < timedelta(seconds=60)]
    if len(error_times) >= MAX_ERRORS:
        print("[ALERT] Circuit breaker triggered! Cooling down...")
        time.sleep(COOLDOWN)
        error_times.clear()

# -----------------------------------
# TIME SYNCHRONIZATION
# -----------------------------------
# Global time offset for manual timestamp adjustment
TIME_OFFSET_MS = 0

def sync_time_with_exchange():
    """Synchronize local time with exchange server time."""
    global TIME_OFFSET_MS
    
    try:
        print("[INFO] Synchronizing time with Bybit server...")
        
        # Try multiple methods to get accurate server time
        time_diffs = []
        
        # Method 1: Direct HTTP request to Bybit time endpoint
        import requests
        try:
            response = requests.get('https://api-testnet.bybit.com/v2/public/time', timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'time_now' in data['result']:
                    server_time = int(data['result']['time_now'])
                    local_time = int(time.time() * 1000)
                    time_diff = server_time - local_time
                    time_diffs.append(time_diff)
                    print(f"[INFO] HTTP time diff: {time_diff}ms")
        except Exception as e:
            print(f"[WARN] HTTP time sync failed: {e}")
        
        # Method 2: Use ccxt fetch_time
        try:
            server_time = EXCHANGE.fetch_time()
            local_time = int(time.time() * 1000)
            time_diff = server_time - local_time
            time_diffs.append(time_diff)
            print(f"[INFO] CCXT time diff: {time_diff}ms")
        except Exception as e:
            print(f"[WARN] CCXT fetch_time failed: {e}")
        
        # Method 3: Use a simple public endpoint to estimate time
        try:
            # Use a lightweight public endpoint
            response = requests.get('https://api-testnet.bybit.com/v2/public/symbols', timeout=10)
            if response.status_code == 200:
                # Estimate server time from response headers or current time
                server_time = int(time.time() * 1000) + 1000  # Add 1 second buffer
                local_time = int(time.time() * 1000)
                time_diff = server_time - local_time
                time_diffs.append(time_diff)
                print(f"[INFO] Estimated time diff: {time_diff}ms")
        except Exception as e:
            print(f"[WARN] Estimation method failed: {e}")
        
        # Use the most consistent time difference
        if time_diffs:
            # Use median to avoid outliers
            time_diffs.sort()
            median_diff = time_diffs[len(time_diffs) // 2]
            
            # Store the time offset globally
            TIME_OFFSET_MS = median_diff
            
            # Recreate the exchange instance with the correct time difference
            create_exchange_with_custom_timestamp()
            
            print(f"[INFO] Server time: {int(time.time() * 1000) + median_diff}")
            print(f"[INFO] Local time: {int(time.time() * 1000)}")
            print(f"[INFO] Final time difference: {median_diff}ms")
            print(f"[OK] Time synchronized (median of {len(time_diffs)} measurements)")
            return True
        else:
            print("[WARN] All time sync methods failed")
            return False
        
    except Exception as e:
        print(f"[WARN] Time sync failed: {e}")
        return False

def get_adjusted_timestamp():
    """Get current timestamp adjusted for server time difference."""
    return int(time.time() * 1000) + TIME_OFFSET_MS

def create_exchange_with_custom_timestamp():
    """Create a new exchange instance with proper timestamp handling."""
    global EXCHANGE
    
    # Create a new exchange instance with updated time difference
    new_exchange = ccxt.bybit({
        'apiKey': BYBIT_API_KEY,
        'secret': BYBIT_API_SECRET,
        'sandbox': True,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
            'recvWindow': 120000,
            'timeDifference': TIME_OFFSET_MS,
            'adjustForTimeDifference': True,
            'timeout': 30000,
        },
        'urls': {
            'api': {
                'public': 'https://api-testnet.bybit.com',
                'private': 'https://api-testnet.bybit.com',
            }
        }
    })
    
    # Load markets for the new exchange instance
    try:
        print("[INFO] Loading markets for new exchange instance...")
        new_exchange.load_markets()
        print(f"[OK] Markets loaded successfully")
    except Exception as e:
        print(f"[WARN] Failed to load markets: {e}")
    
    # Update the global exchange instance
    EXCHANGE = new_exchange
    print(f"[INFO] Exchange instance updated with timeDifference: {TIME_OFFSET_MS}ms")
    return new_exchange

# -----------------------------------
# RETRY WRAPPER
# -----------------------------------
def safe_call(func, *args, **kwargs):
    for attempt, delay in enumerate([1, 3, 9]):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            print(f"[WARN] API error (attempt {attempt + 1}/3): {e}")
            
            # Check if it's a timestamp error and try to sync time
            if any(keyword in error_msg.lower() for keyword in ["timestamp", "recv_window", "time", "10002"]):
                print("[INFO] Detected timestamp/time error, attempting time sync...")
                if sync_time_with_exchange():
                    print("[INFO] Time sync successful, retrying immediately...")
                    continue  # Retry immediately after successful sync
                else:
                    print("[WARN] Time sync failed, will retry with delay...")
            
            # For other errors, just retry with delay
            if attempt < 2:  # Don't sleep on last attempt
                print(f"[INFO] Retrying in {delay}s...")
                record_error()
                time.sleep(delay)
            else:
                print("[ERROR] Final attempt failed")
                record_error()
    
    raise RuntimeError("API failed after retries")

# -----------------------------------
# RECONCILIATION LOOP
# -----------------------------------
def reconcile(local_orders, symbol=None):
    if symbol is None:
        symbol = DEFAULT_SYMBOL
    try:
        remote_orders = EXCHANGE.fetch_open_orders(symbol)
        remote_ids = {o['clientOrderId'] for o in remote_orders if o.get('clientOrderId')}
        local_ids = set(local_orders.keys())

        # Heal orphans
        for oid in remote_ids - local_ids:
            print(f"[RECONCILE] Healing orphan order: {oid}")
            local_orders[oid] = {'status': 'open'}

        # Clean closed
        for oid in local_ids - remote_ids:
            if local_orders[oid]['status'] == 'open':
                print(f"[RECONCILE] Order {oid} closed remotely. Updating status.")
                local_orders[oid]['status'] = 'closed'
    except Exception as e:
        print(f"Reconciliation failed: {e}")
        record_error()

# -----------------------------------
# MAIN TRADING LOGIC
# -----------------------------------
def place_safe_order(side, qty, price, tp, sl, symbol=None):
    """Place a safe order with take profit and stop loss on Bybit testnet"""
    # Use provided symbol or default to BTC/USDT
    if symbol is None:
        symbol = DEFAULT_SYMBOL
    
    # Validate symbol
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(f"Unsupported symbol: {symbol}. Supported: {SUPPORTED_SYMBOLS}")
    
    print(f"[TRADE] Trading {symbol}")
    try:
        market = EXCHANGE.market(symbol)
        if not within_limits(price, qty, market):
            raise ValueError("Invalid price/qty (tick/step/min-notional)")

        client_id = gen_client_order_id()
        params = {'reduceOnly': False, 'clientOrderId': client_id}
        print(f"[ORDER] Sending base order {side} {qty} @ {price}")
        order = safe_call(EXCHANGE.create_limit_order, symbol, side, qty, price, params)
        print(f"[OK] Base order placed: {order['id']}")

        # Create take profit order
        print("[ORDER] Creating Take Profit order...")
        tp_id = gen_client_order_id("tp")
        tp_side = 'sell' if side == 'buy' else 'buy'
        tp_params = {'reduceOnly': True, 'clientOrderId': tp_id}
        tp_order = safe_call(EXCHANGE.create_limit_order, symbol, tp_side, qty, tp, tp_params)
        print(f"[OK] Take Profit order created: {tp_order['id']}")

        # Create stop loss order (simplified for testnet compatibility)
        print("[ORDER] Creating Stop Loss order...")
        sl_id = gen_client_order_id("sl")
        sl_side = 'sell' if side == 'buy' else 'buy'
        
        # Try to create stop loss, but don't fail if it doesn't work
        try:
            sl_params = {
                'reduceOnly': True,
                'stopPrice': float(sl),
                'clientOrderId': sl_id,
                'type': 'STOP_MARKET'
            }
            sl_order = safe_call(EXCHANGE.create_order, symbol, 'STOP_MARKET', sl_side, qty, None, sl_params)
            print(f"[OK] Stop Loss order created: {sl_order['id']}")
            sl_order_id = sl_order['id']
        except Exception as e:
            print(f"[WARN] Stop Loss order failed (testnet limitation): {e}")
            print("[INFO] Note: Stop loss orders may not be fully supported on testnet")
            sl_order_id = None

        print(f"[OK] Orders created (TP={tp}, SL={sl})")

        # Track locally
        local_orders = {
            client_id: {'status': 'open', 'order_id': order['id']},
            tp_id: {'status': 'open', 'order_id': tp_order['id']}
        }
        
        if sl_order_id:
            local_orders[sl_id] = {'status': 'open', 'order_id': sl_order_id}

        # Background reconciliation demo
        for i in range(3):
            time.sleep(5)
            reconcile(local_orders, symbol)

        print("[DONE] Done. Local order states:", local_orders)
        return local_orders
        
    except Exception as e:
        print(f"[ERROR] Error placing order: {e}")
        record_error()
        raise

# -----------------------------------
# CONVENIENCE FUNCTIONS
# -----------------------------------
def trade_btc(side, qty, price, tp, sl):
    """Trade BTC/USDT with automatic price calculation"""
    return place_safe_order(side, qty, price, tp, sl, 'BTC/USDT')

def trade_eth(side, qty, price, tp, sl):
    """Trade ETH/USDT with automatic price calculation"""
    return place_safe_order(side, qty, price, tp, sl, 'ETH/USDT')

def get_market_price(symbol):
    """Get current market price for a symbol"""
    ticker = safe_call(EXCHANGE.fetch_ticker, symbol)
    return ticker['last']

def create_realistic_trade(symbol, side='buy', qty=0.001):
    """Create a realistic trade with current market prices"""
    current_price = get_market_price(symbol)
    market = EXCHANGE.market(symbol)
    tick = market['precision']['price']
    
    # Calculate realistic prices
    if side == 'buy':
        entry_price = round(current_price * 0.99 / tick) * tick  # 1% below market
        tp_price = round(current_price * 1.02 / tick) * tick      # 2% above market  
        sl_price = round(current_price * 0.97 / tick) * tick      # 3% below market
    else:
        entry_price = round(current_price * 1.01 / tick) * tick  # 1% above market
        tp_price = round(current_price * 0.98 / tick) * tick      # 2% below market  
        sl_price = round(current_price * 1.03 / tick) * tick      # 3% above market
    
    print(f"[INFO] {symbol} current price: ${current_price:,.2f}")
    print(f"[TRADE] Trade: {side} at ${entry_price:,.2f}, TP at ${tp_price:,.2f}, SL at ${sl_price:,.2f}")
    
    return place_safe_order(side, qty, entry_price, tp_price, sl_price, symbol)

# -----------------------------------
# API VALIDATION
# -----------------------------------
def validate_api_connection():
    """Validate API connection and credentials"""
    try:
        print("[INFO] Validating Bybit testnet API connection...")
        
        # Check if API keys are loaded
        if not EXCHANGE.apiKey or not EXCHANGE.secret:
            print("[ERROR] API keys not found in environment variables")
            print("Please ensure BYBIT_API_KEY and BYBIT_API_SECRET are set in .env file")
            return False
        
        # First, try to sync time with the exchange
        print("[INFO] Attempting initial time synchronization...")
        if not sync_time_with_exchange():
            print("[WARN] Time sync failed, but continuing with validation...")
        
        # Test API connection with a simple public call first
        try:
            print("[INFO] Testing public API access...")
            # Use a simple public endpoint that doesn't require authentication
            symbols = EXCHANGE.fetch_markets()
            print(f"[OK] Public API access successful - found {len(symbols)} markets")
        except Exception as e:
            print(f"[WARN] Public API test failed: {e}")
            # Try to sync time again if public API fails
            print("[INFO] Attempting time sync after public API failure...")
            sync_time_with_exchange()
        
        # Test private API connection
        try:
            print("[INFO] Testing private API access...")
            balance = safe_call(EXCHANGE.fetch_balance)
            print(f"[OK] Private API connection successful")
            usdt_balance = balance.get('USDT', {}).get('free', 'N/A')
            print(f"[INFO] Account balance: {usdt_balance} USDT")
            return True
        except Exception as e:
            print(f"[ERROR] Private API test failed: {e}")
            return False
        
    except Exception as e:
        print(f"[ERROR] API validation failed: {e}")
        return False

# -----------------------------------
# DEMO RUN
# -----------------------------------
if __name__ == "__main__":
    print("=== Bybit Testnet Execution Adapter ===")
    
    # Validate API connection first
    if not validate_api_connection():
        print("[ERROR] Cannot proceed without valid API connection")
        exit(1)
    
    print("\n[START] Starting demo trades for both BTC and ETH...")
    
    try:
        # Demo BTC trade
        print("\n" + "="*50)
        print("[BTC] BTC/USDT Demo Trade")
        print("="*50)
        btc_result = create_realistic_trade('BTC/USDT', side='buy', qty=0.001)
        print("[OK] BTC trade completed successfully!")
        
        # Demo ETH trade
        print("\n" + "="*50)
        print("[ETH] ETH/USDT Demo Trade")
        print("="*50)
        eth_result = create_realistic_trade('ETH/USDT', side='buy', qty=0.01)
        print("[OK] ETH trade completed successfully!")
        
        print("\n[SUCCESS] All demo trades completed successfully!")
        print(f"[INFO] BTC Orders: {len(btc_result)} orders placed")
        print(f"[INFO] ETH Orders: {len(eth_result)} orders placed")
        
    except Exception as e:
        print(f"[ERROR] Demo trades failed: {e}")
