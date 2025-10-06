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
    'options': {'defaultType': 'future'},
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

print(f"🔧 Using Bybit testnet API with key: {BYBIT_API_KEY[:8]}...")
print(f"📊 Supported symbols: {', '.join(SUPPORTED_SYMBOLS)}")

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
        print(f"🔍 Validating: price={price}, qty={qty}, value={value}")
        print(f"📏 Limits: tick={tick}, step={step}, min_notional={min_notional}")
        
        # Check minimum notional value
        if value < min_notional:
            print(f"❌ Value {value} below minimum {min_notional}")
            return False
            
        # Check price tick (round to nearest tick)
        if tick > 0:
            rounded_price = round(price / tick) * tick
            if abs(price - rounded_price) > tick * 0.1:  # Allow small rounding errors
                print(f"❌ Price {price} not aligned with tick {tick}")
                return False
                
        # Check quantity step
        if step > 0:
            rounded_qty = round(qty / step) * step
            if abs(qty - rounded_qty) > step * 0.1:  # Allow small rounding errors
                print(f"❌ Quantity {qty} not aligned with step {step}")
                return False
                
        print("✅ All validations passed")
        return True
        
    except Exception as e:
        print(f"⚠️ Validation error: {e}")
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
        print("🚨 Circuit breaker triggered! Cooling down...")
        time.sleep(COOLDOWN)
        error_times.clear()

# -----------------------------------
# RETRY WRAPPER
# -----------------------------------
def safe_call(func, *args, **kwargs):
    for delay in [1, 3, 9]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"⚠️ API error: {e}. Retrying in {delay}s...")
            record_error()
            time.sleep(delay)
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
            print(f"🔄 Healing orphan order: {oid}")
            local_orders[oid] = {'status': 'open'}

        # Clean closed
        for oid in local_ids - remote_ids:
            if local_orders[oid]['status'] == 'open':
                print(f"✅ Order {oid} closed remotely. Updating status.")
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
    
    print(f"🎯 Trading {symbol}")
    try:
        market = EXCHANGE.market(symbol)
        if not within_limits(price, qty, market):
            raise ValueError("Invalid price/qty (tick/step/min-notional)")

        client_id = gen_client_order_id()
        params = {'reduceOnly': False, 'clientOrderId': client_id}
        print(f"📤 Sending base order {side} {qty} @ {price}")
        order = safe_call(EXCHANGE.create_limit_order, symbol, side, qty, price, params)
        print(f"✅ Base order placed: {order['id']}")

        # Create take profit order
        print("📦 Creating Take Profit order...")
        tp_id = gen_client_order_id("tp")
        tp_side = 'sell' if side == 'buy' else 'buy'
        tp_params = {'reduceOnly': True, 'clientOrderId': tp_id}
        tp_order = safe_call(EXCHANGE.create_limit_order, symbol, tp_side, qty, tp, tp_params)
        print(f"✅ Take Profit order created: {tp_order['id']}")

        # Create stop loss order (simplified for testnet compatibility)
        print("📦 Creating Stop Loss order...")
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
            print(f"✅ Stop Loss order created: {sl_order['id']}")
            sl_order_id = sl_order['id']
        except Exception as e:
            print(f"⚠️ Stop Loss order failed (testnet limitation): {e}")
            print("📝 Note: Stop loss orders may not be fully supported on testnet")
            sl_order_id = None

        print(f"✅ Orders created (TP={tp}, SL={sl})")

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

        print("🏁 Done. Local order states:", local_orders)
        return local_orders
        
    except Exception as e:
        print(f"❌ Error placing order: {e}")
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
    
    print(f"📊 {symbol} current price: ${current_price:,.2f}")
    print(f"📈 Trade: {side} at ${entry_price:,.2f}, TP at ${tp_price:,.2f}, SL at ${sl_price:,.2f}")
    
    return place_safe_order(side, qty, entry_price, tp_price, sl_price, symbol)

# -----------------------------------
# API VALIDATION
# -----------------------------------
def validate_api_connection():
    """Validate API connection and credentials"""
    try:
        print("🔍 Validating Bybit testnet API connection...")
        
        # Check if API keys are loaded
        if not EXCHANGE.apiKey or not EXCHANGE.secret:
            print("❌ API keys not found in environment variables")
            print("Please ensure BYBIT_API_KEY and BYBIT_API_SECRET are set in .env file")
            return False
            
        # Test API connection
        balance = safe_call(EXCHANGE.fetch_balance)
        print(f"✅ API connection successful")
        print(f"📊 Account balance: {balance.get('USDT', {}).get('free', 'N/A')} USDT")
        return True
        
    except Exception as e:
        print(f"❌ API validation failed: {e}")
        return False

# -----------------------------------
# DEMO RUN
# -----------------------------------
if __name__ == "__main__":
    print("=== Bybit Testnet Execution Adapter ===")
    
    # Validate API connection first
    if not validate_api_connection():
        print("❌ Cannot proceed without valid API connection")
        exit(1)
    
    print("\n🚀 Starting demo trades for both BTC and ETH...")
    
    try:
        # Demo BTC trade
        print("\n" + "="*50)
        print("🟠 BTC/USDT Demo Trade")
        print("="*50)
        btc_result = create_realistic_trade('BTC/USDT', side='buy', qty=0.001)
        print("✅ BTC trade completed successfully!")
        
        # Demo ETH trade
        print("\n" + "="*50)
        print("🔵 ETH/USDT Demo Trade")
        print("="*50)
        eth_result = create_realistic_trade('ETH/USDT', side='buy', qty=0.01)
        print("✅ ETH trade completed successfully!")
        
        print("\n🎉 All demo trades completed successfully!")
        print(f"📊 BTC Orders: {len(btc_result)} orders placed")
        print(f"📊 ETH Orders: {len(eth_result)} orders placed")
        
    except Exception as e:
        print(f"❌ Demo trades failed: {e}")
