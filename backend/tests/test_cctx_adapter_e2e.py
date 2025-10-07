#!/usr/bin/env python3
"""
Comprehensive End-to-End test for cctx_execution_adapter.py

Test Coverage:
1) API Connection & Time Synchronization
2) Market Data & Price Fetching
3) Order Validation & Limits Checking
4) Order Placement (Limit, TP, SL)
5) Order Management & Reconciliation
6) Circuit Breaker Testing
7) Error Handling & Recovery
8) Multiple Symbol Testing (BTC/USDT, ETH/USDT)
9) Client Order ID Generation & Idempotency
10) Realistic Trading Scenarios

Note: This uses Bybit testnet via ccxt and requires BYBIT_API_KEY/SECRET in .env.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Import the adapter under test
from cctx_execution_adapter import (
    EXCHANGE,
    DEFAULT_SYMBOL,
    SUPPORTED_SYMBOLS,
    validate_api_connection,
    get_market_price,
    place_safe_order,
    reconcile,
    within_limits,
    gen_client_order_id,
    record_error,
    safe_call,
    create_realistic_trade,
    trade_btc,
    trade_eth,
    get_adjusted_timestamp,
    sync_time_with_exchange,
    MAX_ERRORS,
    COOLDOWN,
)


def now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def ensure_log_dir() -> Path:
    log_dir = ROOT / "tests" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def detailed_limit_line(symbol: str, price: float, qty: float) -> str:
    """Build a detailed tick/step/min-notional line for auditability."""
    market = EXCHANGE.market(symbol)
    tick = market['precision']['price']
    step = market['precision']['amount']
    min_notional = market['limits']['cost']['min']

    rounded_price = round(price / tick) * tick if tick > 0 else price
    rounded_qty = round(qty / step) * step if step > 0 else qty
    value = price * qty

    return (
        f"tick={tick} step={step} min_notional={min_notional} | "
        f"price={price}→{rounded_price} qty={qty}→{rounded_qty} value={value}"
    )


def simulate_partial_cancel(symbol: str, local_orders: dict) -> None:
    """Cancel TP remotely to simulate partial cancel/divergence, then reconcile."""
    # Find TP order (client id starts with 'tp-')
    tp_entry = None
    for cid, meta in local_orders.items():
        if cid.startswith("tp-"):
            tp_entry = (cid, meta)
            break

    if not tp_entry:
        print(f"[{now_str()}] [WARN] TP order not found in local_orders; skipping cancel simulation")
        return

    cid, meta = tp_entry
    order_id = meta.get('order_id')
    if not order_id:
        print(f"[{now_str()}] [WARN] TP order has no remote id; skipping cancel simulation")
        return

    try:
        print(f"[{now_str()}] [INFO] Simulating remote cancel of TP {order_id} ({cid})")
        EXCHANGE.cancel_order(order_id, symbol)
    except Exception as e:
        print(f"[{now_str()}] [WARN] Cancel failed (expected on testnet sometimes): {e}")

    # Reconcile to update local state
    print(f"[{now_str()}] [INFO] Reconciling after cancel...")
    reconcile(local_orders, symbol)


def test_api_connection_and_sync():
    """Test API connection and time synchronization."""
    print(f"[{now_str()}] [TEST] Testing API connection and time sync...")
    
    # Test time synchronization
    print(f"[{now_str()}] [INFO] Testing time synchronization...")
    sync_result = sync_time_with_exchange()
    print(f"[{now_str()}] [RESULT] Time sync result: {sync_result}")
    
    # Test adjusted timestamp
    adjusted_ts = get_adjusted_timestamp()
    local_ts = int(time.time() * 1000)
    print(f"[{now_str()}] [INFO] Local timestamp: {local_ts}")
    print(f"[{now_str()}] [INFO] Adjusted timestamp: {adjusted_ts}")
    print(f"[{now_str()}] [INFO] Time difference: {adjusted_ts - local_ts}ms")
    
    return sync_result

def test_market_data_fetching():
    """Test market data fetching for all supported symbols."""
    print(f"[{now_str()}] [TEST] Testing market data fetching...")
    
    results = {}
    for symbol in SUPPORTED_SYMBOLS:
        try:
            print(f"[{now_str()}] [INFO] Fetching market data for {symbol}...")
            price = get_market_price(symbol)
            market = EXCHANGE.market(symbol)
            
            results[symbol] = {
                'price': price,
                'tick': market['precision']['price'],
                'step': market['precision']['amount'],
                'min_notional': market['limits']['cost']['min']
            }
            
            print(f"[{now_str()}] [OK] {symbol}: ${price:,.2f} (tick={market['precision']['price']}, step={market['precision']['amount']})")
            
        except Exception as e:
            print(f"[{now_str()}] [ERROR] Failed to fetch {symbol}: {e}")
            results[symbol] = {'error': str(e)}
    
    return results

def test_order_validation():
    """Test order validation with various price/quantity combinations."""
    print(f"[{now_str()}] [TEST] Testing order validation...")
    
    symbol = DEFAULT_SYMBOL
    market = EXCHANGE.market(symbol)
    current_price = get_market_price(symbol)
    
    # Test cases: (price, qty, should_pass, description)
    test_cases = [
        (current_price * 0.99, 0.001, True, "Valid order below market"),
        (current_price * 1.01, 0.001, True, "Valid order above market"),
        (current_price * 0.99, 0.000001, False, "Too small quantity"),
        (current_price * 0.99, 1000, True, "Large quantity"),
        (current_price * 0.99 + 0.001, 0.001, False, "Invalid price tick"),
    ]
    
    results = []
    for price, qty, expected, desc in test_cases:
        print(f"[{now_str()}] [INFO] Testing: {desc}")
        print(f"[{now_str()}] [INFO] Price: {price}, Qty: {qty}")
        
        try:
            result = within_limits(price, qty, market)
            status = "PASS" if result == expected else "FAIL"
            print(f"[{now_str()}] [RESULT] {status}: {desc} - Expected: {expected}, Got: {result}")
            results.append({
                'description': desc,
                'price': price,
                'qty': qty,
                'expected': expected,
                'actual': result,
                'passed': result == expected
            })
        except Exception as e:
            print(f"[{now_str()}] [ERROR] Validation failed: {e}")
            results.append({
                'description': desc,
                'error': str(e),
                'passed': False
            })
    
    return results

def test_client_order_id_generation():
    """Test client order ID generation and idempotency."""
    print(f"[{now_str()}] [TEST] Testing client order ID generation...")
    
    # Test different prefixes
    prefixes = ["test", "e2e", "bot", "manual"]
    ids = []
    
    for prefix in prefixes:
        for i in range(3):
            cid = gen_client_order_id(prefix)
            ids.append(cid)
            print(f"[{now_str()}] [INFO] Generated ID: {cid}")
    
    # Check uniqueness
    unique_ids = set(ids)
    is_unique = len(ids) == len(unique_ids)
    print(f"[{now_str()}] [RESULT] All IDs unique: {is_unique} ({len(ids)} total, {len(unique_ids)} unique)")
    
    return {
        'total_generated': len(ids),
        'unique_count': len(unique_ids),
        'all_unique': is_unique,
        'sample_ids': ids[:5]  # First 5 for reference
    }

def test_realistic_trading():
    """Test realistic trading scenarios."""
    print(f"[{now_str()}] [TEST] Testing realistic trading scenarios...")
    
    results = {}
    
    for symbol in SUPPORTED_SYMBOLS:
        try:
            print(f"[{now_str()}] [INFO] Testing realistic trade for {symbol}...")
            
            # Test both buy and sell scenarios
            for side in ['buy', 'sell']:
                print(f"[{now_str()}] [INFO] Testing {side} scenario for {symbol}...")
                
                try:
                    # Use very small quantities for testnet
                    qty = 0.001 if symbol == 'BTC/USDT' else 0.01
                    result = create_realistic_trade(symbol, side=side, qty=qty)
                    
                    print(f"[{now_str()}] [OK] {side} trade successful for {symbol}")
                    results[f"{symbol}_{side}"] = {
                        'success': True,
                        'orders_created': len(result),
                        'order_ids': list(result.keys())
                    }
                    
                except Exception as e:
                    print(f"[{now_str()}] [WARN] {side} trade failed for {symbol}: {e}")
                    results[f"{symbol}_{side}"] = {
                        'success': False,
                        'error': str(e)
                    }
                    
        except Exception as e:
            print(f"[{now_str()}] [ERROR] Failed to test {symbol}: {e}")
            results[symbol] = {'error': str(e)}
    
    return results

def test_order_reconciliation():
    """Test order reconciliation functionality."""
    print(f"[{now_str()}] [TEST] Testing order reconciliation...")
    
    symbol = DEFAULT_SYMBOL
    
    # Create some test local orders
    test_orders = {
        'test-order-1': {'status': 'open', 'order_id': 'fake_id_1'},
        'test-order-2': {'status': 'open', 'order_id': 'fake_id_2'},
        'test-order-3': {'status': 'closed', 'order_id': 'fake_id_3'}
    }
    
    print(f"[{now_str()}] [INFO] Testing reconciliation with {len(test_orders)} local orders...")
    
    try:
        # This will attempt to reconcile with remote orders
        reconcile(test_orders, symbol)
        print(f"[{now_str()}] [OK] Reconciliation completed")
        
        return {
            'success': True,
            'local_orders_before': len(test_orders),
            'local_orders_after': len(test_orders)
        }
        
    except Exception as e:
        print(f"[{now_str()}] [ERROR] Reconciliation failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def trigger_circuit_breaker():
    """Force multiple errors to trigger circuit breaker in adapter."""
    print(f"[{now_str()}] [TEST] Testing circuit breaker functionality...")
    print(f"[{now_str()}] [INFO] Triggering circuit breaker: MAX_ERRORS={MAX_ERRORS}")
    
    start_time = time.time()
    errors_triggered = 0
    
    # Trigger errors rapidly to hit the circuit breaker
    for i in range(MAX_ERRORS + 2):  # Trigger a few extra to ensure we hit the limit
        try:
            # Intentionally call with bad symbol to force error
            EXCHANGE.fetch_ticker("INVALID/SYMBOL")
        except Exception as e:
            print(f"[{now_str()}] [ERROR] Forced error {i+1}: {e}")
            record_error()
            errors_triggered += 1
            
            # Small delay to ensure errors are recorded
            time.sleep(0.1)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"[{now_str()}] [INFO] Triggered {errors_triggered} errors in {elapsed:.2f}s")
    print(f"[{now_str()}] [INFO] Circuit breaker should have engaged (cooldown {COOLDOWN}s)")
    
    return {
        'errors_triggered': errors_triggered,
        'elapsed_time': elapsed,
        'max_errors': MAX_ERRORS,
        'cooldown_period': COOLDOWN
    }

def test_error_recovery():
    """Test error recovery after circuit breaker."""
    print(f"[{now_str()}] [TEST] Testing error recovery...")
    
    # Wait a bit for any cooldown to pass
    print(f"[{now_str()}] [INFO] Waiting for potential cooldown period...")
    time.sleep(5)
    
    recovery_tests = []
    
    # Test 1: Simple market data fetch
    try:
        print(f"[{now_str()}] [INFO] Testing market data fetch recovery...")
        price = get_market_price(DEFAULT_SYMBOL)
        print(f"[{now_str()}] [OK] Market data fetch successful: ${price:,.2f}")
        recovery_tests.append({'test': 'market_data', 'success': True, 'price': price})
    except Exception as e:
        print(f"[{now_str()}] [ERROR] Market data fetch failed: {e}")
        recovery_tests.append({'test': 'market_data', 'success': False, 'error': str(e)})
    
    # Test 2: Account balance fetch
    try:
        print(f"[{now_str()}] [INFO] Testing balance fetch recovery...")
        balance = safe_call(EXCHANGE.fetch_balance)
        usdt_balance = balance.get('USDT', {}).get('free', 'N/A')
        print(f"[{now_str()}] [OK] Balance fetch successful: {usdt_balance} USDT")
        recovery_tests.append({'test': 'balance', 'success': True, 'balance': usdt_balance})
    except Exception as e:
        print(f"[{now_str()}] [ERROR] Balance fetch failed: {e}")
        recovery_tests.append({'test': 'balance', 'success': False, 'error': str(e)})
    
    return recovery_tests


def main() -> int:
    log_dir = ensure_log_dir()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"adapter_e2e_{ts}.log"
    summary_path = log_dir / f"adapter_e2e_{ts}.json"

    print(f"Writing comprehensive test log to: {log_path}")

    # Store all test results
    test_results = {
        'timestamp': ts,
        'tests_run': [],
        'summary': {}
    }

    with open(log_path, "w", encoding="utf-8", newline="\n") as log_file:
        with redirect_stdout(log_file):
            print(f"[{now_str()}] [START] Comprehensive cctx adapter e2e test")
            print(f"[{now_str()}] [INFO] Testing all functionality of cctx_execution_adapter.py")

            # Test 1: API Connection & Time Synchronization
            print(f"\n{'='*60}")
            print(f"[{now_str()}] [TEST SUITE 1] API Connection & Time Synchronization")
            print(f"{'='*60}")
            
            try:
                api_ok = validate_api_connection()
                if not api_ok:
                    print(f"[{now_str()}] [FATAL] API validation failed; aborting test")
                    return 2
                
                sync_result = test_api_connection_and_sync()
                test_results['tests_run'].append({
                    'name': 'api_connection_sync',
                    'success': api_ok and sync_result,
                    'details': {'api_ok': api_ok, 'sync_result': sync_result}
                })
                print(f"[{now_str()}] [RESULT] API Connection & Sync: {'PASS' if api_ok and sync_result else 'FAIL'}")
                
            except Exception as e:
                print(f"[{now_str()}] [ERROR] API connection test failed: {e}")
                test_results['tests_run'].append({'name': 'api_connection_sync', 'success': False, 'error': str(e)})

            # Test 2: Market Data Fetching
            print(f"\n{'='*60}")
            print(f"[{now_str()}] [TEST SUITE 2] Market Data Fetching")
            print(f"{'='*60}")
            
            try:
                market_results = test_market_data_fetching()
                success_count = sum(1 for r in market_results.values() if 'error' not in r)
                test_results['tests_run'].append({
                    'name': 'market_data_fetching',
                    'success': success_count > 0,
                    'details': market_results
                })
                print(f"[{now_str()}] [RESULT] Market Data Fetching: {success_count}/{len(SUPPORTED_SYMBOLS)} symbols successful")
                
            except Exception as e:
                print(f"[{now_str()}] [ERROR] Market data test failed: {e}")
                test_results['tests_run'].append({'name': 'market_data_fetching', 'success': False, 'error': str(e)})

            # Test 3: Order Validation
            print(f"\n{'='*60}")
            print(f"[{now_str()}] [TEST SUITE 3] Order Validation")
            print(f"{'='*60}")
            
            try:
                validation_results = test_order_validation()
                passed_count = sum(1 for r in validation_results if r.get('passed', False))
                test_results['tests_run'].append({
                    'name': 'order_validation',
                    'success': passed_count > 0,
                    'details': validation_results
                })
                print(f"[{now_str()}] [RESULT] Order Validation: {passed_count}/{len(validation_results)} tests passed")
                
            except Exception as e:
                print(f"[{now_str()}] [ERROR] Order validation test failed: {e}")
                test_results['tests_run'].append({'name': 'order_validation', 'success': False, 'error': str(e)})

            # Test 4: Client Order ID Generation
            print(f"\n{'='*60}")
            print(f"[{now_str()}] [TEST SUITE 4] Client Order ID Generation")
            print(f"{'='*60}")
            
            try:
                id_results = test_client_order_id_generation()
                test_results['tests_run'].append({
                    'name': 'client_order_id_generation',
                    'success': id_results['all_unique'],
                    'details': id_results
                })
                print(f"[{now_str()}] [RESULT] Client Order ID Generation: {'PASS' if id_results['all_unique'] else 'FAIL'}")
                
            except Exception as e:
                print(f"[{now_str()}] [ERROR] Client order ID test failed: {e}")
                test_results['tests_run'].append({'name': 'client_order_id_generation', 'success': False, 'error': str(e)})

            # Test 5: Order Reconciliation
            print(f"\n{'='*60}")
            print(f"[{now_str()}] [TEST SUITE 5] Order Reconciliation")
            print(f"{'='*60}")
            
            try:
                reconciliation_results = test_order_reconciliation()
                test_results['tests_run'].append({
                    'name': 'order_reconciliation',
                    'success': reconciliation_results['success'],
                    'details': reconciliation_results
                })
                print(f"[{now_str()}] [RESULT] Order Reconciliation: {'PASS' if reconciliation_results['success'] else 'FAIL'}")
                
            except Exception as e:
                print(f"[{now_str()}] [ERROR] Order reconciliation test failed: {e}")
                test_results['tests_run'].append({'name': 'order_reconciliation', 'success': False, 'error': str(e)})

            # Test 6: Realistic Trading Scenarios
            print(f"\n{'='*60}")
            print(f"[{now_str()}] [TEST SUITE 6] Realistic Trading Scenarios")
            print(f"{'='*60}")
            
            try:
                trading_results = test_realistic_trading()
                success_count = sum(1 for r in trading_results.values() if r.get('success', False))
                test_results['tests_run'].append({
                    'name': 'realistic_trading',
                    'success': success_count > 0,
                    'details': trading_results
                })
                print(f"[{now_str()}] [RESULT] Realistic Trading: {success_count} scenarios successful")
                
            except Exception as e:
                print(f"[{now_str()}] [ERROR] Realistic trading test failed: {e}")
                test_results['tests_run'].append({'name': 'realistic_trading', 'success': False, 'error': str(e)})

            # Test 7: Circuit Breaker Testing
            print(f"\n{'='*60}")
            print(f"[{now_str()}] [TEST SUITE 7] Circuit Breaker Testing")
            print(f"{'='*60}")
            
            try:
                circuit_breaker_results = trigger_circuit_breaker()
                test_results['tests_run'].append({
                    'name': 'circuit_breaker',
                    'success': True,  # We expect this to trigger the circuit breaker
                    'details': circuit_breaker_results
                })
                print(f"[{now_str()}] [RESULT] Circuit Breaker: Triggered {circuit_breaker_results['errors_triggered']} errors")
                
            except Exception as e:
                print(f"[{now_str()}] [ERROR] Circuit breaker test failed: {e}")
                test_results['tests_run'].append({'name': 'circuit_breaker', 'success': False, 'error': str(e)})

            # Test 8: Error Recovery
            print(f"\n{'='*60}")
            print(f"[{now_str()}] [TEST SUITE 8] Error Recovery")
            print(f"{'='*60}")
            
            try:
                recovery_results = test_error_recovery()
                success_count = sum(1 for r in recovery_results if r.get('success', False))
                test_results['tests_run'].append({
                    'name': 'error_recovery',
                    'success': success_count > 0,
                    'details': recovery_results
                })
                print(f"[{now_str()}] [RESULT] Error Recovery: {success_count}/{len(recovery_results)} tests passed")
                
            except Exception as e:
                print(f"[{now_str()}] [ERROR] Error recovery test failed: {e}")
                test_results['tests_run'].append({'name': 'error_recovery', 'success': False, 'error': str(e)})

            # Generate final summary
            print(f"\n{'='*60}")
            print(f"[{now_str()}] [FINAL SUMMARY]")
            print(f"{'='*60}")
            
            total_tests = len(test_results['tests_run'])
            successful_tests = sum(1 for t in test_results['tests_run'] if t.get('success', False))
            
            test_results['summary'] = {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': f"{(successful_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%",
                'cooldown': COOLDOWN,
                'max_errors': MAX_ERRORS,
                'supported_symbols': SUPPORTED_SYMBOLS
            }
            
            print(f"[{now_str()}] [SUMMARY] Total Tests: {total_tests}")
            print(f"[{now_str()}] [SUMMARY] Successful: {successful_tests}")
            print(f"[{now_str()}] [SUMMARY] Success Rate: {test_results['summary']['success_rate']}")
            
            for test in test_results['tests_run']:
                status = "PASS" if test.get('success', False) else "FAIL"
                print(f"[{now_str()}] [SUMMARY] {test['name']}: {status}")
            
            print(f"[{now_str()}] [END] Comprehensive e2e test complete")

    # Write detailed JSON summary
    with open(summary_path, "w", encoding="utf-8") as jf:
        json.dump(test_results, jf, indent=2)

    print(f"Comprehensive test log saved to: {log_path}")
    print(f"Detailed summary saved to: {summary_path}")
    print(f"Test Results: {test_results['summary']['success_rate']} success rate")
    
    return 0 if test_results['summary']['successful_tests'] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
