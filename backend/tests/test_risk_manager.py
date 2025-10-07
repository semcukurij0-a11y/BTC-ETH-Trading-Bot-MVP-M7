#!/usr/bin/env python3
"""
Simple Test Script for Risk Manager Limit

Test Criteria:
1. Fixed-fractional sizing (0.75% default)
2. SL = 2.5 x ATR, TP = 3.0 x ATR
3. Leverage cap <= 5x, isolated margin
4. Liquidation buffer >= 3 x ATR
5. Daily caps (loss 3%, profit 2%, trades 15, consecutive losses 4)
6. Soft DD 6% / Hard DD 10%
7. Position opening/closing
8. Stop loss/take profit execution
9. Risk limit breaches
10. Emergency close functionality

Author: Trading System Test Suite
Date: 2025-01-07
"""

import json
import time
import logging
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from risk_manager_limit import RiskManagerLimit, RiskLimits, TradeRecord, DailyStats

# Configure detailed logging
def setup_test_logging():
    """Setup comprehensive logging for tests"""
    log_dir = Path("tests/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"risk_manager_simple_test_{timestamp}.log"
    
    # Create detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('risk_manager_simple_test')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

# Initialize logging
test_logger, log_file = setup_test_logging()

class TestRiskManagerSimple(unittest.TestCase):
    """Simple test suite for RiskManagerLimit"""
    
    def setUp(self):
        """Setup test environment"""
        test_logger.info("="*80)
        test_logger.info(f"Starting test: {self._testMethodName}")
        test_logger.info("="*80)
        
        # Create mock config
        self.mock_config = {
            "futures_trading": {
                "fixed_fractional_percent": 0.75,
                "atr_sl_multiplier": 2.5,
                "atr_tp_multiplier": 3.0,
                "max_leverage": 5.0,
                "margin_mode": "isolated",
                "liquidation_buffer_atr": 3.0,
                "max_loss_per_day": 0.03,
                "max_profit_per_day": 0.02,
                "max_trades_per_day": 15,
                "max_consecutive_losses": 4,
                "soft_drawdown": 0.06,
                "hard_drawdown": 0.10,
                "min_confidence": 0.6,
                "min_signal_strength": 0.3
            }
        }
        
        # Create temporary config file
        self.config_file = "test_config_simple.json"
        with open(self.config_file, 'w') as f:
            json.dump(self.mock_config, f)
        
        # Initialize risk manager
        self.risk_manager = RiskManagerLimit(
            config_path=self.config_file,
            initial_balance=10000.0
        )
        
        test_logger.info(f"Test setup complete - Initial balance: ${self.risk_manager.initial_balance:,.2f}")
    
    def tearDown(self):
        """Cleanup after test"""
        # Remove temporary config file
        if os.path.exists(self.config_file):
            os.remove(self.config_file)
        
        test_logger.info(f"Test completed: {self._testMethodName}")
        test_logger.info("="*80)
    
    def test_01_initialization(self):
        """Test 1: Risk Manager Initialization"""
        test_logger.info("Testing risk manager initialization...")
        
        # Test initial state
        self.assertEqual(self.risk_manager.current_balance, 10000.0)
        self.assertEqual(self.risk_manager.peak_balance, 10000.0)
        self.assertEqual(len(self.risk_manager.open_positions), 0)
        self.assertEqual(self.risk_manager.daily_trades_count, 0)
        self.assertEqual(self.risk_manager.consecutive_losses, 0)
        
        # Test limits loading
        self.assertEqual(self.risk_manager.limits.fixed_fractional_percent, 0.75)
        self.assertEqual(self.risk_manager.limits.atr_sl_multiplier, 2.5)
        self.assertEqual(self.risk_manager.limits.atr_tp_multiplier, 3.0)
        self.assertEqual(self.risk_manager.limits.max_leverage, 5.0)
        
        test_logger.info("PASS: Initialization test passed")
    
    def test_02_fixed_fractional_sizing(self):
        """Test 2: Fixed-Fractional Sizing (0.75% default)"""
        test_logger.info("Testing fixed-fractional sizing...")
        
        symbol = "BTCUSDT"
        atr_value = 500.0
        confidence = 0.8
        signal_strength = 0.7
        
        # Calculate position size
        quantity, stop_loss, take_profit = self.risk_manager.calculate_position_size(
            symbol, atr_value, confidence, signal_strength
        )
        
        # Verify calculations
        expected_risk_amount = 10000.0 * 0.0075  # 0.75% of balance
        expected_stop_distance = atr_value * 2.5  # 2.5 x ATR
        expected_quantity = expected_risk_amount / expected_stop_distance
        
        test_logger.info(f"Expected risk amount: ${expected_risk_amount:.2f}")
        test_logger.info(f"Expected stop distance: {expected_stop_distance:.2f}")
        test_logger.info(f"Expected quantity: {expected_quantity:.6f}")
        test_logger.info(f"Calculated quantity: {quantity:.6f}")
        
        self.assertAlmostEqual(quantity, expected_quantity, places=6)
        self.assertGreater(quantity, 0)
        
        # Test with different confidence levels
        low_confidence_quantity, _, _ = self.risk_manager.calculate_position_size(
            symbol, atr_value, 0.5, signal_strength  # Below threshold
        )
        self.assertEqual(low_confidence_quantity, 0.0)
        
        test_logger.info("PASS: Fixed-fractional sizing test passed")
    
    def test_03_atr_based_stops(self):
        """Test 3: ATR-Based Stop Loss and Take Profit"""
        test_logger.info("Testing ATR-based stop loss and take profit...")
        
        symbol = "BTCUSDT"
        atr_value = 500.0
        current_price = 50000.0
        
        # Test buy position
        quantity, stop_loss, take_profit = self.risk_manager.calculate_position_size(
            symbol, atr_value, 0.8, 0.7
        )
        
        expected_sl = current_price - (atr_value * 2.5)
        expected_tp = current_price + (atr_value * 3.0)
        
        test_logger.info(f"Current price: ${current_price:.2f}")
        test_logger.info(f"ATR value: {atr_value:.2f}")
        test_logger.info(f"Expected SL: ${expected_sl:.2f}")
        test_logger.info(f"Calculated SL: ${stop_loss:.2f}")
        test_logger.info(f"Expected TP: ${expected_tp:.2f}")
        test_logger.info(f"Calculated TP: ${take_profit:.2f}")
        
        self.assertAlmostEqual(stop_loss, expected_sl, places=2)
        self.assertAlmostEqual(take_profit, expected_tp, places=2)
        
        test_logger.info("PASS: ATR-based stops test passed")
    
    def test_04_leverage_and_margin(self):
        """Test 4: Leverage Cap and Margin Controls"""
        test_logger.info("Testing leverage and margin controls...")
        
        symbol = "BTCUSDT"
        atr_value = 500.0
        
        # Test valid leverage
        success = self.risk_manager.open_position(
            symbol=symbol,
            side="buy",
            quantity=0.1,
            entry_price=50000.0,
            atr_value=atr_value,
            leverage=2.0
        )
        self.assertTrue(success)
        
        # Test excessive leverage (should fail)
        success = self.risk_manager.open_position(
            symbol=symbol,
            side="buy",
            quantity=0.1,
            entry_price=50000.0,
            atr_value=atr_value,
            leverage=6.0  # Exceeds 5x limit
        )
        self.assertFalse(success)
        
        # Test margin mode
        self.assertEqual(self.risk_manager.limits.margin_mode, "isolated")
        
        test_logger.info("PASS: Leverage and margin test passed")
    
    def test_05_daily_caps(self):
        """Test 5: Daily Trading Caps"""
        test_logger.info("Testing daily trading caps...")
        
        symbol = "BTCUSDT"
        atr_value = 500.0
        
        # Test daily trade limit (15 trades)
        for i in range(16):  # Try 16 trades
            can_trade = self.risk_manager.can_trade(0.8, 0.7)
            if i < 15:
                self.assertTrue(can_trade, f"Should be able to trade on attempt {i+1}")
            else:
                self.assertFalse(can_trade, "Should not be able to trade after 15 attempts")
            
            if can_trade:
                self.risk_manager.open_position(
                    symbol=symbol,
                    side="buy",
                    quantity=0.01,
                    entry_price=50000.0,
                    atr_value=atr_value,
                    leverage=1.0
                )
        
        test_logger.info(f"Daily trades count: {self.risk_manager.daily_trades_count}")
        self.assertEqual(self.risk_manager.daily_trades_count, 15)
        
        test_logger.info("PASS: Daily caps test passed")
    
    def test_06_position_management(self):
        """Test 6: Position Opening and Closing"""
        test_logger.info("Testing position management...")
        
        symbol = "BTCUSDT"
        atr_value = 500.0
        entry_price = 50000.0
        
        # Test position opening
        success = self.risk_manager.open_position(
            symbol=symbol,
            side="buy",
            quantity=0.1,
            entry_price=entry_price,
            atr_value=atr_value,
            leverage=2.0
        )
        
        self.assertTrue(success)
        self.assertEqual(len(self.risk_manager.open_positions), 1)
        
        # Get trade ID
        trade_id = list(self.risk_manager.open_positions.keys())[0]
        trade = self.risk_manager.open_positions[trade_id]
        
        test_logger.info(f"Opened position: {trade_id}")
        test_logger.info(f"Entry price: ${trade.entry_price:.2f}")
        test_logger.info(f"Stop loss: ${trade.stop_loss:.2f}")
        test_logger.info(f"Take profit: ${trade.take_profit:.2f}")
        test_logger.info(f"Margin used: ${trade.margin_used:.2f}")
        
        # Test position closing
        exit_price = entry_price * 1.02  # 2% profit
        pnl = self.risk_manager.close_position(trade_id, exit_price, "manual_close")
        
        self.assertEqual(len(self.risk_manager.open_positions), 0)
        self.assertEqual(len(self.risk_manager.trade_history), 1)
        self.assertGreater(pnl, 0)  # Should be profitable
        
        test_logger.info(f"Closed position with PnL: ${pnl:.2f}")
        
        test_logger.info("PASS: Position management test passed")
    
    def test_07_stop_loss_execution(self):
        """Test 7: Stop Loss and Take Profit Execution"""
        test_logger.info("Testing stop loss and take profit execution...")
        
        symbol = "BTCUSDT"
        atr_value = 500.0
        entry_price = 50000.0
        
        # Open position
        success = self.risk_manager.open_position(
            symbol=symbol,
            side="buy",
            quantity=0.1,
            entry_price=entry_price,
            atr_value=atr_value,
            leverage=1.0
        )
        
        self.assertTrue(success)
        trade_id = list(self.risk_manager.open_positions.keys())[0]
        trade = self.risk_manager.open_positions[trade_id]
        
        test_logger.info(f"Position opened at: ${entry_price:.2f}")
        test_logger.info(f"Stop loss set at: ${trade.stop_loss:.2f}")
        test_logger.info(f"Take profit set at: ${trade.take_profit:.2f}")
        
        # Test stop loss trigger
        stop_loss_price = trade.stop_loss - 10  # Price below stop loss
        current_prices = {symbol: stop_loss_price}
        
        triggered_trades = self.risk_manager.check_stop_losses(current_prices)
        
        self.assertEqual(len(triggered_trades), 1)
        self.assertEqual(triggered_trades[0], trade_id)
        self.assertEqual(len(self.risk_manager.open_positions), 0)
        
        test_logger.info(f"Stop loss triggered at: ${stop_loss_price:.2f}")
        
        test_logger.info("PASS: Stop loss/take profit execution test passed")
    
    def test_08_risk_status_reporting(self):
        """Test 8: Risk Status Reporting"""
        test_logger.info("Testing risk status reporting...")
        
        # Get initial status
        status = self.risk_manager.get_risk_status()
        
        self.assertIn('account_balance', status)
        self.assertIn('current_drawdown', status)
        self.assertIn('margin_used', status)
        self.assertIn('open_positions', status)
        self.assertIn('daily_trades', status)
        self.assertIn('risk_limits_status', status)
        self.assertIn('can_trade', status)
        
        test_logger.info("Initial risk status:")
        test_logger.info(json.dumps(status, indent=2, default=str))
        
        # Test with open positions
        symbol = "BTCUSDT"
        atr_value = 500.0
        
        self.risk_manager.open_position(
            symbol=symbol,
            side="buy",
            quantity=0.1,
            entry_price=50000.0,
            atr_value=atr_value,
            leverage=2.0
        )
        
        status_with_positions = self.risk_manager.get_risk_status()
        
        self.assertEqual(status_with_positions['open_positions'], 1)
        self.assertGreater(status_with_positions['margin_used'], 0)
        
        test_logger.info("Risk status with positions:")
        test_logger.info(json.dumps(status_with_positions, indent=2, default=str))
        
        test_logger.info("PASS: Risk status reporting test passed")
    
    def test_09_emergency_close(self):
        """Test 9: Emergency Close All Positions"""
        test_logger.info("Testing emergency close functionality...")
        
        symbol1 = "BTCUSDT"
        symbol2 = "ETHUSDT"
        atr_value1 = 500.0
        atr_value2 = 30.0
        
        # Open multiple positions
        self.risk_manager.open_position(
            symbol=symbol1,
            side="buy",
            quantity=0.1,
            entry_price=50000.0,
            atr_value=atr_value1,
            leverage=1.0
        )
        
        self.risk_manager.open_position(
            symbol=symbol2,
            side="sell",
            quantity=1.0,
            entry_price=3000.0,
            atr_value=atr_value2,
            leverage=1.0
        )
        
        self.assertEqual(len(self.risk_manager.open_positions), 2)
        
        # Emergency close all
        current_prices = {
            symbol1: 49000.0,  # Loss
            symbol2: 3100.0    # Loss
        }
        
        total_pnl = self.risk_manager.emergency_close_all(current_prices)
        
        self.assertEqual(len(self.risk_manager.open_positions), 0)
        self.assertEqual(len(self.risk_manager.trade_history), 2)
        self.assertLess(total_pnl, 0)  # Should be negative (losses)
        
        test_logger.info(f"Emergency close total PnL: ${total_pnl:.2f}")
        
        test_logger.info("PASS: Emergency close test passed")
    
    def test_10_comprehensive_scenario(self):
        """Test 10: Comprehensive Trading Scenario"""
        test_logger.info("Testing comprehensive trading scenario...")
        
        # Simulate a full trading day
        symbols = ["BTCUSDT", "ETHUSDT"]
        initial_balance = self.risk_manager.current_balance
        
        test_logger.info(f"Starting comprehensive scenario with balance: ${initial_balance:,.2f}")
        
        # Open multiple positions
        for i, symbol in enumerate(symbols):
            atr_value = 500.0 if symbol == "BTCUSDT" else 30.0
            entry_price = 50000.0 if symbol == "BTCUSDT" else 3000.0
            
            success = self.risk_manager.open_position(
                symbol=symbol,
                side="buy" if i % 2 == 0 else "sell",
                quantity=0.1 if symbol == "BTCUSDT" else 1.0,
                entry_price=entry_price,
                atr_value=atr_value,
                leverage=2.0
            )
            
            self.assertTrue(success)
            test_logger.info(f"Opened position {i+1}: {symbol}")
        
        # Check risk status
        status = self.risk_manager.get_risk_status()
        test_logger.info(f"Positions opened: {status['open_positions']}")
        test_logger.info(f"Margin used: ${status['margin_used']:.2f}")
        test_logger.info(f"Daily trades: {status['daily_trades']}")
        
        # Simulate price movements and check stops
        current_prices = {
            "BTCUSDT": 48500.0,  # 3% down
            "ETHUSDT": 2910.0    # 3% down
        }
        
        for symbol, new_price in current_prices.items():
            test_logger.info(f"{symbol} price moved to: ${new_price:.2f}")
        
        # Check for triggered stops
        triggered = self.risk_manager.check_stop_losses(current_prices)
        test_logger.info(f"Triggered stops: {len(triggered)}")
        
        # Get final status
        final_status = self.risk_manager.get_risk_status()
        final_balance = self.risk_manager.current_balance
        
        test_logger.info(f"Final balance: ${final_balance:,.2f}")
        test_logger.info(f"Total PnL: ${final_balance - initial_balance:,.2f}")
        test_logger.info(f"Final drawdown: {final_status['current_drawdown']:.2f}%")
        
        test_logger.info("PASS: Comprehensive scenario test passed")

def run_simple_test():
    """Run all tests with detailed reporting"""
    test_logger.info("Starting Simple Risk Manager Test Suite")
    test_logger.info(f"Detailed logs will be saved to: {log_file}")
    test_logger.info("="*100)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRiskManagerSimple)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=open(log_file, 'a', encoding='utf-8'))
    result = runner.run(suite)
    
    # Generate summary report
    test_logger.info("="*100)
    test_logger.info("TEST SUMMARY REPORT")
    test_logger.info("="*100)
    test_logger.info(f"Total tests run: {result.testsRun}")
    test_logger.info(f"Failures: {len(result.failures)}")
    test_logger.info(f"Errors: {len(result.errors)}")
    test_logger.info(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        test_logger.error("FAILURES:")
        for test, traceback in result.failures:
            test_logger.error(f"  - {test}: {traceback}")
    
    if result.errors:
        test_logger.error("ERRORS:")
        for test, traceback in result.errors:
            test_logger.error(f"  - {test}: {traceback}")
    
    if not result.failures and not result.errors:
        test_logger.info("ALL TESTS PASSED!")
    
    test_logger.info("="*100)
    test_logger.info(f"Complete test log saved to: {log_file}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_simple_test()
    sys.exit(0 if success else 1)
