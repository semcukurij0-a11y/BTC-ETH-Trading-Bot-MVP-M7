"""
Main Futures Trading Interface for Crypto Trading Bot

This module provides a command-line interface for futures trading operations:
- Trade execution with risk management
- Portfolio management and monitoring
- Performance tracking and reporting
- Emergency controls and kill-switch
"""

import argparse
import logging
import sys
import os
import json
import asyncio
from datetime import datetime, timedelta

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.futures_trading import FuturesTradingModule, ExecutionMode
from services.risk_management import RiskManagementModule
from services.order_execution import OrderExecutionModule

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main_futures")


def load_config(config_path="config.json"):
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        return json.load(f)


async def execute_trade_signal(futures_module, symbol, s_fused, confidence, signal_strength, price, atr):
    """Execute a trade signal."""
    try:
        signal = {
            'symbol': symbol,
            's': s_fused,
            'confidence': confidence,
            'signal_strength': signal_strength
        }
        
        market_data = {
            'price': price,
            'atr': atr
        }
        
        # Update market data
        await futures_module.update_market_data(symbol, market_data)
        
        # Execute trade signal
        result = await futures_module.execute_trade_signal(signal, market_data)
        
        if result['success']:
            logger.info(f"Trade executed successfully: {result['trade_id']}")
            return result
        else:
            logger.warning(f"Trade execution failed: {result['error']}")
            return result
            
    except Exception as e:
        logger.error(f"Error executing trade signal: {e}")
        return {'success': False, 'error': str(e)}


async def monitor_portfolio(futures_module, duration_minutes=60):
    """Monitor portfolio for specified duration."""
    try:
        logger.info(f"Starting portfolio monitoring for {duration_minutes} minutes...")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time:
            # Get portfolio summary
            summary = futures_module.get_portfolio_summary()
            
            # Log key metrics
            portfolio = summary.get('portfolio', {})
            performance = summary.get('performance', {})
            
            logger.info(f"Portfolio Status - Balance: {portfolio.get('balance', 0):.2f}, "
                       f"Equity: {portfolio.get('equity', 0):.2f}, "
                       f"Unrealized PnL: {portfolio.get('unrealized_pnl', 0):.2f}")
            
            logger.info(f"Performance - Total Trades: {performance.get('total_trades', 0)}, "
                       f"Win Rate: {performance.get('win_rate', 0):.2%}, "
                       f"Total Return: {performance.get('total_return', 0):.2%}")
            
            # Check for emergency conditions
            risk_summary = summary.get('risk_summary', {})
            if risk_summary.get('kill_switch_triggered', False):
                logger.critical("Kill-switch triggered! Stopping monitoring.")
                break
            
            # Wait before next update
            await asyncio.sleep(60)  # Update every minute
        
        logger.info("Portfolio monitoring completed.")
        
    except Exception as e:
        logger.error(f"Error during portfolio monitoring: {e}")


async def emergency_stop(futures_module, reason="Manual emergency stop"):
    """Trigger emergency stop."""
    try:
        logger.critical(f"Triggering emergency stop: {reason}")
        await futures_module.emergency_stop(reason)
        logger.info("Emergency stop completed.")
        
    except Exception as e:
        logger.error(f"Error during emergency stop: {e}")


async def test_trading_system(futures_module):
    """Test the trading system with sample data."""
    try:
        logger.info("Testing trading system...")
        
        # Test 1: Long signal
        logger.info("Test 1: Long signal")
        result1 = await execute_trade_signal(
            futures_module, 
            "BTCUSDT", 
            s_fused=0.5, 
            confidence=0.7, 
            signal_strength=0.4, 
            price=50000, 
            atr=500
        )
        
        # Test 2: Short signal
        logger.info("Test 2: Short signal")
        result2 = await execute_trade_signal(
            futures_module, 
            "BTCUSDT", 
            s_fused=-0.5, 
            confidence=0.8, 
            signal_strength=0.5, 
            price=49000, 
            atr=450
        )
        
        # Test 3: Weak signal (should be rejected)
        logger.info("Test 3: Weak signal (should be rejected)")
        result3 = await execute_trade_signal(
            futures_module, 
            "BTCUSDT", 
            s_fused=0.1, 
            confidence=0.3, 
            signal_strength=0.2, 
            price=49500, 
            atr=475
        )
        
        # Get final portfolio summary
        summary = futures_module.get_portfolio_summary()
        logger.info(f"Final portfolio summary: {summary}")
        
        return {
            'test1': result1,
            'test2': result2,
            'test3': result3,
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"Error testing trading system: {e}")
        return {'error': str(e)}


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Futures Trading Operations for Crypto Trading Bot")
    parser.add_argument("command", choices=[
        "trade", "monitor", "emergency", "test", "status", "portfolio"
    ], help="Command to execute")
    
    # Trade execution arguments
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--signal", type=float, help="Fused signal value (-1 to +1)")
    parser.add_argument("--confidence", type=float, help="Signal confidence (0 to 1)")
    parser.add_argument("--strength", type=float, help="Signal strength (0 to 1)")
    parser.add_argument("--price", type=float, help="Current market price")
    parser.add_argument("--atr", type=float, help="Current ATR value")
    
    # Monitoring arguments
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in minutes")
    
    # Emergency arguments
    parser.add_argument("--reason", type=str, default="Manual emergency stop", help="Emergency stop reason")
    
    # General arguments
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--mode", type=str, default="simulation", choices=["simulation", "paper", "live"], 
                       help="Execution mode")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                       help="Set the logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level.upper()))
    for handler in logging.root.handlers:
        handler.setLevel(getattr(logging, args.log_level.upper()))
    
    # Load configuration
    config = load_config(args.config)
    futures_config = config.get('futures_trading', {})
    
    # Set execution mode
    execution_mode = ExecutionMode.SIMULATION
    if args.mode == "paper":
        execution_mode = ExecutionMode.PAPER_TRADING
    elif args.mode == "live":
        execution_mode = ExecutionMode.LIVE_TRADING
    
    # Initialize futures trading module
    futures_module = FuturesTradingModule(
        config=futures_config,
        execution_mode=execution_mode
    )
    
    async def run_command():
        try:
            if args.command == "trade":
                if not all([args.signal, args.confidence, args.strength, args.price, args.atr]):
                    logger.error("Trade command requires --signal, --confidence, --strength, --price, and --atr")
                    return
                
                result = await execute_trade_signal(
                    futures_module,
                    args.symbol,
                    args.signal,
                    args.confidence,
                    args.strength,
                    args.price,
                    args.atr
                )
                logger.info(f"Trade result: {result}")
            
            elif args.command == "monitor":
                await monitor_portfolio(futures_module, args.duration)
            
            elif args.command == "emergency":
                await emergency_stop(futures_module, args.reason)
            
            elif args.command == "test":
                result = await test_trading_system(futures_module)
                logger.info(f"Test result: {result}")
            
            elif args.command == "status":
                summary = futures_module.get_portfolio_summary()
                logger.info(f"Portfolio status: {json.dumps(summary, indent=2, default=str)}")
            
            elif args.command == "portfolio":
                summary = futures_module.get_portfolio_summary()
                portfolio = summary.get('portfolio', {})
                performance = summary.get('performance', {})
                
                print("\n=== PORTFOLIO SUMMARY ===")
                print(f"Balance: ${portfolio.get('balance', 0):,.2f}")
                print(f"Equity: ${portfolio.get('equity', 0):,.2f}")
                print(f"Margin Used: ${portfolio.get('margin_used', 0):,.2f}")
                print(f"Free Margin: ${portfolio.get('free_margin', 0):,.2f}")
                print(f"Unrealized PnL: ${portfolio.get('unrealized_pnl', 0):,.2f}")
                print(f"Realized PnL: ${portfolio.get('realized_pnl', 0):,.2f}")
                
                print("\n=== PERFORMANCE METRICS ===")
                print(f"Total Trades: {performance.get('total_trades', 0)}")
                print(f"Winning Trades: {performance.get('winning_trades', 0)}")
                print(f"Losing Trades: {performance.get('losing_trades', 0)}")
                print(f"Win Rate: {performance.get('win_rate', 0):.2%}")
                print(f"Average Win: ${performance.get('average_win', 0):,.2f}")
                print(f"Average Loss: ${performance.get('average_loss', 0):,.2f}")
                print(f"Profit Factor: {performance.get('profit_factor', 0):.2f}")
                print(f"Total Return: {performance.get('total_return', 0):.2%}")
                
                print("\n=== TRADING STATE ===")
                trading_state = summary.get('trading_state', {})
                print(f"Trading Enabled: {trading_state.get('trading_enabled', False)}")
                print(f"Emergency Stop: {trading_state.get('emergency_stop', False)}")
                print(f"Execution Mode: {trading_state.get('execution_mode', 'unknown')}")
                
                print("\n=== ACTIVE POSITIONS ===")
                positions = summary.get('positions', {})
                if positions.get('total', 0) > 0:
                    for pos in positions.get('details', []):
                        print(f"Symbol: {pos.get('symbol', 'N/A')}")
                        print(f"Side: {pos.get('side', 'N/A')}")
                        print(f"Quantity: {pos.get('quantity', 0)}")
                        print(f"Entry Price: ${pos.get('entry_price', 0):,.2f}")
                        print(f"Current Price: ${pos.get('current_price', 0):,.2f}")
                        print(f"Unrealized PnL: ${pos.get('unrealized_pnl', 0):,.2f}")
                        print("---")
                else:
                    print("No active positions")
        
        finally:
            await futures_module.close()
    
    # Run the command
    asyncio.run(run_command())


if __name__ == "__main__":
    main()




