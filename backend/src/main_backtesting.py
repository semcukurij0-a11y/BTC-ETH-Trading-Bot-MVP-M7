"""
Main Backtesting Interface for Crypto Trading Bot

This module provides a command-line interface for running comprehensive backtests:
- Vectorized backtesting with realistic costs
- Ablation studies and walk-forward analysis
- Performance evaluation and acceptance gates
- Results visualization and reporting
"""

import argparse
import logging
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.backtesting_engine import BacktestingEngine, BacktestMode
from services.feature_engineering import FeatureEngineer

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main_backtesting")


def load_config(config_path="config.json"):
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def prepare_backtest_data(symbol: str, interval: str, days_back: int = 365):
    """
    Prepare data for backtesting by loading and engineering features.
    
    Args:
        symbol: Trading symbol
        interval: Time interval
        days_back: Days of historical data
        
    Returns:
        DataFrame with features for backtesting
    """
    try:
        logger.info(f"Preparing backtest data for {symbol} {interval}")
        
        # Initialize feature engineer
        fe = FeatureEngineer(
            data_folder="src/data/parquet",
            output_folder="src/data/features",
            log_level="INFO",
            enable_extra_features=True  # Enable all features for backtesting
        )
        
        # Load latest features
        features_df = fe.load_latest_features(symbol, interval)
        
        if features_df is None or features_df.empty:
            logger.warning(f"No features found for {symbol} {interval}")
            return pd.DataFrame()
        
        # Filter to requested period
        logger.info(f"Features DataFrame index type: {type(features_df.index)}")
        logger.info(f"Features DataFrame columns: {list(features_df.columns)}")
        
        # Check if we have open_time column for date filtering
        if 'open_time' in features_df.columns:
            features_df['open_time'] = pd.to_datetime(features_df['open_time'])
            end_date = features_df['open_time'].max()
            start_date = end_date - timedelta(days=days_back)
            features_df = features_df[features_df['open_time'] >= start_date]
            features_df = features_df.set_index('open_time')
        else:
            # Use last N rows if no date column
            features_df = features_df.tail(days_back * 96)  # Approximate 15-minute intervals per day
        
        logger.info(f"Loaded {len(features_df)} data points for backtesting")
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in features_df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error preparing backtest data: {e}")
        return pd.DataFrame()


def run_single_backtest(engine: BacktestingEngine, 
                       data: pd.DataFrame, 
                       mode: BacktestMode,
                       start_date: str = None,
                       end_date: str = None):
    """
    Run a single backtest.
    
    Args:
        engine: Backtesting engine
        data: Input data
        mode: Backtesting mode
        start_date: Start date
        end_date: End date
        
    Returns:
        Backtest results
    """
    try:
        logger.info(f"Running {mode.value} backtest")
        
        result = engine.vectorized_backtest(
            data=data,
            mode=mode,
            start_date=start_date,
            end_date=end_date
        )
        
        # Print summary
        performance = result.get('performance', {})
        print(f"\n=== {mode.value.upper()} BACKTEST RESULTS ===")
        print(f"Period: {result.get('period', {}).get('start', 'N/A')} to {result.get('period', {}).get('end', 'N/A')}")
        print(f"Total Trades: {performance.get('total_trades', 0)}")
        print(f"Hit Rate: {performance.get('hit_rate', 0):.1%}")
        print(f"Total Return: {performance.get('total_return', 0):.1%}")
        print(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.3f}")
        print(f"Max Drawdown: {performance.get('max_drawdown', 0):.1%}")
        print(f"Profit Factor: {performance.get('profit_factor', 0):.2f}")
        print(f"Total Costs: ${performance.get('total_costs', 0):,.0f}")
        
        # Check acceptance gates
        acceptance = result.get('acceptance', {})
        if acceptance:
            print(f"Acceptance Gates: {'✅ PASSED' if acceptance.get('all_gates_passed', False) else '❌ FAILED'}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error running {mode.value} backtest: {e}")
        return {'error': str(e)}


def run_ablation_study(engine: BacktestingEngine, data: pd.DataFrame, test_period: str):
    """
    Run ablation study comparing different signal sources.
    
    Args:
        engine: Backtesting engine
        data: Input data
        test_period: Test period (e.g., "2024-01-01")
        
    Returns:
        Ablation study results
    """
    try:
        logger.info("Running ablation study")
        
        # Define periods (simplified - would normally use proper train/val/test splits)
        train_start = "2023-01-01"
        val_start = "2023-10-01"
        test_start = test_period
        
        result = engine.run_ablation_study(
            data=data,
            train_start=train_start,
            val_start=val_start,
            test_start=test_start,
            test_end="2024-12-31"
        )
        
        # Print comparison table
        if 'comparison' in result and not result['comparison'].empty:
            print("\n=== ABLATION STUDY COMPARISON ===")
            print(result['comparison'].to_string(index=False))
        
        return result
        
    except Exception as e:
        logger.error(f"Error running ablation study: {e}")
        return {'error': str(e)}


def run_walk_forward_analysis(engine: BacktestingEngine, data: pd.DataFrame):
    """
    Run walk-forward analysis.
    
    Args:
        engine: Backtesting engine
        data: Input data
        
    Returns:
        Walk-forward results
    """
    try:
        logger.info("Running walk-forward analysis")
        
        result = engine.walk_forward_analysis(
            data=data,
            train_period=252,  # 1 year
            test_period=63,    # 3 months
            step_size=21       # 1 month steps
        )
        
        # Print summary
        if 'aggregated' in result:
            agg = result['aggregated']
            print(f"\n=== WALK-FORWARD ANALYSIS SUMMARY ===")
            print(f"Total Periods: {agg.get('total_periods', 0)}")
            print(f"Total Trades: {agg.get('total_trades', 0)}")
            print(f"Net PnL: ${agg.get('net_pnl', 0):,.0f}")
            print(f"Average Sharpe: {agg.get('avg_sharpe_ratio', 0):.3f}")
            print(f"Consistency: {agg.get('consistency', 0):.1%}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error running walk-forward analysis: {e}")
        return {'error': str(e)}


def save_results(results: dict, output_file: str):
    """
    Save backtest results to file.
    
    Args:
        results: Backtest results
        output_file: Output file path
    """
    try:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            return obj
        
        # Convert results
        converted_results = json.loads(json.dumps(results, default=convert_numpy))
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(converted_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Futures Backtesting & Evaluation for Crypto Trading Bot")
    parser.add_argument("command", choices=[
        "single", "ablation", "walkforward", "compare"
    ], help="Command to execute")
    
    # Data parameters
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--interval", type=str, default="15m", help="Time interval")
    parser.add_argument("--days-back", type=int, default=365, help="Days of historical data")
    
    # Backtest parameters
    parser.add_argument("--mode", type=str, default="full_system", 
                       choices=["full_system", "ml_only", "ta_only", "sent_only", "fusion_only", "buy_hold"],
                       help="Backtesting mode")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--test-period", type=str, default="2024-01-01", help="Test period start for ablation study")
    
    # Output parameters
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--config", type=str, default="config.json", help="Configuration file")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level.upper()))
    for handler in logging.root.handlers:
        handler.setLevel(getattr(logging, args.log_level.upper()))
    
    # Load configuration
    config = load_config(args.config)
    backtest_config = config.get('backtesting', {
        'initial_balance': 100000.0,
        'leverage': 3.0,
        'fees': {
            'maker': 0.0001,
            'taker': 0.0006,
            'funding_rate': 0.0001,
            'slippage_bps': 2.0,
            'spread_floor_bps': 1.0
        },
        'acceptance_gates': {
            'min_sharpe': 1.0,
            'max_drawdown': 0.12,
            'min_profit_factor': 1.0,
            'min_hit_rate': 0.45
        }
    })
    
    # Initialize backtesting engine
    engine = BacktestingEngine(backtest_config)
    
    # Prepare data
    data = prepare_backtest_data(args.symbol, args.interval, args.days_back)
    if data.empty:
        logger.error("No data available for backtesting")
        sys.exit(1)
    
    logger.info(f"Loaded {len(data)} data points for backtesting")
    
    try:
        if args.command == "single":
            # Run single backtest
            mode = BacktestMode(args.mode)
            result = run_single_backtest(
                engine, data, mode, args.start_date, args.end_date
            )
            
            if args.output:
                save_results(result, args.output)
        
        elif args.command == "ablation":
            # Run ablation study
            result = run_ablation_study(engine, data, args.test_period)
            
            if args.output:
                save_results(result, args.output)
        
        elif args.command == "walkforward":
            # Run walk-forward analysis
            result = run_walk_forward_analysis(engine, data)
            
            if args.output:
                save_results(result, args.output)
        
        elif args.command == "compare":
            # Run all modes and compare
            modes = [
                BacktestMode.BUY_HOLD,
                BacktestMode.ML_ONLY,
                BacktestMode.TA_ONLY,
                BacktestMode.SENT_ONLY,
                BacktestMode.FUSION_ONLY,
                BacktestMode.FULL_SYSTEM
            ]
            
            results = {}
            for mode in modes:
                result = run_single_backtest(
                    engine, data, mode, args.start_date, args.end_date
                )
                results[mode.value] = result
            
            # Create summary comparison
            comparison_data = []
            for mode, result in results.items():
                perf = result.get('performance', {})
                comparison_data.append({
                    'Mode': mode,
                    'Trades': perf.get('total_trades', 0),
                    'Hit Rate': f"{perf.get('hit_rate', 0):.1%}",
                    'Return': f"{perf.get('total_return', 0):.1%}",
                    'Sharpe': f"{perf.get('sharpe_ratio', 0):.3f}",
                    'Max DD': f"{perf.get('max_drawdown', 0):.1%}",
                    'Profit Factor': f"{perf.get('profit_factor', 0):.2f}",
                    'Costs': f"${perf.get('total_costs', 0):,.0f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            print("\n=== COMPREHENSIVE COMPARISON ===")
            print(comparison_df.to_string(index=False))
            
            if args.output:
                save_results(results, args.output)
        
        logger.info("Backtesting completed successfully")
        
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
