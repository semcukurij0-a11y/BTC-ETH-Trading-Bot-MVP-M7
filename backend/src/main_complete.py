"""
Complete Trading System Orchestrator

This script orchestrates the complete trading system including:
- Data ingestion
- Feature engineering
- ML training/inference
- Technical analysis
- Sentiment analysis
- Signal fusion
- Trading logic
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.data_ingestor import DataIngestor
from services.feature_engineering import FeatureEngineer
from services.ml_trainer import MLTrainer
from services.ml_inference import MLInference
from services.technical_analysis import TechnicalAnalysisModule
from services.sentiment_analysis import SentimentAnalysisModule
from services.signal_fusion import SignalFusionModule
from services.trading_logic import TradingLogicModule


class CompleteTradingSystem:
    """
    Complete trading system orchestrator.
    
    Integrates all modules:
    - Data pipeline
    - ML pipeline
    - Technical analysis
    - Sentiment analysis
    - Signal fusion
    - Trading logic
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the complete trading system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.get('log_level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize modules
        self._initialize_modules()
        
        # System state
        self.last_run_time = None
        self.system_status = {
            'data_pipeline': 'idle',
            'ml_pipeline': 'idle',
            'signal_generation': 'idle',
            'trading_logic': 'idle'
        }
    
    def _initialize_modules(self):
        """Initialize all system modules."""
        self.logger.info("Initializing trading system modules...")
        
        # Data pipeline modules
        self.data_ingestor = DataIngestor(
            parquet_folder=self.config.get('parquet_folder', 'data/parquet'),
            revalidation_bars=self.config.get('revalidation_bars', 5)
        )
        
        self.feature_engineer = FeatureEngineer(
            data_folder=self.config.get('parquet_folder', 'data/parquet'),
            output_folder=self.config.get('features_folder', 'data/features'),
            enable_extra_features=self.config.get('enable_extra_features', False)
        )
        
        # ML modules
        ml_config = self.config.get('ml', {})
        self.ml_trainer = MLTrainer(config=ml_config)
        self.ml_inference = MLInference("src/models", ml_config)
        
        # Analysis modules
        ta_config = self.config.get('technical_analysis', {})
        self.technical_analysis = TechnicalAnalysisModule(
            config=ta_config,
            enable_elliott_wave=ta_config.get('enable_elliott_wave', False)
        )
        
        sentiment_config = self.config.get('sentiment_analysis', {})
        self.sentiment_analysis = SentimentAnalysisModule(
            config=sentiment_config,
            enable_social_media=sentiment_config.get('enable_social_media', True),
            enable_news_analysis=sentiment_config.get('enable_news_analysis', True)
        )
        
        # Signal fusion and trading logic
        fusion_config = self.config.get('signal_fusion', {})
        self.signal_fusion = SignalFusionModule(config=fusion_config)
        
        trading_config = self.config.get('trading_logic', {})
        self.trading_logic = TradingLogicModule(config=trading_config)
        
        self.logger.info("All modules initialized successfully")
    
    def run_data_pipeline(self, symbols: List[str] = None, intervals: List[str] = None) -> Dict[str, Any]:
        """
        Run the complete data pipeline.
        
        Args:
            symbols: List of trading symbols
            intervals: List of time intervals
            
        Returns:
            Pipeline results
        """
        self.logger.info("Starting data pipeline...")
        self.system_status['data_pipeline'] = 'running'
        
        symbols = symbols or self.config.get('symbols', ['BTCUSDT'])
        intervals = intervals or self.config.get('intervals', ['15m'])
        
        results = {
            'data_ingestion': {},
            'feature_engineering': {},
            'total_records': 0,
            'success': True,
            'errors': []
        }
        
        try:
            # Data ingestion
            for symbol in symbols:
                for interval in intervals:
                    self.logger.info(f"Processing {symbol} {interval}")
                    
                    # Fetch and save data
                    ingestion_result = self.data_ingestor.fetch_and_save_data(
                        symbol=symbol,
                        interval=interval,
                        ohlcv_limit=self.config.get('ohlcv_limit', 1000),
                        funding_limit=self.config.get('funding_limit', 500),
                        incremental=True,
                        days_back=self.config.get('days_back', 60)
                    )
                    
                    results['data_ingestion'][f"{symbol}_{interval}"] = ingestion_result
                    
                    # Feature engineering
                    feature_result = self.feature_engineer.create_feature_matrix(
                        symbol=symbol,
                        interval=interval,
                        days_back=self.config.get('days_back', 60)
                    )
                    
                    results['feature_engineering'][f"{symbol}_{interval}"] = feature_result
                    results['total_records'] += feature_result.get('total_records', 0)
            
            self.system_status['data_pipeline'] = 'completed'
            self.logger.info(f"Data pipeline completed successfully. Total records: {results['total_records']}")
            
        except Exception as e:
            self.logger.error(f"Data pipeline failed: {e}")
            results['success'] = False
            results['errors'].append(str(e))
            self.system_status['data_pipeline'] = 'failed'
        
        return results
    
    def run_ml_pipeline(self, symbols: List[str] = None, intervals: List[str] = None) -> Dict[str, Any]:
        """
        Run the ML pipeline (training and inference).
        
        Args:
            symbols: List of trading symbols
            intervals: List of time intervals
            
        Returns:
            ML pipeline results
        """
        self.logger.info("Starting ML pipeline...")
        self.system_status['ml_pipeline'] = 'running'
        
        symbols = symbols or self.config.get('symbols', ['BTCUSDT'])
        intervals = intervals or self.config.get('intervals', ['15m'])
        
        results = {
            'training': {},
            'inference': {},
            'success': True,
            'errors': []
        }
        
        try:
            ml_config = self.config.get('ml', {})
            
            for symbol in symbols:
                for interval in intervals:
                    self.logger.info(f"ML processing {symbol} {interval}")
                    
                    # Check if retraining is needed
                    if self.ml_trainer.should_retrain(symbol, interval):
                        # Train model
                        training_result = self.ml_trainer.train_model(
                            symbol=symbol,
                            interval=interval,
                            days_back=ml_config.get('days_back', 30),
                            target_hours=ml_config.get('target_hours', 1),
                            threshold=ml_config.get('threshold', 0.001)
                        )
                        
                        results['training'][f"{symbol}_{interval}"] = training_result
                        self.logger.info(f"Model trained for {symbol} {interval}")
                    
                    # Run inference
                    try:
                        inference_result = self.ml_inference.get_latest_prediction(symbol, interval)
                        results['inference'][f"{symbol}_{interval}"] = inference_result
                    except Exception as e:
                        self.logger.warning(f"Inference failed for {symbol} {interval}: {e}")
                        results['inference'][f"{symbol}_{interval}"] = {'error': str(e)}
            
            self.system_status['ml_pipeline'] = 'completed'
            self.logger.info("ML pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"ML pipeline failed: {e}")
            results['success'] = False
            results['errors'].append(str(e))
            self.system_status['ml_pipeline'] = 'failed'
        
        return results
    
    def run_signal_generation(self, symbols: List[str] = None, intervals: List[str] = None) -> Dict[str, Any]:
        """
        Run signal generation pipeline (technical analysis, sentiment, fusion).
        
        Args:
            symbols: List of trading symbols
            intervals: List of time intervals
            
        Returns:
            Signal generation results
        """
        self.logger.info("Starting signal generation...")
        self.system_status['signal_generation'] = 'running'
        
        symbols = symbols or self.config.get('symbols', ['BTCUSDT'])
        intervals = intervals or self.config.get('intervals', ['15m'])
        
        results = {
            'technical_analysis': {},
            'sentiment_analysis': {},
            'signal_fusion': {},
            'success': True,
            'errors': []
        }
        
        try:
            for symbol in symbols:
                for interval in intervals:
                    self.logger.info(f"Generating signals for {symbol} {interval}")
                    
                    # Load features data
                    features_df = self.feature_engineer.load_latest_features(symbol, interval)
                    if features_df is None or features_df.empty:
                        self.logger.warning(f"No features available for {symbol} {interval}")
                        continue
                    
                    # Technical analysis
                    ta_result = self.technical_analysis.analyze_technical_signals(features_df)
                    results['technical_analysis'][f"{symbol}_{interval}"] = ta_result
                    
                    # Sentiment analysis
                    sentiment_result = self.sentiment_analysis.generate_s_sent_signal(features_df, symbol)
                    results['sentiment_analysis'][f"{symbol}_{interval}"] = sentiment_result
                    
                    # Combine all signals
                    combined_df = sentiment_result.copy()
                    combined_df['s_ml'] = 0.0  # Placeholder - would come from ML inference
                    combined_df['s_ta'] = ta_result['s_ta']
                    combined_df['s_sent'] = sentiment_result['s_sent']
                    combined_df['fear_greed'] = sentiment_result.get('fear_greed', 0.5)
                    
                    # Signal fusion
                    fusion_result = self.signal_fusion.fuse_signals(combined_df)
                    results['signal_fusion'][f"{symbol}_{interval}"] = fusion_result
            
            self.system_status['signal_generation'] = 'completed'
            self.logger.info("Signal generation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            results['success'] = False
            results['errors'].append(str(e))
            self.system_status['signal_generation'] = 'failed'
        
        return results
    
    def run_trading_logic(self, symbols: List[str] = None, intervals: List[str] = None) -> Dict[str, Any]:
        """
        Run trading logic and generate trade decisions.
        
        Args:
            symbols: List of trading symbols
            intervals: List of time intervals
            
        Returns:
            Trading logic results
        """
        self.logger.info("Starting trading logic...")
        self.system_status['trading_logic'] = 'running'
        
        symbols = symbols or self.config.get('symbols', ['BTCUSDT'])
        intervals = intervals or self.config.get('intervals', ['15m'])
        
        results = {
            'trade_decisions': {},
            'current_positions': {},
            'performance': {},
            'success': True,
            'errors': []
        }
        
        try:
            for symbol in symbols:
                for interval in intervals:
                    self.logger.info(f"Running trading logic for {symbol} {interval}")
                    
                    # Get fused signals (would come from signal generation)
                    # For now, create sample data
                    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
                    sample_data = pd.DataFrame({
                        'open_time': dates,
                        'close': 50000 + np.cumsum(np.random.randn(len(dates)) * 100),
                        's_fused': np.random.uniform(-1, 1, len(dates)),
                        's_fused_confidence': np.random.uniform(0.5, 1.0, len(dates)),
                        's_fused_strength': np.random.uniform(0.3, 1.0, len(dates))
                    })
                    
                    # Process trading signals
                    trade_decisions = self.trading_logic.process_trading_signals(sample_data)
                    results['trade_decisions'][f"{symbol}_{interval}"] = trade_decisions
                    
                    # Get current position
                    position_info = self.trading_logic.get_current_position_info()
                    results['current_positions'][f"{symbol}_{interval}"] = position_info
                    
                    # Get performance metrics
                    performance = self.trading_logic.get_performance_summary()
                    results['performance'][f"{symbol}_{interval}"] = performance
            
            self.system_status['trading_logic'] = 'completed'
            self.logger.info("Trading logic completed successfully")
            
        except Exception as e:
            self.logger.error(f"Trading logic failed: {e}")
            results['success'] = False
            results['errors'].append(str(e))
            self.system_status['trading_logic'] = 'failed'
        
        return results
    
    def run_complete_system(self, symbols: List[str] = None, intervals: List[str] = None) -> Dict[str, Any]:
        """
        Run the complete trading system.
        
        Args:
            symbols: List of trading symbols
            intervals: List of time intervals
            
        Returns:
            Complete system results
        """
        self.logger.info("Starting complete trading system...")
        start_time = datetime.now()
        
        symbols = symbols or self.config.get('symbols', ['BTCUSDT'])
        intervals = intervals or self.config.get('intervals', ['15m'])
        
        results = {
            'data_pipeline': {},
            'ml_pipeline': {},
            'signal_generation': {},
            'trading_logic': {},
            'system_status': self.system_status.copy(),
            'execution_time': 0,
            'success': True,
            'errors': []
        }
        
        try:
            # Run all pipelines
            results['data_pipeline'] = self.run_data_pipeline(symbols, intervals)
            results['ml_pipeline'] = self.run_ml_pipeline(symbols, intervals)
            results['signal_generation'] = self.run_signal_generation(symbols, intervals)
            results['trading_logic'] = self.run_trading_logic(symbols, intervals)
            
            # Update execution time
            execution_time = datetime.now() - start_time
            results['execution_time'] = str(execution_time)
            
            # Check overall success
            pipeline_results = [
                results['data_pipeline'].get('success', True),
                results['ml_pipeline'].get('success', True),
                results['signal_generation'].get('success', True),
                results['trading_logic'].get('success', True)
            ]
            
            results['success'] = all(pipeline_results)
            
            self.last_run_time = datetime.now()
            
            if results['success']:
                self.logger.info(f"Complete trading system executed successfully in {execution_time}")
            else:
                self.logger.warning("Trading system completed with some failures")
            
        except Exception as e:
            self.logger.error(f"Complete trading system failed: {e}")
            results['success'] = False
            results['errors'].append(str(e))
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'system_status': self.system_status.copy(),
            'last_run_time': self.last_run_time,
            'modules_initialized': True,
            'config_loaded': True
        }


def main():
    """Main entry point for the complete trading system."""
    parser = argparse.ArgumentParser(description='Complete Crypto Trading System')
    parser.add_argument('--mode', choices=['data', 'ml', 'signals', 'trading', 'complete'], 
                       default='complete', help='Run mode')
    parser.add_argument('--symbols', nargs='+', help='Trading symbols')
    parser.add_argument('--intervals', nargs='+', help='Time intervals')
    parser.add_argument('--config', default='config.json', help='Config file path')
    
    args = parser.parse_args()
    
    # Initialize system
    system = CompleteTradingSystem(args.config)
    
    # Run based on mode
    if args.mode == 'data':
        results = system.run_data_pipeline(args.symbols, args.intervals)
    elif args.mode == 'ml':
        results = system.run_ml_pipeline(args.symbols, args.intervals)
    elif args.mode == 'signals':
        results = system.run_signal_generation(args.symbols, args.intervals)
    elif args.mode == 'trading':
        results = system.run_trading_logic(args.symbols, args.intervals)
    else:  # complete
        results = system.run_complete_system(args.symbols, args.intervals)
    
    # Print results summary
    print(f"\n=== {args.mode.upper()} MODE RESULTS ===")
    print(f"Success: {results['success']}")
    if results.get('errors'):
        print(f"Errors: {results['errors']}")
    
    if args.mode == 'complete':
        print(f"Execution time: {results['execution_time']}")
        print(f"System status: {results['system_status']}")


if __name__ == "__main__":
    main()
