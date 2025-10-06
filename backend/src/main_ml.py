"""
Main ML Training and Inference Script for Crypto Trading Bot

This script provides a command-line interface for ML operations:
- Training models
- Running inference
- Managing scheduled retraining
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append('src')

from services.ml_trainer import MLTrainer
from services.ml_inference import MLInference
from services.ml_scheduler import MLScheduler


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/ml_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def train_model(args):
    """Train a single model."""
    config = load_config(args.config)
    ml_config = config.get('ml', {})
    
    # Override config with command line arguments
    if args.symbol:
        ml_config['symbols'] = [args.symbol]
    if args.interval:
        ml_config['intervals'] = [args.interval]
    if args.days_back:
        ml_config['days_back'] = args.days_back
    if args.model_type:
        ml_config['model_type'] = args.model_type
    
    trainer = MLTrainer(config=ml_config)
    
    try:
        result = trainer.train_model(
            symbol=args.symbol or 'BTCUSDT',
            interval=args.interval or '15m',
            days_back=args.days_back or 30,
            target_hours=args.target_hours or 1,
            threshold=args.threshold or 0.001
        )
        
        print("Training completed successfully!")
        print(f"Test AUC: {result['metrics']['test_auc']:.4f}")
        print(f"Test Loss: {result['metrics']['test_loss']:.4f}")
        print(f"Features: {result['metrics']['n_features']}")
        
        if args.show_importance:
            print("\nTop 10 Feature Importance:")
            print(result['feature_importance'].head(10).to_string(index=False))
        
    except Exception as e:
        print(f"Training failed: {e}")
        return 1
    
    return 0


def run_inference(args):
    """Run inference on latest data."""
    config = load_config(args.config)
    ml_config = config.get('ml', {})
    
    inference = MLInference(config=ml_config)
    
    try:
        prediction = inference.get_latest_prediction(
            symbol=args.symbol or 'BTCUSDT',
            interval=args.interval or '15m'
        )
        
        if 'error' in prediction:
            print(f"Inference failed: {prediction['error']}")
            return 1
        
        print("Inference completed successfully!")
        print(f"Symbol: {prediction['symbol']}")
        print(f"Interval: {prediction['interval']}")
        print(f"s_ml: {prediction['s_ml']:.4f}")
        print(f"Signal: {prediction['signal']}")
        print(f"Confidence: {prediction['confidence']:.4f}")
        print(f"Prediction Strength: {prediction['prediction_strength']:.4f}")
        
    except Exception as e:
        print(f"Inference failed: {e}")
        return 1
    
    return 0


def start_scheduler(args):
    """Start the ML scheduler."""
    config = load_config(args.config)
    ml_config = config.get('ml', {})
    
    scheduler = MLScheduler(config=ml_config)
    
    try:
        scheduler.start_scheduler()
        print("ML Scheduler started successfully!")
        print(f"Next retrain: {scheduler.get_next_retrain_time()}")
        
        # Keep running
        import time
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nStopping scheduler...")
            scheduler.stop_scheduler()
            print("Scheduler stopped.")
        
    except Exception as e:
        print(f"Scheduler failed: {e}")
        return 1
    
    return 0


def show_status(args):
    """Show ML system status."""
    config = load_config(args.config)
    ml_config = config.get('ml', {})
    
    scheduler = MLScheduler(config=ml_config)
    status = scheduler.get_training_status()
    
    print("ML System Status:")
    print(f"Scheduler Enabled: {status['scheduler_enabled']}")
    print(f"Retrain Schedule: {status['retrain_schedule']}")
    print(f"Next Retrain: {scheduler.get_next_retrain_time()}")
    print("\nModel Status:")
    
    for model_key, model_status in status['models'].items():
        print(f"  {model_key}:")
        print(f"    Status: {model_status['status']}")
        print(f"    Last Training: {model_status['last_training']}")
        print(f"    Needs Retrain: {model_status['needs_retrain']}")
        if model_status['error']:
            print(f"    Error: {model_status['error']}")
    
    return 0


def force_retrain(args):
    """Force retrain all models."""
    config = load_config(args.config)
    ml_config = config.get('ml', {})
    
    scheduler = MLScheduler(config=ml_config)
    
    try:
        print("Starting force retrain...")
        result = scheduler.force_retrain_all()
        
        if 'error' in result:
            print(f"Force retrain failed: {result['error']}")
            return 1
        
        print("Force retrain completed!")
        print(f"Successful: {result['successful']}")
        print(f"Failed: {result['failed']}")
        print(f"Duration: {result['duration']}")
        
    except Exception as e:
        print(f"Force retrain failed: {e}")
        return 1
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Crypto Trading Bot ML System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--symbol', help='Trading symbol (default: BTCUSDT)')
    train_parser.add_argument('--interval', help='Timeframe (default: 15m)')
    train_parser.add_argument('--days-back', type=int, help='Days of training data (default: 30)')
    train_parser.add_argument('--target-hours', type=int, help='Target prediction hours (default: 1)')
    train_parser.add_argument('--threshold', type=float, help='Return threshold (default: 0.001)')
    train_parser.add_argument('--model-type', choices=['lightgbm', 'xgboost'], help='Model type')
    train_parser.add_argument('--show-importance', action='store_true', help='Show feature importance')
    train_parser.add_argument('--config', default='config.json', help='Config file path')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--symbol', help='Trading symbol (default: BTCUSDT)')
    inference_parser.add_argument('--interval', help='Timeframe (default: 15m)')
    inference_parser.add_argument('--config', default='config.json', help='Config file path')
    
    # Scheduler command
    scheduler_parser = subparsers.add_parser('scheduler', help='Start ML scheduler')
    scheduler_parser.add_argument('--config', default='config.json', help='Config file path')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show ML system status')
    status_parser.add_argument('--config', default='config.json', help='Config file path')
    
    # Force retrain command
    retrain_parser = subparsers.add_parser('retrain', help='Force retrain all models')
    retrain_parser.add_argument('--config', default='config.json', help='Config file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    config = load_config(args.config)
    setup_logging(config.get('log_level', 'INFO'))
    
    # Execute command
    if args.command == 'train':
        return train_model(args)
    elif args.command == 'inference':
        return run_inference(args)
    elif args.command == 'scheduler':
        return start_scheduler(args)
    elif args.command == 'status':
        return show_status(args)
    elif args.command == 'retrain':
        return force_retrain(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
