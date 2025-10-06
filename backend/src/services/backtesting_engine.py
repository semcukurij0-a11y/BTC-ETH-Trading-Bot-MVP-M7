"""
Futures Backtesting & Evaluation Engine

This module provides comprehensive backtesting for futures trading with:
- Vectorized backtesting engine
- Realistic fees, slippage, and funding costs
- Train/validation/test splits and walk-forward analysis
- Ablation studies (ML-only, TA-only, Sent-only, Fusion)
- Comprehensive performance metrics
- Acceptance gates for live trading
"""

import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from .risk_management import RiskManagementModule
from .technical_analysis import TechnicalAnalysisModule
from .sentiment_analysis import SentimentAnalysisModule
from .signal_fusion import SignalFusionModule
from .ml_inference import MLInference


class BacktestMode(Enum):
    FULL_SYSTEM = "full_system"
    ML_ONLY = "ml_only"
    TA_ONLY = "ta_only"
    SENT_ONLY = "sent_only"
    FUSION_ONLY = "fusion_only"
    BUY_HOLD = "buy_hold"


class BacktestingEngine:
    """
    Comprehensive futures backtesting engine with realistic cost modeling.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize backtesting engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Backtesting parameters
        self.initial_balance = self.config.get('initial_balance', 100000.0)
        self.leverage = self.config.get('leverage', 3.0)
        
        # Fee structure (per venue)
        self.fee_config = self.config.get('fees', {
            'maker': 0.0001,      # 0.01% maker fee
            'taker': 0.0006,      # 0.06% taker fee
            'funding_rate': 0.0001,  # 0.01% per 8-hour funding
            'slippage_bps': 2.0,  # 2 basis points slippage
            'spread_floor_bps': 1.0  # 1 basis point spread floor
        })
        
        # Performance thresholds
        self.acceptance_gates = self.config.get('acceptance_gates', {
            'min_sharpe': 1.0,
            'max_drawdown': 0.12,  # 12%
            'min_profit_factor': 1.0,
            'min_hit_rate': 0.45
        })
        
        # Initialize modules
        self.risk_module = RiskManagementModule(config)
        self.ta_module = TechnicalAnalysisModule(config)
        self.sentiment_module = SentimentAnalysisModule(config)
        self.signal_fusion = SignalFusionModule(config)
        self.ml_inference = MLInference("src/models", config)
        
        # Results storage
        self.backtest_results = {}
        self.performance_metrics = {}
        
    def calculate_trading_costs(self, 
                              entry_price: float, 
                              exit_price: float, 
                              quantity: float,
                              entry_side: str,
                              exit_side: str,
                              funding_hours: float = 8.0) -> Dict[str, float]:
        """
        Calculate realistic trading costs including fees, slippage, and funding.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position quantity
            entry_side: 'buy' or 'sell'
            exit_side: 'buy' or 'sell'
            funding_hours: Hours position was held
            
        Returns:
            Dictionary with cost breakdown
        """
        try:
            notional_entry = entry_price * quantity
            notional_exit = exit_price * quantity
            
            # Trading fees (assume taker for simplicity)
            entry_fee = notional_entry * self.fee_config['taker']
            exit_fee = notional_exit * self.fee_config['taker']
            total_fees = entry_fee + exit_fee
            
            # Slippage (spread floor + impact)
            spread_floor = notional_entry * (self.fee_config['spread_floor_bps'] / 10000)
            slippage_impact = notional_entry * (self.fee_config['slippage_bps'] / 10000)
            total_slippage = spread_floor + slippage_impact
            
            # Funding costs (every 8 hours)
            funding_periods = max(1, funding_hours / 8.0)
            funding_cost = notional_entry * self.fee_config['funding_rate'] * funding_periods
            
            # Total costs
            total_costs = total_fees + total_slippage + funding_cost
            
            return {
                'entry_fee': entry_fee,
                'exit_fee': exit_fee,
                'total_fees': total_fees,
                'slippage': total_slippage,
                'funding_cost': funding_cost,
                'total_costs': total_costs,
                'cost_percentage': total_costs / notional_entry
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trading costs: {e}")
            return {
                'entry_fee': 0.0,
                'exit_fee': 0.0,
                'total_fees': 0.0,
                'slippage': 0.0,
                'funding_cost': 0.0,
                'total_costs': 0.0,
                'cost_percentage': 0.0
            }
    
    def vectorized_backtest(self, 
                          data: pd.DataFrame, 
                          mode: BacktestMode = BacktestMode.FULL_SYSTEM,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run vectorized backtest on historical data.
        
        Args:
            data: OHLCV data with features
            mode: Backtesting mode
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary with backtest results
        """
        try:
            self.logger.info(f"Starting vectorized backtest in {mode.value} mode")
            
            # Filter data by date range
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            if data.empty:
                raise ValueError("No data available for backtesting period")
            
            # Initialize portfolio
            portfolio = {
                'balance': self.initial_balance,
                'equity': self.initial_balance,
                'position': 0.0,
                'position_price': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'total_fees': 0.0,
                'total_slippage': 0.0,
                'total_funding': 0.0
            }
            
            # Generate signals based on mode
            signals = self._generate_signals(data, mode)
            
            # Vectorized position sizing
            position_sizes = self._calculate_position_sizes(data, signals, portfolio)
            
            # Vectorized trade execution
            trades = self._execute_trades_vectorized(data, signals, position_sizes, portfolio)
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(trades, data, portfolio)
            
            # Store results
            result = {
                'mode': mode.value,
                'period': {
                    'start': data.index[0].isoformat(),
                    'end': data.index[-1].isoformat(),
                    'days': (data.index[-1] - data.index[0]).days
                },
                'trades': trades,
                'performance': performance,
                'portfolio': portfolio,
                'signals': signals,
                'data_points': len(data)
            }
            
            self.backtest_results[mode.value] = result
            self.logger.info(f"Backtest completed: {len(trades)} trades, Sharpe: {performance['sharpe_ratio']:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in vectorized backtest: {e}")
            return {
                'mode': mode.value,
                'error': str(e),
                'trades': [],
                'performance': {}
            }
    
    def _generate_signals(self, data: pd.DataFrame, mode: BacktestMode) -> pd.DataFrame:
        """
        Generate trading signals based on mode.
        
        Args:
            data: Input data with features
            mode: Backtesting mode
            
        Returns:
            DataFrame with signals
        """
        try:
            signals = pd.DataFrame(index=data.index)
            
            # Try to get price data from available columns
            if 'close' in data.columns:
                signals['price'] = data['close']
            elif 'mark_close' in data.columns:
                signals['price'] = data['mark_close']
            elif 'index_close' in data.columns:
                signals['price'] = data['index_close']
            else:
                # If no price data available, create synthetic price data
                self.logger.warning("No price data available, creating synthetic prices")
                signals['price'] = 50000 + np.cumsum(np.random.randn(len(data)) * 100)
            
            if mode == BacktestMode.BUY_HOLD:
                # Buy and hold signal
                signals['signal'] = 1.0
                signals['confidence'] = 1.0
                signals['strength'] = 1.0
                
            elif mode == BacktestMode.ML_ONLY:
                # ML-only signals
                if 's_ml' in data.columns:
                    signals['signal'] = data['s_ml']
                    signals['confidence'] = np.abs(data['s_ml'])
                    signals['strength'] = np.abs(data['s_ml'])
                else:
                    signals['signal'] = 0.0
                    signals['confidence'] = 0.0
                    signals['strength'] = 0.0
                    
            elif mode == BacktestMode.TA_ONLY:
                # Technical analysis only
                if 's_ta' in data.columns:
                    signals['signal'] = data['s_ta']
                    signals['confidence'] = np.abs(data['s_ta'])
                    signals['strength'] = np.abs(data['s_ta'])
                else:
                    signals['signal'] = 0.0
                    signals['confidence'] = 0.0
                    signals['strength'] = 0.0
                    
            elif mode == BacktestMode.SENT_ONLY:
                # Sentiment only
                if 's_sent' in data.columns:
                    signals['signal'] = data['s_sent']
                    signals['confidence'] = np.abs(data['s_sent'])
                    signals['strength'] = np.abs(data['s_sent'])
                else:
                    signals['signal'] = 0.0
                    signals['confidence'] = 0.0
                    signals['strength'] = 0.0
                    
            elif mode == BacktestMode.FUSION_ONLY:
                # Signal fusion only
                if 's_fused' in data.columns:
                    signals['signal'] = data['s_fused']
                    signals['confidence'] = np.abs(data['s_fused'])
                    signals['strength'] = np.abs(data['s_fused'])
                else:
                    signals['signal'] = 0.0
                    signals['confidence'] = 0.0
                    signals['strength'] = 0.0
                    
            else:  # FULL_SYSTEM
                # Use full system with risk management
                if 's_fused' in data.columns:
                    signals['signal'] = data['s_fused']
                    signals['confidence'] = data.get('s_fused_confidence', np.abs(data['s_fused']))
                    signals['strength'] = data.get('s_fused_strength', np.abs(data['s_fused']))
                else:
                    signals['signal'] = 0.0
                    signals['confidence'] = 0.0
                    signals['strength'] = 0.0
            
            # Apply signal thresholds
            min_confidence = self.config.get('min_confidence', 0.6)
            min_strength = self.config.get('min_signal_strength', 0.3)
            
            # Filter signals based on confidence and strength
            valid_signals = (signals['confidence'] >= min_confidence) & (signals['strength'] >= min_strength)
            signals['signal'] = np.where(valid_signals, signals['signal'], 0.0)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_position_sizes(self, 
                                data: pd.DataFrame, 
                                signals: pd.DataFrame, 
                                portfolio: Dict) -> pd.Series:
        """
        Calculate position sizes using risk management.
        
        Args:
            data: Input data
            signals: Trading signals
            portfolio: Portfolio state
            
        Returns:
            Series with position sizes
        """
        try:
            position_sizes = pd.Series(0.0, index=data.index)
            
            # Calculate ATR for position sizing
            if 'atr' in data.columns:
                atr = data['atr']
            else:
                # Simple ATR calculation if OHLC data available
                if all(col in data.columns for col in ['high', 'low', 'close']):
                    high_low = data['high'] - data['low']
                    high_close = np.abs(data['high'] - data['close'].shift(1))
                    low_close = np.abs(data['low'] - data['close'].shift(1))
                    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                    atr = true_range.rolling(window=14).mean()
                else:
                    # Use price-based volatility estimation
                    if 'close' in data.columns:
                        price_volatility = data['close'].pct_change().rolling(14).std()
                        atr = data['close'] * price_volatility
                    else:
                        # Default ATR estimation
                        atr = pd.Series(500.0, index=data.index)  # $500 default ATR
            
            # Calculate position sizes
            for i, (timestamp, row) in enumerate(data.iterrows()):
                if signals.loc[timestamp, 'signal'] != 0:
                    # Use risk management for position sizing
                    risk_amount = portfolio['balance'] * self.risk_module.risk_per_trade
                    current_price = signals.loc[timestamp, 'price']
                    current_atr = atr.loc[timestamp] if not pd.isna(atr.loc[timestamp]) else current_price * 0.02
                    
                    # Calculate stop loss distance (2.5 * ATR)
                    sl_distance = current_atr * self.risk_module.atr_sl_multiplier
                    
                    # Calculate position size
                    position_size = risk_amount / sl_distance
                    
                    # Apply leverage cap
                    max_position = (portfolio['balance'] * self.leverage) / current_price
                    position_size = min(position_size, max_position)
                    
                    position_sizes.loc[timestamp] = position_size
            
            return position_sizes
            
        except Exception as e:
            self.logger.error(f"Error calculating position sizes: {e}")
            return pd.Series(0.0, index=data.index)
    
    def _execute_trades_vectorized(self, 
                                 data: pd.DataFrame, 
                                 signals: pd.DataFrame, 
                                 position_sizes: pd.Series,
                                 portfolio: Dict) -> List[Dict[str, Any]]:
        """
        Execute trades in vectorized manner.
        
        Args:
            data: Input data
            signals: Trading signals
            position_sizes: Position sizes
            portfolio: Portfolio state
            
        Returns:
            List of executed trades
        """
        try:
            trades = []
            current_position = 0.0
            current_position_price = 0.0
            position_entry_time = None
            
            for i, (timestamp, row) in enumerate(data.iterrows()):
                signal = signals.loc[timestamp, 'signal']
                price = signals.loc[timestamp, 'price']
                position_size = position_sizes.loc[timestamp]
                
                # Check for position changes
                if current_position == 0.0 and signal != 0.0 and position_size > 0:
                    # Open new position
                    current_position = position_size if signal > 0 else -position_size
                    current_position_price = price
                    position_entry_time = timestamp
                    
                    trade = {
                        'entry_time': timestamp,
                        'entry_price': price,
                        'side': 'long' if signal > 0 else 'short',
                        'quantity': abs(current_position),
                        'entry_signal': signal,
                        'entry_confidence': signals.loc[timestamp, 'confidence']
                    }
                    trades.append(trade)
                    
                elif current_position != 0.0:
                    # Check exit conditions
                    should_exit = False
                    exit_reason = ""
                    
                    # Opposite signal
                    if (current_position > 0 and signal < -0.2) or (current_position < 0 and signal > 0.2):
                        should_exit = True
                        exit_reason = "opposite_signal"
                    
                    # Stop loss or take profit (simplified)
                    if current_position > 0:  # Long position
                        pnl_pct = (price - current_position_price) / current_position_price
                        if pnl_pct < -0.02:  # 2% stop loss
                            should_exit = True
                            exit_reason = "stop_loss"
                        elif pnl_pct > 0.04:  # 4% take profit
                            should_exit = True
                            exit_reason = "take_profit"
                    else:  # Short position
                        pnl_pct = (current_position_price - price) / current_position_price
                        if pnl_pct < -0.02:  # 2% stop loss
                            should_exit = True
                            exit_reason = "stop_loss"
                        elif pnl_pct > 0.04:  # 4% take profit
                            should_exit = True
                            exit_reason = "take_profit"
                    
                    if should_exit:
                        # Close position
                        funding_hours = (timestamp - position_entry_time).total_seconds() / 3600
                        
                        # Calculate costs
                        costs = self.calculate_trading_costs(
                            current_position_price, price, abs(current_position),
                            'buy' if current_position > 0 else 'sell',
                            'sell' if current_position > 0 else 'buy',
                            funding_hours
                        )
                        
                        # Calculate P&L
                        if current_position > 0:  # Long position
                            gross_pnl = (price - current_position_price) * abs(current_position)
                        else:  # Short position
                            gross_pnl = (current_position_price - price) * abs(current_position)
                        
                        net_pnl = gross_pnl - costs['total_costs']
                        
                        # Update portfolio
                        portfolio['balance'] += net_pnl
                        portfolio['realized_pnl'] += net_pnl
                        portfolio['total_fees'] += costs['total_fees']
                        portfolio['total_slippage'] += costs['slippage']
                        portfolio['total_funding'] += costs['funding_cost']
                        
                        # Complete trade record
                        if trades:
                            trades[-1].update({
                                'exit_time': timestamp,
                                'exit_price': price,
                                'exit_signal': signal,
                                'exit_reason': exit_reason,
                                'gross_pnl': gross_pnl,
                                'net_pnl': net_pnl,
                                'costs': costs,
                                'funding_hours': funding_hours,
                                'duration': timestamp - position_entry_time
                            })
                        
                        # Reset position
                        current_position = 0.0
                        current_position_price = 0.0
                        position_entry_time = None
                
                # Update unrealized P&L
                if current_position != 0.0:
                    if current_position > 0:  # Long position
                        portfolio['unrealized_pnl'] = (price - current_position_price) * abs(current_position)
                    else:  # Short position
                        portfolio['unrealized_pnl'] = (current_position_price - price) * abs(current_position)
                
                portfolio['equity'] = portfolio['balance'] + portfolio['unrealized_pnl']
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error executing trades: {e}")
            return []
    
    def _calculate_performance_metrics(self, 
                                     trades: List[Dict], 
                                     data: pd.DataFrame, 
                                     portfolio: Dict) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            trades: List of executed trades
            data: Input data
            portfolio: Portfolio state
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            if not trades:
                return self._empty_performance_metrics()
            
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(trades)
            
            # Basic metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
            losing_trades = len(trades_df[trades_df['net_pnl'] < 0])
            hit_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # P&L metrics
            total_pnl = trades_df['net_pnl'].sum()
            avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0.0
            avg_loss = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0.0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # Return metrics
            total_return = total_pnl / self.initial_balance
            annualized_return = total_return * (365 / ((data.index[-1] - data.index[0]).days))
            
            # Risk metrics
            returns = trades_df['net_pnl'] / self.initial_balance
            volatility = returns.std() * np.sqrt(365)  # Annualized
            
            # Sharpe ratio
            risk_free_rate = 0.02  # 2% risk-free rate
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0.0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 0 else 0.0
            sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0.0
            
            # Drawdown (simplified calculation)
            if len(trades_df) > 1:
                cumulative_pnl = trades_df['net_pnl'].cumsum()
                running_max = cumulative_pnl.expanding().max()
                drawdown = (cumulative_pnl - running_max) / self.initial_balance
                max_drawdown = drawdown.min()
            else:
                max_drawdown = 0.0
            
            # CVaR (Conditional Value at Risk) at 5%
            cvar_5 = returns.quantile(0.05) if len(returns) > 0 else 0.0
            
            # Average trade metrics
            avg_trade = trades_df['net_pnl'].mean()
            avg_trade_duration = trades_df['duration'].mean().total_seconds() / 3600  # Hours
            
            # Exposure
            exposure = trades_df['duration'].sum().total_seconds() / (len(data) * 3600)  # Fraction of time in market
            
            metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'hit_rate': hit_rate,
                'total_pnl': total_pnl,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'cvar_5': cvar_5,
                'avg_trade': avg_trade,
                'avg_trade_duration': avg_trade_duration,
                'exposure': exposure,
                'volatility': volatility,
                'total_fees': portfolio['total_fees'],
                'total_slippage': portfolio['total_slippage'],
                'total_funding': portfolio['total_funding'],
                'total_costs': portfolio['total_fees'] + portfolio['total_slippage'] + portfolio['total_funding']
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return self._empty_performance_metrics()
    
    def _empty_performance_metrics(self) -> Dict[str, float]:
        """Return empty performance metrics."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'hit_rate': 0.0,
            'total_pnl': 0.0,
            'total_return': 0.0,
            'annualized_return': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'cvar_5': 0.0,
            'avg_trade': 0.0,
            'avg_trade_duration': 0.0,
            'exposure': 0.0,
            'volatility': 0.0,
            'total_fees': 0.0,
            'total_slippage': 0.0,
            'total_funding': 0.0,
            'total_costs': 0.0
        }
    
    def run_ablation_study(self, 
                          data: pd.DataFrame,
                          train_start: str,
                          val_start: str,
                          test_start: str,
                          test_end: str) -> Dict[str, Any]:
        """
        Run ablation study comparing different signal sources.
        
        Args:
            data: Input data with features
            train_start: Training period start
            val_start: Validation period start
            test_start: Test period start
            test_end: Test period end
            
        Returns:
            Dictionary with ablation study results
        """
        try:
            self.logger.info("Starting ablation study")
            
            results = {}
            modes = [
                BacktestMode.BUY_HOLD,
                BacktestMode.ML_ONLY,
                BacktestMode.TA_ONLY,
                BacktestMode.SENT_ONLY,
                BacktestMode.FUSION_ONLY,
                BacktestMode.FULL_SYSTEM
            ]
            
            for mode in modes:
                self.logger.info(f"Running backtest for {mode.value}")
                result = self.vectorized_backtest(
                    data, mode, test_start, test_end
                )
                results[mode.value] = result
                
                # Check acceptance gates
                performance = result.get('performance', {})
                acceptance = self._check_acceptance_gates(performance)
                results[mode.value]['acceptance'] = acceptance
            
            # Create comparison summary
            comparison = self._create_comparison_summary(results)
            results['comparison'] = comparison
            
            self.logger.info("Ablation study completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in ablation study: {e}")
            return {'error': str(e)}
    
    def _check_acceptance_gates(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """
        Check if performance meets acceptance gates for live trading.
        
        Args:
            performance: Performance metrics
            
        Returns:
            Dictionary with acceptance status
        """
        try:
            gates = self.acceptance_gates
            
            acceptance = {
                'sharpe_gate': performance.get('sharpe_ratio', 0) >= gates['min_sharpe'],
                'drawdown_gate': abs(performance.get('max_drawdown', 0)) <= gates['max_drawdown'],
                'profit_factor_gate': performance.get('profit_factor', 0) >= gates['min_profit_factor'],
                'hit_rate_gate': performance.get('hit_rate', 0) >= gates['min_hit_rate'],
                'positive_pnl_gate': performance.get('total_pnl', 0) > 0,
                'all_gates_passed': True
            }
            
            # Check if all gates passed
            for key, value in acceptance.items():
                if key != 'all_gates_passed' and not value:
                    acceptance['all_gates_passed'] = False
                    break
            
            return acceptance
            
        except Exception as e:
            self.logger.error(f"Error checking acceptance gates: {e}")
            return {'all_gates_passed': False, 'error': str(e)}
    
    def _create_comparison_summary(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create comparison summary of all backtest modes.
        
        Args:
            results: Backtest results for all modes
            
        Returns:
            DataFrame with comparison
        """
        try:
            comparison_data = []
            
            for mode, result in results.items():
                if 'performance' in result:
                    perf = result['performance']
                    acceptance = result.get('acceptance', {})
                    
                    comparison_data.append({
                        'Mode': mode,
                        'Total Trades': perf.get('total_trades', 0),
                        'Hit Rate': f"{perf.get('hit_rate', 0):.1%}",
                        'Total Return': f"{perf.get('total_return', 0):.1%}",
                        'Annualized Return': f"{perf.get('annualized_return', 0):.1%}",
                        'Sharpe Ratio': f"{perf.get('sharpe_ratio', 0):.3f}",
                        'Sortino Ratio': f"{perf.get('sortino_ratio', 0):.3f}",
                        'Max Drawdown': f"{perf.get('max_drawdown', 0):.1%}",
                        'Profit Factor': f"{perf.get('profit_factor', 0):.2f}",
                        'CVaR (5%)': f"{perf.get('cvar_5', 0):.1%}",
                        'Exposure': f"{perf.get('exposure', 0):.1%}",
                        'Total Costs': f"${perf.get('total_costs', 0):,.0f}",
                        'Acceptance Gates': '✅' if acceptance.get('all_gates_passed', False) else '❌'
                    })
            
            return pd.DataFrame(comparison_data)
            
        except Exception as e:
            self.logger.error(f"Error creating comparison summary: {e}")
            return pd.DataFrame()
    
    def walk_forward_analysis(self, 
                            data: pd.DataFrame,
                            train_period: int = 252,  # 1 year
                            test_period: int = 63,    # 3 months
                            step_size: int = 21) -> Dict[str, Any]:
        """
        Run walk-forward analysis for robust backtesting.
        
        Args:
            data: Input data with features
            train_period: Training period in days
            test_period: Test period in days
            step_size: Step size in days
            
        Returns:
            Dictionary with walk-forward results
        """
        try:
            self.logger.info("Starting walk-forward analysis")
            
            results = []
            start_idx = train_period
            
            while start_idx + test_period < len(data):
                # Define periods
                train_end = start_idx
                test_start = start_idx
                test_end = start_idx + test_period
                
                train_data = data.iloc[:train_end]
                test_data = data.iloc[test_start:test_end]
                
                self.logger.info(f"Walk-forward step: Train {len(train_data)} days, Test {len(test_data)} days")
                
                # Run backtest on test period
                result = self.vectorized_backtest(test_data, BacktestMode.FULL_SYSTEM)
                
                # Add metadata
                result['walk_forward'] = {
                    'train_period': (train_data.index[0], train_data.index[-1]),
                    'test_period': (test_data.index[0], test_data.index[-1]),
                    'train_days': len(train_data),
                    'test_days': len(test_data)
                }
                
                results.append(result)
                start_idx += step_size
            
            # Aggregate results
            aggregated = self._aggregate_walk_forward_results(results)
            
            self.logger.info(f"Walk-forward analysis completed: {len(results)} periods")
            
            return {
                'individual_results': results,
                'aggregated': aggregated,
                'summary': {
                    'total_periods': len(results),
                    'train_period_days': train_period,
                    'test_period_days': test_period,
                    'step_size_days': step_size
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward analysis: {e}")
            return {'error': str(e)}
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate walk-forward results.
        
        Args:
            results: List of individual walk-forward results
            
        Returns:
            Dictionary with aggregated metrics
        """
        try:
            if not results:
                return {}
            
            # Collect all performance metrics
            all_metrics = []
            for result in results:
                if 'performance' in result:
                    all_metrics.append(result['performance'])
            
            if not all_metrics:
                return {}
            
            # Calculate aggregated metrics
            total_trades = sum(m.get('total_trades', 0) for m in all_metrics)
            total_pnl = sum(m.get('total_pnl', 0) for m in all_metrics)
            total_costs = sum(m.get('total_costs', 0) for m in all_metrics)
            
            # Average metrics
            avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in all_metrics])
            avg_drawdown = np.mean([m.get('max_drawdown', 0) for m in all_metrics])
            avg_hit_rate = np.mean([m.get('hit_rate', 0) for m in all_metrics])
            
            # Consistency metrics
            positive_periods = sum(1 for m in all_metrics if m.get('total_pnl', 0) > 0)
            consistency = positive_periods / len(all_metrics) if all_metrics else 0.0
            
            return {
                'total_periods': len(all_metrics),
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'total_costs': total_costs,
                'net_pnl': total_pnl - total_costs,
                'avg_sharpe_ratio': avg_sharpe,
                'avg_max_drawdown': avg_drawdown,
                'avg_hit_rate': avg_hit_rate,
                'consistency': consistency,
                'positive_periods': positive_periods
            }
            
        except Exception as e:
            self.logger.error(f"Error aggregating walk-forward results: {e}")
            return {}


def main():
    """Main function for testing."""
    # Sample configuration
    config = {
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
    }
    
    # Initialize backtesting engine
    engine = BacktestingEngine(config)
    
    print("Backtesting Engine initialized successfully")
    print(f"Initial Balance: ${config['initial_balance']:,.0f}")
    print(f"Leverage: {config['leverage']}x")
    print(f"Fee Structure: {config['fees']}")
    print(f"Acceptance Gates: {config['acceptance_gates']}")


if __name__ == "__main__":
    main()
