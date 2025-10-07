#!/usr/bin/env python3
"""
Leveraged Backtest - 3x Leverage Implementation with Success Criteria

Advanced Requirements with Leverage:
- 3x leverage for enhanced returns
- Entry refinement: MFE ≥ 30 realistic 
- Tight stops: MAE ≤ 10 with leverage
- Exit strategy: ATR-based stops with 1.5:1 risk/reward ratio (tighter stops for higher win rate)
- Success criteria: Sharpe ≥ 1.2, Avg gain ≥ 1-2%, Max DD ≤ 1-1.5%

Author: Quant Developer
Date: 2024
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
from scipy import stats
from scipy.stats import pearsonr
import itertools
warnings.filterwarnings('ignore')

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period).mean()
    return atr


def calculate_kelly_fraction(returns: List[float], min_trades: int = 10) -> float:
    """Calculate Kelly fraction based on historical returns."""
    if len(returns) < min_trades:
        return 0.25  # Default conservative fraction
    
    returns_array = np.array(returns)
    
    # Calculate win rate and average win/loss
    wins = returns_array[returns_array > 0]
    losses = returns_array[returns_array < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return 0.25  # Default if no wins or losses
    
    win_rate = len(wins) / len(returns)
    avg_win = np.mean(wins)
    avg_loss = abs(np.mean(losses))
    
    # Kelly formula: f = (bp - q) / b
    # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
    if avg_loss == 0:
        return 0.25  # Avoid division by zero
    
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - win_rate
    
    kelly_fraction = (b * p - q) / b
    
    # Constrain Kelly fraction to reasonable bounds
    kelly_fraction = max(0.05, min(0.5, kelly_fraction))  # Between 5% and 50%
    
    return kelly_fraction

class LeveragedBacktestEngine:
    """Leveraged backtesting engine with 5x leverage and conservative risk management."""
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 leverage: float = 5.0,  # 5x leverage
                 commission: float = 0.0001,
                 slippage: float = 0.00005,
                 max_position_size: float = 0.20,  # 20.0% of capital (5x = 100% exposure)
                 base_position_size: float = 0.10,  # 10.0% base (5x = 50% exposure)
                 stop_loss_pct: float = 0.001,  # 0.1% stop loss (tight for leverage)
                 profit_target_pct: float = 0.008,  # 0.8% profit target
                 trailing_stop_pct: float = 0.15,  # 15% trailing stop
                 use_atr_stops: bool = True,  # Use ATR-based stops
                 atr_multiplier: float = 0.6,  # 0.6× ATR multiplier (very tight stops for lower volatility)
                 atr_profit_multiplier: float = 3.0,  # Base ATR profit multiplier (dynamic 2.5-3.5)
                 atr_period: int = 14,  # ATR period
                 use_trend_following: bool = True,  # Use trend-following exit signals instead of fixed TP
                 use_fixed_take_profit: bool = False,  # Use fixed take-profit (disabled for trend-following)
                 use_fixed_stop_loss: bool = True,  # Keep stop-loss for risk management
                 use_kelly_sizing: bool = True,  # Use Kelly fraction for position sizing
                 kelly_lookback: int = 50,  # Number of trades to look back for Kelly calculation
                 kelly_fraction_multiplier: float = 1.0,  # Kelly fraction scaling multiplier
                 use_volatility_filter: bool = False,  # Use volatility filter to avoid extreme noise (disabled for more trades)
                 volatility_threshold: float = 0.03,  # 3% volatility threshold (balanced for trade execution)
                 min_liquidity_hours: List[int] = None,  # Hours to avoid trading (low liquidity)
                 max_drawdown_limit: float = 0.015,  # 1.5% maximum drawdown limit (strict risk control)
                 daily_stop_loss: float = 0.01,  # 1% daily stop-loss limit
                 use_volatility_regime_filter: bool = True,  # Use volatility regime filter to avoid noisy conditions
                 max_atr_threshold: float = 0.67,  # Skip trades if ATR > 67.0% (allows more crypto volatility)
                 min_signal_strength: float = 0.4,  # Very relaxed threshold for maximum opportunities (40%)
                 min_confidence: float = 0.3,      # Very relaxed threshold for maximum opportunities (30%)
                 min_target_return: float = 0.002,  # 0.2% minimum expected return
                 max_portfolio_risk: float = 0.02,  # 2% max portfolio risk
                 margin_call_threshold: float = 0.8):  # 80% margin call threshold
        
        # Core parameters
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.base_position_size = base_position_size
        self.stop_loss_pct = stop_loss_pct
        self.profit_target_pct = profit_target_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.use_atr_stops = use_atr_stops
        self.atr_multiplier = atr_multiplier
        self.atr_profit_multiplier = atr_profit_multiplier
        self.atr_period = atr_period
        self.use_trend_following = use_trend_following
        self.use_fixed_take_profit = use_fixed_take_profit
        self.use_fixed_stop_loss = use_fixed_stop_loss
        self.use_kelly_sizing = use_kelly_sizing
        self.kelly_lookback = kelly_lookback
        self.kelly_fraction_multiplier = kelly_fraction_multiplier
        self.use_volatility_filter = use_volatility_filter
        self.volatility_threshold = volatility_threshold
        self.min_liquidity_hours = min_liquidity_hours or [22, 23, 0, 1, 2, 3, 4, 5]  # Default: avoid late night/early morning
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_stop_loss = daily_stop_loss
        self.use_volatility_regime_filter = use_volatility_regime_filter
        self.max_atr_threshold = max_atr_threshold
        self.min_signal_strength = min_signal_strength
        self.min_confidence = min_confidence
        self.min_target_return = min_target_return
        self.max_portfolio_risk = max_portfolio_risk
        self.margin_call_threshold = margin_call_threshold
        
        # Trading state
        self.capital = initial_capital
        self.available_margin = initial_capital
        self.used_margin = 0.0
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.margin_calls = 0
        
        # Drawdown tracking
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        self.trading_halted = False
        
        # Daily tracking
        self.daily_start_capital = initial_capital
        self.current_date = None
        self.daily_loss = 0.0
        
    def reset(self):
        """Reset all state."""
        self.capital = self.initial_capital
        self.available_margin = self.initial_capital
        self.used_margin = 0.0
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.margin_calls = 0
        
        # Reset drawdown tracking
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        self.trading_halted = False
    
    def get_timestamp_column(self, df: pd.DataFrame) -> str:
        """Get timestamp column."""
        if 'start_time' in df.columns:
            return 'start_time'
        elif 'timestamps' in df.columns:
            return 'timestamps'
        else:
            return 'timestamp'
    
    def calculate_margin_requirements(self, position_value: float) -> float:
        """Calculate margin requirement for leveraged position."""
        return position_value / self.leverage
    
    def check_margin_call(self) -> bool:
        """Check if margin call is triggered."""
        margin_ratio = self.used_margin / self.capital if self.capital > 0 else 0
        return margin_ratio >= self.margin_call_threshold
    
    def calculate_dynamic_profit_multiplier(self, df: pd.DataFrame, current_atr: float) -> float:
        """Calculate dynamic profit multiplier based on market conditions."""
        base_multiplier = self.atr_profit_multiplier  # 4.0
        
        # Get recent volatility and trend information
        if len(df) < 20:
            return base_multiplier
        
        recent_vol = df['close'].iloc[-20:].pct_change().std()
        recent_trend = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
        
        # Adjust multiplier based on volatility
        if recent_vol < 0.01:  # Low volatility
            volatility_adjustment = 0.4  # Increase profit target even more aggressively
        elif recent_vol > 0.03:  # High volatility
            volatility_adjustment = -0.3  # Decrease profit target less aggressively
        else:
            volatility_adjustment = 0.0
        
        # Adjust multiplier based on trend strength
        if abs(recent_trend) > 0.05:  # Strong trend
            trend_adjustment = 0.5  # Increase profit target even more aggressively
        elif abs(recent_trend) < 0.01:  # Weak trend
            trend_adjustment = -0.2  # Decrease profit target less aggressively
        else:
            trend_adjustment = 0.0
        
        # Calculate dynamic multiplier
        dynamic_multiplier = base_multiplier + volatility_adjustment + trend_adjustment
        
        # Constrain to [2.5, 3.5] range for more consistent profit targets
        return max(2.5, min(3.5, dynamic_multiplier))
    
    def calculate_simple_position_size(self, signal_strength: float, confidence: float) -> float:
        """
        Calculate position size without requiring trade history (for CSV processing).
        Uses the same logic as the original position sizing but without Kelly fraction.
        """
        # Use base position sizing logic
        base_fraction = self.base_position_size
        quality_multiplier = (abs(signal_strength) * confidence)
        
        if quality_multiplier > 0.8:
            return self.max_position_size
        elif quality_multiplier > 0.6:
            return base_fraction * 1.5
        elif quality_multiplier > 0.4:
            return base_fraction * 1.2
        else:
            return base_fraction * 0.8
    
    def calculate_atr_stop_levels(self, df: pd.DataFrame, entry_price: float, is_long: bool) -> Dict[str, float]:
        """Calculate ATR-based stop levels."""
        if not self.use_atr_stops:
            # Use fixed percentage stops
            if is_long:
                stop_loss = entry_price * (1 - self.stop_loss_pct)
                profit_target = entry_price * (1 + self.profit_target_pct)
            else:
                stop_loss = entry_price * (1 + self.stop_loss_pct)
                profit_target = entry_price * (1 - self.profit_target_pct)
            
            return {
                'stop_loss': stop_loss,
                'profit_target': profit_target,
                'trailing_stop': self.trailing_stop_pct
            }
        
        # Calculate ATR
        atr = calculate_atr(df['high'], df['low'], df['close'], self.atr_period)
        current_atr = atr.iloc[-1] if not atr.empty else entry_price * 0.01  # Fallback to 1%
        
        # Calculate ATR-based levels
        atr_stop_distance = current_atr * self.atr_multiplier
        
        # Use dynamic profit multiplier
        dynamic_profit_multiplier = self.calculate_dynamic_profit_multiplier(df, current_atr)
        
        if is_long:
            stop_loss = entry_price - atr_stop_distance
            profit_target = entry_price + (atr_stop_distance * dynamic_profit_multiplier)
        else:
            stop_loss = entry_price + atr_stop_distance
            profit_target = entry_price - (atr_stop_distance * dynamic_profit_multiplier)
        
        # Convert ATR distance to percentage for trailing stop
        atr_trailing_pct = atr_stop_distance / entry_price
        
        return {
            'stop_loss': stop_loss,
            'profit_target': profit_target,
            'trailing_stop': atr_trailing_pct,
            'atr_value': current_atr,
            'dynamic_profit_multiplier': dynamic_profit_multiplier
        }
    
    def calculate_kelly_position_size(self, signal_strength: float, confidence: float) -> float:
        """Calculate position size using Kelly fraction based on past performance."""
        if not self.use_kelly_sizing:
            # Use original position sizing logic
            base_fraction = self.base_position_size
            quality_multiplier = (abs(signal_strength) * confidence)
            if quality_multiplier > 0.8:
                return self.max_position_size
            elif quality_multiplier > 0.6:
                return base_fraction * 1.5
            elif quality_multiplier > 0.4:
                return base_fraction * 1.2
            else:
                return base_fraction * 0.8
        
        # Get recent trade returns for Kelly calculation
        recent_returns = []
        if len(self.trades) > 0:
            # Get returns from recent trades (up to kelly_lookback)
            recent_trades = self.trades[-self.kelly_lookback:]
            for trade in recent_trades:
                if 'return_pct' in trade:
                    recent_returns.append(trade['return_pct'])
        
        # Calculate Kelly fraction
        kelly_fraction = calculate_kelly_fraction(recent_returns)
        
        # FIXED: Use Kelly fraction to scale between base and max position sizes
        # Kelly fraction of 0.25 (default) = base position
        # Kelly fraction of 0.5 (max) = max position
        # This ensures we get the intended 5.0-10.0% positions (50-100% exposure)
        
        # Map Kelly fraction (0.05-0.5) to position size range (base to max)
        kelly_range = 0.5 - 0.05  # 0.45 range
        position_range = self.max_position_size - self.base_position_size  # 0.05 range (10.0% - 5.0%)
        
        # Scale Kelly fraction to position size
        kelly_scaled = (kelly_fraction - 0.05) / kelly_range  # 0 to 1
        kelly_scaled = max(0.0, min(1.0, kelly_scaled))  # Clamp to 0-1
        
        # Calculate position size: base + (scaled_kelly * range)
        kelly_adjusted_base = self.base_position_size + (kelly_scaled * position_range)
        
        # Apply signal quality multiplier
        quality_multiplier = (abs(signal_strength) * confidence)
        if quality_multiplier > 0.8:
            position_fraction = min(self.max_position_size, kelly_adjusted_base * 1.2)
        elif quality_multiplier > 0.6:
            position_fraction = kelly_adjusted_base * 1.1
        elif quality_multiplier > 0.4:
            position_fraction = kelly_adjusted_base
        else:
            position_fraction = kelly_adjusted_base * 0.9
        
        # Ensure position size is within bounds (minimum 1% for proper allocation)
        position_fraction = max(0.01, min(self.max_position_size, position_fraction))
        
        return position_fraction
    
    def check_volatility_filter(self, df: pd.DataFrame, current_idx: int, timestamp: str) -> bool:
        """Optimized volatility filtering for faster backtesting while maintaining effectiveness."""
        if not self.use_volatility_filter:
            return True
        
        # Check liquidity hours (avoid low liquidity periods)
        try:
            # Extract hour from timestamp
            if ' ' in timestamp:
                time_part = timestamp.split(' ')[0]  # Get time part (HH:MM)
                hour = int(time_part.split(':')[0])
            else:
                # If timestamp format is different, skip hour check
                hour = 12  # Default to midday
            
            if hour in self.min_liquidity_hours:
                return False  # Don't trade during low liquidity hours
        except (ValueError, IndexError):
            # If timestamp parsing fails, continue with volatility check
            pass
        
        # Need minimum data for volatility analysis
        if current_idx < 20:  # Reduced from 50 to 20 for faster processing
            return True  # Allow trading if not enough data
        
        # 1. Primary volatility check (20-period) - most important filter
        recent_prices = df['close'].iloc[current_idx-19:current_idx+1]
        price_changes = recent_prices.pct_change().dropna()
        current_volatility = price_changes.std()
        
        # Don't trade if volatility is too high (primary filter)
        if current_volatility > self.volatility_threshold:
            return False
        
        # 2. Price range analysis (quick check)
        recent_high = df['high'].iloc[current_idx-4:current_idx+1].max()
        recent_low = df['low'].iloc[current_idx-4:current_idx+1].min()
        price_range_pct = (recent_high - recent_low) / recent_low
        
        # Don't trade if price range is too large (relaxed threshold)
        if price_range_pct > (self.volatility_threshold * 3.0):  # Increased from 2.0 to 3.0 for more opportunities
            return False
        
        # 3. ATR-based volatility filter (if available, use pre-calculated) - relaxed
        if 'atr_pct' in df.columns and current_idx > 0:
            current_atr_pct = df['atr_pct'].iloc[current_idx]
            # Don't trade if ATR is in extreme territory (>5% for crypto) - relaxed from 3%
            if current_atr_pct > 0.05:
                return False
        
        # 4. Optional: Volatility trend check (only if enough data and not too expensive) - relaxed
        if current_idx >= 50 and current_idx % 10 == 0:  # Check every 10th iteration (reduced frequency)
            # Compare recent 20-period volatility with previous 20-period
            recent_20_vol = current_volatility
            previous_20_vol = df['close'].iloc[current_idx-39:current_idx-19].pct_change().std()
            
            if previous_20_vol > 0:
                volatility_trend = recent_20_vol / previous_20_vol
                # Don't trade if volatility is increasing rapidly (>100% increase) - relaxed from 60%
                if volatility_trend > 2.0:
                    return False
        
        return True
    
    def update_drawdown_tracking(self, current_portfolio_value: float):
        """Update drawdown tracking."""
        if current_portfolio_value > self.peak_capital:
            self.peak_capital = current_portfolio_value
            self.trading_halted = False  # Reset trading halt if we reach new peak
        
        self.current_drawdown = (current_portfolio_value - self.peak_capital) / self.peak_capital
    
    def check_drawdown_limit(self) -> bool:
        """Check if drawdown limit has been exceeded."""
        return abs(self.current_drawdown) >= self.max_drawdown_limit
    
    def check_daily_stop_loss(self, timestamp: str, current_portfolio_value: float) -> bool:
        """Check if daily stop-loss limit has been exceeded."""
        # Extract date from timestamp
        try:
            current_date = timestamp.split(' ')[0]  # Get date part
        except:
            return False
        
        # Reset daily tracking if new day
        if self.current_date != current_date:
            self.current_date = current_date
            self.daily_start_capital = current_portfolio_value
            self.daily_loss = 0.0
        
        # Calculate daily loss
        self.daily_loss = (self.daily_start_capital - current_portfolio_value) / self.daily_start_capital
        
        # Check if daily stop-loss exceeded
        return self.daily_loss >= self.daily_stop_loss
    
    
    def check_volatility_regime_filter(self, row: pd.Series) -> bool:
        """Check if current volatility regime is suitable for trading."""
        if not self.use_volatility_regime_filter:
            return True
        
        # Get log return and volume zscore from the row for volatility assessment
        log_return = row.get('log_return_current', 0)
        volume_zscore = row.get('volume_zscore_current', 0)
        
        # DEBUG: Log filter values (only for first few iterations to avoid spam)
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 0
            
        if self._debug_count < 5:  # Log first 5 checks
            print(f"    🔍 Volatility Filter Debug: log_return={log_return:.4f}, volume_zscore={volume_zscore:.4f}")
            print(f"    🔍 Thresholds: log_return_max=0.05, volume_zscore_max=3.0")
        
        # Skip trades if log return is too extreme (noisy conditions)
        if abs(log_return) > 0.05:  # Skip if log return > 5% (too volatile)
            if self._debug_count < 5:
                print(f"    ❌ REJECTED: Log return too extreme ({log_return:.4f} > 0.05)")
            return False
        
        # Check volume zscore - must be reasonable for crypto
        if abs(volume_zscore) > 3.0:  # Skip if volume is too extreme (3+ standard deviations)
            if self._debug_count < 5:
                print(f"    ❌ REJECTED: Volume zscore too extreme ({volume_zscore:.4f} > 3.0)")
            return False
        
        if self._debug_count < 5:
            print(f"    ✅ PASSED: Volatility regime filter")
        return True
    
    def force_liquidate_positions(self, timestamp: str, reason: str = 'MARGIN_CALL') -> List[Dict[str, Any]]:
        """Force liquidate all positions due to margin call."""
        liquidated_trades = []
        
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            # Use current market price (simplified - in reality would need current price)
            current_price = position['entry_price']  # Simplified for demo
            
            exit_trade = self._execute_exit(symbol, current_price, timestamp, reason, 0, 0)
            if exit_trade:
                liquidated_trades.append(exit_trade)
                if reason == 'MARGIN_CALL':
                    self.margin_calls += 1
        
        return liquidated_trades
    
    def generate_leveraged_signal(self, row: pd.Series, symbol: str = '') -> Dict[str, Any]:
        """Generate high-quality signal optimized for leveraged trading with cross-asset analysis."""
        # Get target values (now calculated directly in backtest_symbol)
        target_direction = row.get('target_direction', 0)
        target_return = row.get('target_return_1', 0)
        
        # Handle case where target values might be NaN (end of dataset)
        if pd.isna(target_direction):
            target_direction = 0
        if pd.isna(target_return):
            target_return = 0
        
        # Technical indicators (using current values from parquet file)
        rsi_14 = row.get('rsi_14_current', 50)  # RSI 14
        sma_20 = row.get('sma_20_current', 0)
        sma_100 = row.get('sma_100_current', 0)
        volume_zscore = row.get('volume_zscore_current', 0)
        log_return = row.get('log_return_current', 0)
        
        # Calculate MA slope from SMA 20 and SMA 100
        ma_slope_20 = (sma_20 - sma_100) / sma_100 if sma_100 != 0 else 0
        
        
        # Initialize signal components
        signal_strength = 0.0
        confidence = 0.0
        
        # Enhanced signal generation for leveraged trading
        if target_direction == 1 and target_return > self.min_target_return:
            signal_strength = 0.5
            confidence = 0.4
            
            # RSI confirmation (more conservative for leverage)
            if rsi_14 < 25:  # Extreme oversold
                signal_strength += 0.4
                confidence += 0.3
            elif rsi_14 < 30:  # Very oversold
                signal_strength += 0.3
                confidence += 0.25
            elif rsi_14 < 35:  # Oversold
                signal_strength += 0.2
                confidence += 0.15
            elif rsi_14 > 70:  # Overbought - strong penalty for leverage
                signal_strength -= 0.3
                confidence -= 0.2
            
            # Strong trend confirmation using SMA 20 vs SMA 100
            if ma_slope_20 > 0.05:  # Very strong uptrend (SMA 20 > 5% above SMA 100)
                signal_strength += 0.3
                confidence += 0.25
            elif ma_slope_20 > 0.02:  # Strong uptrend (SMA 20 > 2% above SMA 100)
                signal_strength += 0.25
                confidence += 0.2
            elif ma_slope_20 > 0.01:  # Moderate uptrend (SMA 20 > 1% above SMA 100)
                signal_strength += 0.15
                confidence += 0.1
            elif ma_slope_20 < -0.01:  # Downtrend - penalty (SMA 20 < 1% below SMA 100)
                signal_strength -= 0.25
                confidence -= 0.15
            
            # Volume surge confirmation using volume zscore
            if volume_zscore > 2.5:  # Extreme volume (2.5+ standard deviations)
                signal_strength += 0.25
                confidence += 0.2
            elif volume_zscore > 1.5:  # High volume (1.5+ standard deviations)
                signal_strength += 0.2
                confidence += 0.15
            elif volume_zscore > 0.5:  # Moderate volume (0.5+ standard deviations)
                signal_strength += 0.1
                confidence += 0.1
            elif volume_zscore < -0.5:  # Low volume - penalty (below average)
                signal_strength -= 0.2
                confidence -= 0.15
            
            # Log return analysis for momentum confirmation
            if log_return > 0.02:  # Strong positive momentum (2%+ log return)
                signal_strength += 0.2
                confidence += 0.15
            elif log_return > 0.01:  # Moderate positive momentum (1%+ log return)
                signal_strength += 0.15
                confidence += 0.1
            elif log_return > 0.005:  # Weak positive momentum (0.5%+ log return)
                signal_strength += 0.1
                confidence += 0.05
            elif log_return < -0.01:  # Negative momentum - penalty
                signal_strength -= 0.3
                confidence -= 0.2
            
                
        elif target_direction == 0 and target_return < -self.min_target_return:
            signal_strength = -0.5
            confidence = 0.4
            
            # RSI confirmation for shorts (more conservative)
            if rsi_14 > 75:  # Extreme overbought
                signal_strength -= 0.4
                confidence += 0.3
            elif rsi_14 > 70:  # Very overbought
                signal_strength -= 0.3
                confidence += 0.25
            elif rsi_14 > 65:  # Overbought
                signal_strength -= 0.2
                confidence += 0.15
            elif rsi_14 < 30:  # Oversold - penalty for shorts
                signal_strength += 0.3
                confidence -= 0.2
            
            # Strong downtrend confirmation using SMA 20 vs SMA 100
            if ma_slope_20 < -0.05:  # Very strong downtrend (SMA 20 < 5% below SMA 100)
                signal_strength -= 0.3
                confidence += 0.25
            elif ma_slope_20 < -0.02:  # Strong downtrend (SMA 20 < 2% below SMA 100)
                signal_strength -= 0.25
                confidence += 0.2
            elif ma_slope_20 < -0.01:  # Moderate downtrend (SMA 20 < 1% below SMA 100)
                signal_strength -= 0.15
                confidence += 0.1
            elif ma_slope_20 > 0.01:  # Uptrend - penalty for shorts (SMA 20 > 1% above SMA 100)
                signal_strength += 0.25
                confidence -= 0.15
            
            # Volume confirmation for shorts using volume zscore
            if volume_zscore > 2.5:  # Extreme volume (2.5+ standard deviations)
                signal_strength -= 0.25
                confidence += 0.2
            elif volume_zscore > 1.5:  # High volume (1.5+ standard deviations)
                signal_strength -= 0.2
                confidence += 0.15
            elif volume_zscore > 0.5:  # Moderate volume (0.5+ standard deviations)
                signal_strength -= 0.1
                confidence += 0.1
            elif volume_zscore < -0.5:  # Low volume - penalty (below average)
                signal_strength += 0.2
                confidence -= 0.15
            
            # Log return analysis for shorts momentum
            if log_return < -0.02:  # Strong negative momentum (2%+ negative log return)
                signal_strength -= 0.2
                confidence += 0.15
            elif log_return < -0.01:  # Moderate negative momentum (1%+ negative log return)
                signal_strength -= 0.15
                confidence += 0.1
            elif log_return < -0.005:  # Weak negative momentum (0.5%+ negative log return)
                signal_strength -= 0.1
                confidence += 0.05
            elif log_return > 0.01:  # Positive momentum - penalty for shorts
                signal_strength += 0.3
                confidence -= 0.2
            
        
        # Normalize and apply quality filters
        signal_strength = np.clip(signal_strength, -1.0, 1.0)
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # Determine final signal with very high thresholds for leverage
        if abs(signal_strength) >= self.min_signal_strength and confidence >= self.min_confidence:
            signal = 'BUY' if signal_strength > 0 else 'SELL'
        elif abs(signal_strength) >= 0.2 and confidence >= 0.2:  # Very relaxed weak signal thresholds
            signal = 'WEAK_BUY' if signal_strength > 0 else 'WEAK_SELL'
        else:
            signal = 'HOLD'
            signal_strength = 0.0
        
        return {
            'signal': signal,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'target_return': target_return,
            'rsi_14': rsi_14,
            'sma_20': sma_20,
            'sma_100': sma_100,
            'ma_slope_20': ma_slope_20,
            'volume_zscore': volume_zscore,
            'log_return': log_return
        }

    def generate_exit_signal(self, row: pd.Series, symbol: str, current_position: Dict) -> Dict[str, Any]:
        """
        Generate exit signal based on trend reversal detection.
        This is used for trend-following behavior instead of fixed take-profit.
        """
        if not self.use_trend_following:
            return {'exit_signal': 'HOLD', 'exit_strength': 0.0, 'exit_confidence': 0.0}
        
        # Get current position direction
        position_size = current_position.get('size', 0)
        is_long_position = position_size > 0
        
        # Generate current market signal using same logic as entry
        current_signal_data = self.generate_leveraged_signal(row, symbol)
        current_signal = current_signal_data.get('signal', 'HOLD')
        current_signal_strength = current_signal_data.get('signal_strength', 0.0)
        current_confidence = current_signal_data.get('confidence', 0.0)
        
        # Determine exit signal based on position direction and current signal
        exit_signal = 'HOLD'
        exit_strength = 0.0
        exit_confidence = 0.0
        
        if is_long_position:
            # For long positions, exit on SELL signals or trend reversal
            if current_signal in ['SELL', 'WEAK_SELL']:
                exit_signal = 'SELL'
                exit_strength = abs(current_signal_strength)
                exit_confidence = current_confidence
            elif current_signal == 'HOLD' and current_confidence < 0.2:
                # Exit if trend weakens significantly (low confidence)
                exit_signal = 'SELL'
                exit_strength = 0.3  # Moderate exit strength
                exit_confidence = 0.2
        else:
            # For short positions, exit on BUY signals or trend reversal
            if current_signal in ['BUY', 'WEAK_BUY']:
                exit_signal = 'BUY'
                exit_strength = abs(current_signal_strength)
                exit_confidence = current_confidence
            elif current_signal == 'HOLD' and current_confidence < 0.2:
                # Exit if trend weakens significantly (low confidence)
                exit_signal = 'BUY'
                exit_strength = 0.3  # Moderate exit strength
                exit_confidence = 0.2
        
        # Additional trend reversal detection using technical indicators
        rsi_14 = row.get('rsi_14_current', 50)
        sma_20 = row.get('sma_20_current', 0)
        sma_100 = row.get('sma_100_current', 0)
        ma_slope_20 = (sma_20 - sma_100) / sma_100 if sma_100 != 0 else 0
        
        if is_long_position:
            # For long positions, check for bearish reversal signals
            if rsi_14 > 70 and ma_slope_20 < -0.01:  # Overbought + declining MA (SMA 20 below SMA 100)
                if exit_signal == 'HOLD':
                    exit_signal = 'SELL'
                    exit_strength = 0.4
                    exit_confidence = 0.3
                else:
                    # Strengthen existing exit signal
                    exit_strength = max(exit_strength, 0.4)
                    exit_confidence = max(exit_confidence, 0.3)
        else:
            # For short positions, check for bullish reversal signals
            if rsi_14 < 30 and ma_slope_20 > 0.01:  # Oversold + rising MA (SMA 20 above SMA 100)
                if exit_signal == 'HOLD':
                    exit_signal = 'BUY'
                    exit_strength = 0.4
                    exit_confidence = 0.3
                else:
                    # Strengthen existing exit signal
                    exit_strength = max(exit_strength, 0.4)
                    exit_confidence = max(exit_confidence, 0.3)
        
        return {
            'exit_signal': exit_signal,
            'exit_strength': exit_strength,
            'exit_confidence': exit_confidence,
            'current_signal': current_signal,
            'current_signal_strength': current_signal_strength,
            'current_confidence': current_confidence,
            'rsi_14': rsi_14,
            'sma_20': sma_20,
            'sma_100': sma_100,
            'ma_slope_20': ma_slope_20
        }
    
    def manage_leveraged_position(self, symbol: str, current_price: float, 
                                 high: float, low: float, timestamp: str, row: pd.Series = None) -> Optional[Dict[str, Any]]:
        """Advanced position management for leveraged trading with trend-following exits."""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        position_size = position['size']
        
        # Update high/low tracking
        if 'highest_price' not in position:
            position['highest_price'] = current_price
            position['lowest_price'] = current_price
        
        position['highest_price'] = max(position['highest_price'], high)
        position['lowest_price'] = min(position['lowest_price'], low)
        
        # Calculate current metrics
        current_return = (current_price - entry_price) / entry_price
        mfe = (position['highest_price'] - entry_price) / entry_price
        mae = (entry_price - position['lowest_price']) / entry_price
        
        # Update position with current market value and unrealized P&L
        current_market_value = position_size * current_price
        entry_cost = position_size * entry_price
        unrealized_pnl = current_market_value - entry_cost
        
        position['value'] = current_market_value
        position['unrealized_pnl'] = unrealized_pnl
        
        # 1. ALWAYS check stop-loss for risk management (if enabled)
        if self.use_fixed_stop_loss and current_price <= position['stop_loss']:
            return self._execute_exit(symbol, current_price, timestamp, 'STOP_LOSS', mfe, mae)
        
        # 2. Check trend-following exit signals (if enabled)
        if self.use_trend_following and row is not None:
            exit_signal_data = self.generate_exit_signal(row, symbol, position)
            exit_signal = exit_signal_data.get('exit_signal', 'HOLD')
            exit_strength = exit_signal_data.get('exit_strength', 0.0)
            exit_confidence = exit_signal_data.get('exit_confidence', 0.0)
            
            # Exit if we have a strong exit signal
            if exit_signal != 'HOLD' and exit_strength >= 0.3 and exit_confidence >= 0.2:
                return self._execute_exit(symbol, current_price, timestamp, 'TREND_REVERSAL', mfe, mae)
        
        # 3. Check fixed take-profit (only if trend-following is disabled)
        if self.use_fixed_take_profit and not self.use_trend_following:
            if current_price >= position['profit_target']:
                return self._execute_exit(symbol, current_price, timestamp, 'PROFIT_TARGET', mfe, mae)
        
        # 4. Check trailing stop (only if trend-following is disabled)
        if not self.use_trend_following and mfe > 0.005:  # Only apply trailing stop after 0.5% profit
            peak_profit_price = position['highest_price']
            trailing_stop_price = peak_profit_price * (1 - position['trailing_stop_pct'])
            
            if current_price <= trailing_stop_price:
                return self._execute_exit(symbol, current_price, timestamp, 'TRAILING_STOP', mfe, mae)
        
        return None
    
    def _execute_exit(self, symbol: str, price: float, timestamp: str, 
                     exit_type: str, mfe: float, mae: float) -> Dict[str, Any]:
        """Execute leveraged position exit."""
        position = self.positions[symbol]
        position_size = position['size']
        entry_price = position['entry_price']
        margin_used = position['margin_used']
        
        # Calculate trade details
        execution_price = price * (1 - self.slippage)
        trade_value = position_size * execution_price
        commission_cost = trade_value * self.commission
        
        entry_value = position_size * entry_price
        profit_loss = trade_value - entry_value - commission_cost
        trade_return_pct = profit_loss / margin_used if margin_used > 0 else 0
        
        # Update capital and margin
        self.capital += profit_loss
        self.used_margin -= margin_used
        self.available_margin += margin_used
        del self.positions[symbol]
        
        
        # Update statistics
        self.total_trades += 1
        if profit_loss > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        return {
            'timestamp': timestamp,
            'symbol': symbol,
            'signal': exit_type,
            'action': 'SELL',
            'price': execution_price,
            'position_size': -position_size,
            'trade_value': trade_value,
            'commission': commission_cost,
            'profit_loss': profit_loss,
            'trade_return_pct': trade_return_pct * 100,
            'margin_used': margin_used,
            'capital_before': self.capital - profit_loss,
            'capital_after': self.capital,
            'mfe': mfe,
            'mae': mae,
            'exit_reason': exit_type
        }
    
    def execute_leveraged_trade(self, timestamp: str, symbol: str, signal_data: Dict, price: float, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute leveraged trade with margin management."""
        signal = signal_data['signal']
        signal_strength = signal_data['signal_strength']
        confidence = signal_data['confidence']
        
        trade_info = {
            'timestamp': timestamp,
            'symbol': symbol,
            'signal': signal,
            'action': 'HOLD',
            'signal_strength': signal_strength,
            'confidence': confidence,
            'price': price
        }
        
        if signal in ['BUY', 'WEAK_BUY'] and symbol not in self.positions:
            
            # Calculate position size using Kelly fraction
            position_fraction = self.calculate_kelly_position_size(signal_strength, confidence)
            
            # Calculate leveraged position
            position_value = self.capital * position_fraction * self.leverage
            position_size = position_value / price
            margin_required = self.calculate_margin_requirements(position_value)
            
            # Check margin availability
            if margin_required <= self.available_margin:
                # Apply costs
                execution_price = price * (1 + self.slippage)
                trade_value = position_size * execution_price
                commission_cost = trade_value * self.commission
                
                # Calculate ATR-based stop levels
                atr_levels = self.calculate_atr_stop_levels(df, execution_price, True)
                
                # Create leveraged position
                self.positions[symbol] = {
                    'size': position_size,
                    'entry_price': execution_price,
                    'value': trade_value,
                    'margin_used': margin_required,
                    'entry_time': timestamp,
                    'highest_price': execution_price,
                    'lowest_price': execution_price,
                    'stop_loss': atr_levels['stop_loss'],
                    'profit_target': atr_levels['profit_target'],
                    'trailing_stop_pct': atr_levels['trailing_stop'],
                    'atr_value': atr_levels.get('atr_value', 0),
                    'unrealized_pnl': 0.0  # Initialize unrealized P&L
                }
                
                # Update margin
                self.used_margin += margin_required
                self.available_margin -= margin_required
                self.capital -= commission_cost
                
                trade_info.update({
                    'action': 'BUY',
                    'position_size': position_size,
                    'trade_value': trade_value,
                    'commission': commission_cost,
                    'margin_used': margin_required,
                    'leverage': self.leverage,
                    'capital_after': self.capital,
                    'available_margin': self.available_margin
                })
        
        return trade_info
    
    def add_signal_columns_to_csv(self, csv_file_path: str, symbol: str) -> str:
        """
        Add 'target_signal' and 'target_position_size' columns to CSV file.
        
        Args:
            csv_file_path: Path to the CSV file
            symbol: Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            Path to the updated CSV file
        """
        print(f"📊 Adding signal columns to {symbol} CSV...")
        
        # Read the CSV file
        try:
            df = pd.read_csv(csv_file_path, low_memory=False)
            print(f"   Loaded {len(df)} rows from {csv_file_path}")
        except Exception as e:
            print(f"   ❌ Error reading CSV: {e}")
            return csv_file_path
        
        # Initialize new columns
        df['target_signal'] = 'HOLD'
        df['target_position_size'] = 0.0
        
        # Process each row to generate signals and position sizes
        signals_generated = 0
        positions_calculated = 0
        
        for idx, row in df.iterrows():
            if idx % 10000 == 0 and idx > 0:
                print(f"   Processing row {idx}/{len(df)}...")
            
            try:
                # Generate signal using existing logic
                signal_data = self.generate_leveraged_signal(row, symbol)
                signal = signal_data.get('signal', 'HOLD')
                signal_strength = signal_data.get('signal_strength', 0.0)
                confidence = signal_data.get('confidence', 0.0)
                
                # Calculate position size using existing logic
                if signal != 'HOLD':
                    # Use simplified position sizing for CSV processing (no trade history available)
                    position_size = self.calculate_simple_position_size(
                        signal_strength, confidence
                    )
                    positions_calculated += 1
                else:
                    position_size = 0.0
                
                # Update the dataframe
                df.at[idx, 'target_signal'] = signal
                df.at[idx, 'target_position_size'] = position_size
                
                if signal != 'HOLD':
                    signals_generated += 1
                    
            except Exception as e:
                # If there's an error, keep default values
                df.at[idx, 'target_signal'] = 'HOLD'
                df.at[idx, 'target_position_size'] = 0.0
                continue
        
        # Create output filename
        base_name = os.path.splitext(csv_file_path)[0]
        output_path = f"{base_name}_with_signals.csv"
        
        # Save the updated CSV
        try:
            df.to_csv(output_path, index=False)
            print(f"   ✅ Updated CSV saved to: {output_path}")
            print(f"   📈 Generated {signals_generated} signals out of {len(df)} rows")
            print(f"   💰 Calculated {positions_calculated} position sizes")
            
            # Print signal distribution
            signal_counts = df['target_signal'].value_counts()
            print(f"   📊 Signal distribution:")
            for signal, count in signal_counts.items():
                percentage = (count / len(df)) * 100
                print(f"      {signal}: {count} ({percentage:.1f}%)")
            
            return output_path
            
        except Exception as e:
            print(f"   ❌ Error saving CSV: {e}")
            return csv_file_path

    def add_signal_columns_to_all_csvs(self, data_dir: str = 'features_with_residuals') -> Dict[str, str]:
        """
        Add signal columns to all CSV files in the specified directory.
        
        Args:
            data_dir: Directory containing CSV files
            
        Returns:
            Dictionary mapping original file paths to updated file paths
        """
        print(f"🚀 Adding signal columns to all CSV files in {data_dir}...")
        
        if not os.path.exists(data_dir):
            print(f"   ❌ Directory {data_dir} does not exist")
            return {}
        
        # Find all CSV files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'features_with_residuals' in f]
        
        if not csv_files:
            print(f"   ❌ No CSV files found in {data_dir}")
            return {}
        
        print(f"   📁 Found {len(csv_files)} CSV files to process")
        
        updated_files = {}
        
        for csv_file in csv_files:
            # Extract symbol from filename
            symbol = csv_file.replace('_features_with_residuals.csv', '')
            
            # Full path to the CSV file
            csv_path = os.path.join(data_dir, csv_file)
            
            print(f"\n📊 Processing {symbol}...")
            
            # Add signal columns
            updated_path = self.add_signal_columns_to_csv(csv_path, symbol)
            updated_files[csv_path] = updated_path
        
        print(f"\n✅ Successfully processed {len(updated_files)} CSV files")
        return updated_files
    
    def calculate_target_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate target features directly from OHLCV data to avoid dependency on pre-calculated target columns."""
        print("  🎯 Calculating target features directly from OHLCV data...")
        
        # Validate required columns
        if 'close' not in df.columns:
            raise ValueError("Missing 'close' column required for target calculation")
        
        # Calculate base returns
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_15'] = df['close'].pct_change(15)
        
        # Calculate log returns (handle division by zero)
        df['log_return'] = np.where(
            df['close'].shift(1) > 0,
            np.log(df['close'] / df['close'].shift(1)),
            0
        )
        
        # Create target features (shifted forward by 1 bar to prevent lookahead bias)
        target_features = ['return_1', 'return_5', 'return_15', 'log_return']
        
        for feature in target_features:
            if feature in df.columns:
                df[f'target_{feature}'] = df[feature].shift(-1)
        
        # Create target direction (binary: 1 if next return > 0, 0 otherwise)
        # Handle NaN values at the end of the dataset
        next_return = df['return_1'].shift(-1)
        df['target_direction'] = np.where(
            pd.isna(next_return), 
            0,  # Default to 0 for NaN values (end of dataset)
            np.where(next_return > 0, 1, 0)
        )
        
        # Fill NaN values in target columns with 0 (for the last row)
        for feature in target_features:
            target_col = f'target_{feature}'
            if target_col in df.columns:
                df[target_col] = df[target_col].fillna(0)
        
        print(f"  ✅ Target features calculated: target_direction, target_return_1, target_return_5, target_return_15, target_log_return")
        print(f"  📊 Target direction distribution: {df['target_direction'].value_counts().to_dict()}")
        return df

    def backtest_symbol(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Run leveraged backtest."""
        print(f"🔄 Leveraged backtesting {symbol} (5x leverage)...")
        
        self.reset()
        
        # Prepare data
        try:
            timestamp_col = self.get_timestamp_column(df)
        except ValueError as e:
            return {'error': str(e)}
        
        # Require basic OHLCV columns and technical indicators (using parquet column names)
        required_cols = [timestamp_col, 'close', 'high', 'low', 'open', 'volume']
        technical_cols = ['rsi_14_current', 'sma_20_current', 'sma_100_current', 'volume_zscore_current', 'log_return_current']
        all_required_cols = required_cols + technical_cols
        
        missing_cols = [col for col in all_required_cols if col not in df.columns]
        if missing_cols:
            return {'error': f'Missing required columns: {missing_cols}'}
        
        df['timestamp'] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['formatted_timestamp'] = df['timestamp'].dt.strftime('%H:%M %Y/%m/%d')
        
        # Calculate target features directly from OHLCV data
        df = self.calculate_target_features(df)
        
        # Track portfolio values
        initial_portfolio_value = self.capital
        peak_portfolio_value = self.capital
        
        # Run backtest
        for idx, row in df.iterrows():
            timestamp = row['formatted_timestamp']
            price = row['close']
            high = row['high']
            low = row['low']
            
            # Check for margin call
            if self.check_margin_call():
                liquidated_trades = self.force_liquidate_positions(timestamp)
                self.trades.extend(liquidated_trades)
                continue
            
            # Update all positions with current market values and unrealized P&L
            for pos_symbol, position in self.positions.items():
                if pos_symbol == symbol:  # Only update the current symbol's position
                    current_market_value = position['size'] * price
                    entry_cost = position['size'] * position['entry_price']
                    unrealized_pnl = current_market_value - entry_cost
                    
                    position['value'] = current_market_value
                    position['unrealized_pnl'] = unrealized_pnl
            
            # Manage leveraged positions (pass row data for trend-following exits)
            exit_trade = self.manage_leveraged_position(symbol, price, high, low, timestamp, row)
            if exit_trade:
                self.trades.append(exit_trade)
                continue
            
            # DEBUG: Track filter rejections
            debug_info = {
                'timestamp': timestamp,
                'symbol': symbol,
                'idx': idx,
                'rejected_by': []
            }
            
            # Check volatility filter before generating signal
            if not self.check_volatility_filter(df, idx, timestamp):
                debug_info['rejected_by'].append('volatility_filter')
                if idx % 10000 == 0:  # Log every 10k iterations
                    print(f"  🔍 DEBUG {symbol} idx={idx}: Rejected by volatility_filter")
                continue  # Skip trading during extreme noise or low liquidity
            
            # Check volatility regime filter before generating signal
            if not self.check_volatility_regime_filter(row):
                debug_info['rejected_by'].append('volatility_regime_filter')
                if idx % 10000 == 0:  # Log every 10k iterations
                    print(f"  🔍 DEBUG {symbol} idx={idx}: Rejected by volatility_regime_filter")
                continue  # Skip trading during high volatility regimes
            
            # Check drawdown limit - halt trading if exceeded
            if self.check_drawdown_limit():
                debug_info['rejected_by'].append('drawdown_limit')
                if idx % 10000 == 0:  # Log every 10k iterations
                    print(f"  🔍 DEBUG {symbol} idx={idx}: Rejected by drawdown_limit")
                self.trading_halted = True
                continue  # Skip trading if drawdown limit exceeded
            
            # Check daily stop-loss - halt trading if exceeded
            current_portfolio_value = self.capital + sum(pos.get('unrealized_pnl', 0) for pos in self.positions.values())
            if self.check_daily_stop_loss(timestamp, current_portfolio_value):
                debug_info['rejected_by'].append('daily_stop_loss')
                if idx % 10000 == 0:  # Log every 10k iterations
                    print(f"  🔍 DEBUG {symbol} idx={idx}: Rejected by daily_stop_loss")
                self.trading_halted = True
                continue  # Skip trading if daily stop-loss exceeded
            
            # Generate leveraged signal
            signal_data = self.generate_leveraged_signal(row, symbol)
            
            # DEBUG: Check signal generation
            if idx % 10000 == 0:  # Log every 10k iterations
                print(f"  🔍 DEBUG {symbol} idx={idx}: Signal={signal_data['signal']}, Strength={signal_data['signal_strength']:.3f}, Confidence={signal_data['confidence']:.3f}")
            
            # DEBUG: Track signal rejections
            if signal_data['signal'] == 'HOLD':
                debug_info['rejected_by'].append('signal_generation')
                if idx % 10000 == 0:  # Log every 10k iterations
                    print(f"  🔍 DEBUG {symbol} idx={idx}: Rejected by signal_generation (HOLD)")
            
            # Execute leveraged trade
            trade_info = self.execute_leveraged_trade(timestamp, symbol, signal_data, price, df)
            if trade_info['action'] != 'HOLD':
                self.trades.append(trade_info)
            
            # Update portfolio tracking - FIXED: Use unrealized P&L instead of incorrect margin calculation
            current_portfolio_value = self.capital + sum(pos.get('unrealized_pnl', 0) for pos in self.positions.values())
            if current_portfolio_value > peak_portfolio_value:
                peak_portfolio_value = current_portfolio_value
            self.portfolio_values.append(current_portfolio_value)
        
            # Update drawdown tracking
            self.update_drawdown_tracking(current_portfolio_value)
            
        
        # Calculate final metrics - FIXED: Use unrealized P&L instead of incorrect margin calculation
        final_portfolio_value = self.capital + sum(pos.get('unrealized_pnl', 0) for pos in self.positions.values())
        total_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value
        
        # Calculate performance metrics
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
        if len(self.portfolio_values) > 1:
            returns = pd.Series(self.portfolio_values).pct_change().dropna()
            if len(returns) > 0:
                if returns.std() > 0:
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                
                # Calculate Sortino ratio (downside deviation)
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    downside_deviation = negative_returns.std() * np.sqrt(252)
                    if downside_deviation > 0:
                        sortino_ratio = returns.mean() * np.sqrt(252) / downside_deviation
        
        max_drawdown = 0.0
        if len(self.portfolio_values) > 0:
            peak = np.maximum.accumulate(self.portfolio_values)
            drawdown = (self.portfolio_values - peak) / peak
            max_drawdown = np.min(drawdown)
        
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Calculate hit rate (approximate - percentage of profitable periods)
        hit_rate = 0.0
        if len(self.portfolio_values) > 1:
            portfolio_returns = pd.Series(self.portfolio_values).pct_change().dropna()
            profitable_periods = (portfolio_returns > 0).sum()
            hit_rate = profitable_periods / len(portfolio_returns) if len(portfolio_returns) > 0 else 0
        
        # Calculate exposure (average position size as percentage of portfolio)
        exposure = 0.0
        if len(self.portfolio_values) > 0:
            # Calculate average exposure from trade data
            total_exposure = 0.0
            exposure_periods = 0
            for trade in self.trades:
                if trade.get('action') == 'BUY' and 'margin_used' in trade:
                    # Calculate exposure as margin used / capital at time of trade
                    capital_at_trade = trade.get('capital_after', self.initial_capital) + trade.get('margin_used', 0)
                    if capital_at_trade > 0:
                        trade_exposure = trade['margin_used'] / capital_at_trade
                        total_exposure += trade_exposure
                        exposure_periods += 1
            
            exposure = total_exposure / exposure_periods if exposure_periods > 0 else 0
        
        # Calculate CVaR (Conditional Value at Risk) at 5% level
        cvar_5 = 0.0
        if len(self.portfolio_values) > 1:
            portfolio_returns = pd.Series(self.portfolio_values).pct_change().dropna()
            if len(portfolio_returns) > 0:
                # Calculate 5% VaR (Value at Risk)
                var_5 = np.percentile(portfolio_returns, 5)
                # Calculate CVaR as mean of returns below 5% VaR
                returns_below_var = portfolio_returns[portfolio_returns <= var_5]
                if len(returns_below_var) > 0:
                    cvar_5 = returns_below_var.mean()
        
        # Calculate average gain per trade (key metric)
        avg_gain_per_trade_pct = 0.0
        trade_returns = []
        for trade in self.trades:
            if trade.get('action') == 'SELL' and 'trade_return_pct' in trade:
                trade_returns.append(trade['trade_return_pct'])
        
        if trade_returns:
            avg_gain_per_trade_pct = np.mean(trade_returns)
        
        # MFE/MAE analysis
        mfe_values = [trade.get('mfe', 0) for trade in self.trades if trade.get('action') == 'SELL']
        mae_values = [trade.get('mae', 0) for trade in self.trades if trade.get('action') == 'SELL']
        avg_mfe = np.mean(mfe_values) if mfe_values else 0
        avg_mae = np.mean(mae_values) if mae_values else 0
        
        results = {
            'symbol': symbol,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'hit_rate': hit_rate,
            'exposure': exposure,
            'cvar_5': cvar_5,
            'avg_gain_per_trade_pct': avg_gain_per_trade_pct,
            'avg_mfe': avg_mfe,
            'avg_mae': avg_mae,
            'margin_calls': self.margin_calls,
            'leverage': self.leverage,
            'trades': self.trades.copy()
        }
        
        # Success criteria
        success_criteria = {
            'sharpe_ratio_ok': sharpe_ratio >= 1.2,
            'avg_gain_ok': avg_gain_per_trade_pct >= 1.0,
            'max_drawdown_ok': abs(max_drawdown) <= 0.015,
            'mae_control_ok': avg_mae <= 0.01,
            'overall_success': (sharpe_ratio >= 1.2 and avg_gain_per_trade_pct >= 1.0 and 
                              abs(max_drawdown) <= 0.015 and avg_mae <= 0.01)
        }
        
        print(f"  ✅ {symbol} leveraged backtest completed:")
        print(f"    📊 Total Return: {total_return:.2%}")
        print(f"    📈 Sharpe Ratio: {sharpe_ratio:.4f} {'✅' if success_criteria['sharpe_ratio_ok'] else '❌'}")
        print(f"    📉 Sortino Ratio: {sortino_ratio:.4f}")
        print(f"    📉 Max Drawdown: {max_drawdown:.2%} {'✅' if success_criteria['max_drawdown_ok'] else '❌'}")
        print(f"    🎯 Total Trades: {self.total_trades}")
        print(f"    🏆 Win Rate: {win_rate:.2%}")
        print(f"    🎯 Hit Rate: {hit_rate:.2%}")
        print(f"    💰 Avg Gain per Trade: {avg_gain_per_trade_pct:.4f}% {'✅' if success_criteria['avg_gain_ok'] else '❌'}")
        print(f"    📈 Avg MFE: {avg_mfe:.4f}")
        print(f"    📉 Avg MAE: {avg_mae:.4f} {'✅' if success_criteria['mae_control_ok'] else '❌'}")
        print(f"    📊 Exposure: {exposure:.2%}")
        print(f"    ⚠️  CVaR (5%): {cvar_5:.4f}")
        print(f"    ⚡ Margin Calls: {self.margin_calls}")
        print(f"    🏆 Success Criteria: {'✅ PASSED' if success_criteria['overall_success'] else '❌ FAILED'}")
        
        results['success_criteria'] = success_criteria
        return results


def clean_previous_results():
    """Clean up previous backtest result files."""
    print("🧹 Cleaning up previous backtest results...")
    
    # Clean up leveraged_backtest_results directory (main target)
    results_dir = 'leveraged_backtest_results'
    csv_count = 0
    
    if os.path.exists(results_dir):
        try:
            files_in_dir = os.listdir(results_dir)
            csv_files = [f for f in files_in_dir if f.endswith('.csv')]
            
            if csv_files:
                print(f"   📁 Found {len(csv_files)} CSV files in {results_dir}:")
                for file in csv_files:
                    file_path = os.path.join(results_dir, file)
                    os.remove(file_path)
                    print(f"   🗑️  Removed: {file}")
                    csv_count += 1
            else:
                print(f"   📁 No CSV files found in {results_dir}")
                
        except Exception as e:
            print(f"   ⚠️  Warning: Could not clean {results_dir}: {e}")
    else:
        print(f"   📁 Directory {results_dir} does not exist - will be created")
    
    # Clean up any backtest_results_*.csv files in the current directory
    try:
        current_dir_csvs = [f for f in os.listdir('.') if f.startswith('backtest_results_') and f.endswith('.csv')]
        for file in current_dir_csvs:
            os.remove(file)
            print(f"   🗑️  Removed: {file}")
            csv_count += 1
    except Exception as e:
        print(f"   ⚠️  Warning: Could not clean backtest_results files: {e}")
    
    # Clean up backtest_trades directory
    trades_dir = 'backtest_trades'
    if os.path.exists(trades_dir):
        try:
            trade_csvs = [f for f in os.listdir(trades_dir) if f.endswith('.csv')]
            for file in trade_csvs:
                file_path = os.path.join(trades_dir, file)
                os.remove(file_path)
                print(f"   🗑️  Removed: {file}")
                csv_count += 1
        except Exception as e:
            print(f"   ⚠️  Warning: Could not clean {trades_dir}: {e}")
    
    # Clean up any temporary CSV files
    try:
        temp_csvs = [f for f in os.listdir('.') if f.startswith('temp_') and f.endswith('.csv')]
        for file in temp_csvs:
            os.remove(file)
            print(f"   🗑️  Removed temporary file: {file}")
            csv_count += 1
    except Exception as e:
        print(f"   ⚠️  Warning: Could not clean temporary files: {e}")
    
    if csv_count > 0:
        print(f"   ✅ Cleanup completed! Removed {csv_count} CSV files")
    else:
        print("   ✅ Cleanup completed! No CSV files found to remove")

def clean_leveraged_results_only():
    """Clean up only the leveraged_backtest_results directory CSV files."""
    print("🧹 Cleaning up leveraged_backtest_results CSV files...")
    
    results_dir = 'leveraged_backtest_results'
    csv_count = 0
    
    if os.path.exists(results_dir):
        try:
            files_in_dir = os.listdir(results_dir)
            csv_files = [f for f in files_in_dir if f.endswith('.csv')]
            
            if csv_files:
                print(f"   📁 Found {len(csv_files)} CSV files in {results_dir}:")
                for file in csv_files:
                    file_path = os.path.join(results_dir, file)
                    os.remove(file_path)
                    print(f"   🗑️  Removed: {file}")
                    csv_count += 1
                print(f"   ✅ Removed {csv_count} CSV files from {results_dir}")
            else:
                print(f"   📁 No CSV files found in {results_dir}")
                
        except Exception as e:
            print(f"   ⚠️  Error cleaning {results_dir}: {e}")
            return False
    else:
        print(f"   📁 Directory {results_dir} does not exist - will be created")
    
    return True

def main():
    """Main function."""
    print("⚡ Leveraged Trading Signal Backtest System (5x Leverage)")
    print("=" * 60)
    print("🎯 TREND-FOLLOWING HIGH-RETURN STRATEGY WITH LEVERAGE:")
    print("  • 5x leverage for balanced returns")
    print("  • Balanced allocation: 10.0-20.0% per trade (50-100% exposure)")
    print("  • Drawdown limit: Stop trading if portfolio drawdown > 1.5% (strict risk control)")
    print("  • Daily stop-loss: Stop trading if daily loss > 1%")
    print("  • Volatility regime filter: Skip trades if ATR > 67.0% or rolling_vol outside 0.0-5.0% range")
    print("  • Volatility filter: Disabled for maximum trade frequency")
    print("  • ATR-based stops: 0.6× ATR(14) stop loss for risk management")
    print("  • Kelly fraction position sizing: FIXED - Optimizes between 10.0-20.0% positions (50-100% exposure) based on performance")
    print("  • Entry refinement: Signal strength ≥ 40%, Confidence ≥ 30%")
    print("  • Exit strategy: TREND-FOLLOWING - Hold positions until trend reversal signals (no fixed TP)")
    print("  • Trend reversal detection: Exit long on SELL signals, exit short on BUY signals")
    print("  • Success criteria: Sharpe ≥ 1.2, Avg gain ≥ 1-2%, Max DD ≤ 1.5%")
    print("=" * 60)
    
    # Clean up previous results before starting
    clean_previous_results()
    print()
    
    engine = LeveragedBacktestEngine()
    
    # Only backtest BTCUSDT and ETHUSDT using parquet files
    symbols = ['BTCUSDT', 'ETHUSDT']
    
    print(f"📊 Backtesting {len(symbols)} specific symbols: {symbols}")
    
    all_results = {}
    summary_results = []
    successful_assets = 0
    
    for symbol in symbols:
        # Use parquet files from feature directories
        if symbol == 'BTCUSDT':
            file_path = 'feature/btcusdt/btcusdt_features.parquet'
        elif symbol == 'ETHUSDT':
            file_path = 'feature/ethusdt/ethusdt_features.parquet'
        else:
            continue
        
        try:
            df = pd.read_parquet(file_path)
            print(f"\n📈 Loading {symbol} data from {file_path}: {len(df)} rows")
            
            results = engine.backtest_symbol(df, symbol)
            
            if 'error' not in results:
                all_results[symbol] = results
                
                if results.get('success_criteria', {}).get('overall_success', False):
                    successful_assets += 1
                
                summary_results.append({
                    'symbol': symbol,
                    'total_return': results['total_return'],
                    'sharpe_ratio': results['sharpe_ratio'],
                    'sortino_ratio': results['sortino_ratio'],
                    'max_drawdown': results['max_drawdown'],
                    'total_trades': results['total_trades'],
                    'win_rate': results['win_rate'],
                    'hit_rate': results['hit_rate'],
                    'exposure': results['exposure'],
                    'cvar_5': results['cvar_5'],
                    'avg_gain_per_trade_pct': results['avg_gain_per_trade_pct'],
                    'avg_mfe': results['avg_mfe'],
                    'avg_mae': results['avg_mae'],
                    'margin_calls': results['margin_calls'],
                    'success_criteria_met': results.get('success_criteria', {}).get('overall_success', False)
                })
                
            else:
                print(f"  ❌ {symbol} backtest failed: {results['error']}")
                all_results[symbol] = results
        
        except Exception as e:
            print(f"  ❌ Error processing {symbol}: {str(e)}")
            all_results[symbol] = {'error': str(e)}
    
    # Summary
    if summary_results:
        os.makedirs('leveraged_backtest_results', exist_ok=True)
        
        # Save per-asset summary files with naming like btc_summary.csv, eth_summary.csv
        for result in summary_results:
            sym = result.get('symbol', '')
            sym_lower = str(sym).lower()
            abbr = sym_lower.replace('usdt', '')  # e.g., btcusdt -> btc
            summary_path = os.path.join('leveraged_backtest_results', f"{abbr}_summary.csv")
            pd.DataFrame([result]).to_csv(summary_path, index=False)
            print(f"📄 Summary saved: {summary_path}")
        
        # Collect all trades across symbols
        all_trades = []
        for symbol, results in all_results.items():
            if 'error' not in results and 'trades' in results:
                for trade in results['trades']:
                    trade['symbol'] = symbol
                    all_trades.append(trade)
        
        # Save per-asset trades files with naming like btcusdt_trades.csv, ethusdt_trades.csv
        if all_trades:
            trades_df_all = pd.DataFrame(all_trades)
            print(f"   📈 Total trades recorded: {len(all_trades)}")
            
            # Print trade summary by symbol and write per-symbol files
            print(f"\n📊 Trade Summary by Symbol:")
            for symbol in symbols:
                sym_lower = str(symbol).lower()
                symbol_trades_df = trades_df_all[trades_df_all['symbol'] == symbol]
                if not symbol_trades_df.empty:
                    buy_trades = (symbol_trades_df['action'] == 'BUY').sum()
                    sell_trades = (symbol_trades_df['action'] == 'SELL').sum()
                    print(f"   {symbol}: {int(buy_trades)} entries, {int(sell_trades)} exits")
                    trades_path = os.path.join('leveraged_backtest_results', f"{sym_lower}_trades.csv")
                    symbol_trades_df.to_csv(trades_path, index=False)
                    print(f"📊 Trades saved: {trades_path}")
        
        print("\n" + "=" * 60)
        print("⚡ LEVERAGED BACKTEST SUMMARY (5x Leverage)")
        print("=" * 60)
        print(f"🎯 Assets meeting success criteria: {successful_assets}/2 required")
        print("=" * 60)
        
        for result in summary_results:
            success_icon = "✅" if result['success_criteria_met'] else "❌"
            print(f"📈 {result['symbol']} {success_icon}:")
            print(f"  Return: {result['total_return']:.2%}")
            print(f"  Sharpe: {result['sharpe_ratio']:.4f}")
            print(f"  Sortino: {result['sortino_ratio']:.4f}")
            print(f"  Drawdown: {result['max_drawdown']:.2%}")
            print(f"  Trades: {result['total_trades']}")
            print(f"  Win Rate: {result['win_rate']:.2%}")
            print(f"  Hit Rate: {result['hit_rate']:.2%}")
            print(f"  Exposure: {result['exposure']:.2%}")
            print(f"  CVaR (5%): {result['cvar_5']:.4f}")
            print(f"  Avg Gain/Trade: {result['avg_gain_per_trade_pct']:.4f}%")
            print(f"  Avg MFE: {result['avg_mfe']:.4f}")
            print(f"  Avg MAE: {result['avg_mae']:.4f}")
            print(f"  Margin Calls: {result['margin_calls']}")
            print()
    
    print(f"🎉 Leveraged backtest completed!")
    print(f"📁 Results saved to: leveraged_backtest_results")


def add_signals_to_csvs():
    """
    Example function to add signal columns to all CSV files.
    This demonstrates how to use the new signal column functionality.
    """
    print("🚀 Adding Signal Columns to CSV Files")
    print("=" * 50)
    
    # Clean up previous results before starting
    clean_previous_results()
    print()
    
    # Initialize the backtest engine
    engine = LeveragedBacktestEngine()
    
    # Add signal columns to specific parquet files (BTCUSDT and ETHUSDT only)
    updated_files = {}
    
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        # Use parquet files from feature directories
        if symbol == 'BTCUSDT':
            file_path = 'feature/btcusdt/btcusdt_features.parquet'
        elif symbol == 'ETHUSDT':
            file_path = 'feature/ethusdt/ethusdt_features.parquet'
        else:
            continue
        
        if os.path.exists(file_path):
            print(f"\n📊 Processing {symbol} from {file_path}...")
            # Convert parquet to CSV for signal processing, then convert back
            df = pd.read_parquet(file_path)
            temp_csv_path = f"temp_{symbol}_features.csv"
            df.to_csv(temp_csv_path, index=False)
            
            updated_path = engine.add_signal_columns_to_csv(temp_csv_path, symbol)
            updated_files[file_path] = updated_path
            
            # Clean up temporary CSV file
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
        else:
            print(f"   ❌ File not found: {file_path}")
    
    if updated_files:
        print(f"\n📊 Summary of Updated Files:")
        for original_path, updated_path in updated_files.items():
            symbol = os.path.basename(original_path).replace('_features.parquet', '')
            print(f"   {symbol}: {updated_path}")
        
        print(f"\n✅ Successfully added signal columns to {len(updated_files)} files")
        print("📁 New files have '_with_signals' suffix")
        print("📈 New columns added: 'target_signal', 'target_position_size'")
    else:
        print("❌ No files were processed")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--add-signals":
            add_signals_to_csvs()
        elif sys.argv[1] == "--clean":
            # Clean only leveraged_backtest_results CSV files
            clean_leveraged_results_only()
        elif sys.argv[1] == "--clean-all":
            # Clean all previous results
            clean_previous_results()
        else:
            print("Usage: python backtester.py [--add-signals|--clean|--clean-all]")
            print("  --add-signals: Add signal columns to CSV files")
            print("  --clean: Clean only leveraged_backtest_results CSV files")
            print("  --clean-all: Clean all previous result files")
            sys.exit(1)
    else:
        main()
