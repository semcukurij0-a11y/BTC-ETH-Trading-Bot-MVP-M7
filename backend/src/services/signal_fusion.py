"""
Signal Fusion Module for Crypto Trading Bot

This module combines multiple trading signals with hysteresis to reduce churn.
Features:
- Combines s_ml, s_sent, s_ta signals in [-1, +1] range
- Integrates fg (Fear & Greed) in [0, 1] range
- Deterministic scoring with configurable weights
- Hysteresis logic to reduce signal churn
- Default formula: s = 0.45*s_ml + 0.20*s_sent + 0.25*s_ta + 0.10*(2*fg-1)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SignalFusionModule:
    """
    Signal Fusion Module for combining multiple trading signals.
    
    Implements:
    - Multi-signal combination with weights
    - Hysteresis logic to reduce churn
    - Confidence scoring
    - Entry/exit signal generation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Signal Fusion Module.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Signal weights (default from specification)
        self.s_ml_weight = self.config.get('s_ml_weight', 0.45)
        self.s_sent_weight = self.config.get('s_sent_weight', 0.20)
        self.s_ta_weight = self.config.get('s_ta_weight', 0.25)
        self.fg_weight = self.config.get('fg_weight', 0.10)
        
        # Hysteresis parameters
        self.hysteresis_enabled = self.config.get('hysteresis_enabled', True)
        self.hysteresis_threshold = self.config.get('hysteresis_threshold', 0.1)
        self.hysteresis_buffer = self.config.get('hysteresis_buffer', 0.05)
        
        # Entry/exit thresholds
        self.entry_threshold_long = self.config.get('entry_threshold_long', 0.35)
        self.entry_threshold_short = self.config.get('entry_threshold_short', -0.35)
        self.exit_threshold_long = self.config.get('exit_threshold_long', -0.10)
        self.exit_threshold_short = self.config.get('exit_threshold_short', 0.10)
        
        # Confidence thresholds
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.confidence_weight_signal_strength = self.config.get('confidence_weight_signal_strength', 0.7)
        self.confidence_weight_signal_agreement = self.config.get('confidence_weight_signal_agreement', 0.3)
        
        # State tracking for hysteresis
        self.current_position = 'HOLD'  # 'LONG', 'SHORT', 'HOLD'
        self.last_signal_strength = 0.0
        self.signal_history = []
    
    def calculate_signal_agreement(self, signals: Dict[str, float]) -> float:
        """
        Calculate agreement between different signals.
        
        Args:
            signals: Dictionary of signal values
            
        Returns:
            Agreement score [0, 1]
        """
        if len(signals) < 2:
            return 0.5
        
        signal_values = list(signals.values())
        
        # Calculate how many signals agree on direction
        positive_signals = sum(1 for s in signal_values if s > 0.1)
        negative_signals = sum(1 for s in signal_values if s < -0.1)
        neutral_signals = len(signal_values) - positive_signals - negative_signals
        
        total_signals = len(signal_values)
        
        # Agreement is the maximum of positive/negative consensus
        max_consensus = max(positive_signals, negative_signals)
        agreement = max_consensus / total_signals
        
        return agreement
    
    def calculate_signal_strength(self, signals: Dict[str, float]) -> float:
        """
        Calculate overall signal strength.
        
        Args:
            signals: Dictionary of signal values
            
        Returns:
            Signal strength [0, 1]
        """
        if not signals:
            return 0.0
        
        # Calculate weighted average signal strength
        weighted_sum = 0.0
        total_weight = 0.0
        
        for signal_name, signal_value in signals.items():
            if signal_name == 'fg':
                # Fear & Greed is in [0, 1], convert to [-1, 1]
                normalized_value = 2 * signal_value - 1
                weight = self.fg_weight
            else:
                normalized_value = signal_value
                if signal_name == 's_ml':
                    weight = self.s_ml_weight
                elif signal_name == 's_sent':
                    weight = self.s_sent_weight
                elif signal_name == 's_ta':
                    weight = self.s_ta_weight
                else:
                    weight = 0.0
            
            weighted_sum += normalized_value * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Normalize and calculate strength
        normalized_signal = weighted_sum / total_weight
        signal_strength = abs(normalized_signal)
        
        return signal_strength
    
    def apply_hysteresis(self, current_signal: float, current_strength: float) -> Tuple[float, str]:
        """
        Apply hysteresis logic to reduce signal churn.
        
        Args:
            current_signal: Current combined signal
            current_strength: Current signal strength
            
        Returns:
            Tuple of (adjusted_signal, position)
        """
        if not self.hysteresis_enabled:
            return current_signal, self._determine_position(current_signal)
        
        # Calculate signal change
        signal_change = abs(current_signal - self.last_signal_strength)
        
        # Apply hysteresis buffer
        if self.current_position == 'LONG':
            # For long positions, require stronger negative signal to exit
            if current_signal < self.exit_threshold_long:
                if signal_change > self.hysteresis_threshold:
                    self.current_position = 'HOLD'
                    return current_signal, 'HOLD'
                else:
                    return self.last_signal_strength, 'LONG'
            elif current_signal > self.entry_threshold_long:
                return current_signal, 'LONG'
            else:
                return self.last_signal_strength, 'LONG'
        
        elif self.current_position == 'SHORT':
            # For short positions, require stronger positive signal to exit
            if current_signal > self.exit_threshold_short:
                if signal_change > self.hysteresis_threshold:
                    self.current_position = 'HOLD'
                    return current_signal, 'HOLD'
                else:
                    return self.last_signal_strength, 'SHORT'
            elif current_signal < self.entry_threshold_short:
                return current_signal, 'SHORT'
            else:
                return self.last_signal_strength, 'SHORT'
        
        else:  # HOLD position
            # For hold positions, require strong signal to enter
            if current_signal > self.entry_threshold_long:
                if signal_change > self.hysteresis_threshold:
                    self.current_position = 'LONG'
                    return current_signal, 'LONG'
                else:
                    return 0.0, 'HOLD'
            elif current_signal < self.entry_threshold_short:
                if signal_change > self.hysteresis_threshold:
                    self.current_position = 'SHORT'
                    return current_signal, 'SHORT'
                else:
                    return 0.0, 'HOLD'
            else:
                return 0.0, 'HOLD'
    
    def _determine_position(self, signal: float) -> str:
        """Determine position based on signal value."""
        if signal > self.entry_threshold_long:
            return 'LONG'
        elif signal < self.entry_threshold_short:
            return 'SHORT'
        else:
            return 'HOLD'
    
    def calculate_confidence(self, signals: Dict[str, float], signal_strength: float) -> float:
        """
        Calculate confidence score for the combined signal.
        
        Args:
            signals: Dictionary of signal values
            signal_strength: Overall signal strength
            
        Returns:
            Confidence score [0, 1]
        """
        # Signal strength component
        strength_confidence = signal_strength
        
        # Signal agreement component
        agreement_confidence = self.calculate_signal_agreement(signals)
        
        # Combined confidence
        confidence = (
            strength_confidence * self.confidence_weight_signal_strength +
            agreement_confidence * self.confidence_weight_signal_agreement
        )
        
        return min(confidence, 1.0)
    
    def fuse_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fuse multiple signals into a single trading signal.
        
        Args:
            df: DataFrame with individual signals
            
        Returns:
            DataFrame with fused signals
        """
        # Initialize fused signal columns
        df['s_fused'] = 0.0
        df['s_fused_strength'] = 0.0
        df['s_fused_confidence'] = 0.0
        df['position'] = 'HOLD'
        df['signal_change'] = 0.0
        
        if df.empty:
            return df
        
        for idx, row in df.iterrows():
            # Collect available signals
            signals = {}
            
            if 's_ml' in df.columns and not pd.isna(row['s_ml']):
                signals['s_ml'] = row['s_ml']
            
            if 's_sent' in df.columns and not pd.isna(row['s_sent']):
                signals['s_sent'] = row['s_sent']
            
            if 's_ta' in df.columns and not pd.isna(row['s_ta']):
                signals['s_ta'] = row['s_ta']
            
            if 'fear_greed' in df.columns and not pd.isna(row['fear_greed']):
                signals['fg'] = row['fear_greed']
            
            if not signals:
                continue
            
            # Calculate signal strength
            signal_strength = self.calculate_signal_strength(signals)
            
            # Calculate combined signal using default formula
            s_ml = signals.get('s_ml', 0.0)
            s_sent = signals.get('s_sent', 0.0)
            s_ta = signals.get('s_ta', 0.0)
            fg = signals.get('fg', 0.5)  # Default to neutral
            
            # Default formula: s = 0.45*s_ml + 0.20*s_sent + 0.25*s_ta + 0.10*(2*fg-1)
            combined_signal = (
                s_ml * self.s_ml_weight +
                s_sent * self.s_sent_weight +
                s_ta * self.s_ta_weight +
                (2 * fg - 1) * self.fg_weight
            )
            
            # Apply hysteresis
            adjusted_signal, position = self.apply_hysteresis(combined_signal, signal_strength)
            
            # Calculate confidence
            confidence = self.calculate_confidence(signals, signal_strength)
            
            # Update state
            signal_change = abs(adjusted_signal - self.last_signal_strength)
            self.last_signal_strength = adjusted_signal
            
            # Store results
            df.at[idx, 's_fused'] = adjusted_signal
            df.at[idx, 's_fused_strength'] = signal_strength
            df.at[idx, 's_fused_confidence'] = confidence
            df.at[idx, 'position'] = position
            df.at[idx, 'signal_change'] = signal_change
        
        return df
    
    def get_latest_fused_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get the latest fused signal.
        
        Args:
            df: DataFrame with fused signals
            
        Returns:
            Dictionary with latest signal information
        """
        if df.empty:
            return {
                's_fused': 0.0,
                's_fused_strength': 0.0,
                's_fused_confidence': 0.0,
                'position': 'HOLD',
                'signal': 'HOLD',
                'timestamp': datetime.now().isoformat()
            }
        
        latest = df.iloc[-1]
        
        # Determine signal direction
        if latest['s_fused'] > self.entry_threshold_long:
            signal = 'BUY'
        elif latest['s_fused'] < self.entry_threshold_short:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            's_fused': float(latest['s_fused']),
            's_fused_strength': float(latest['s_fused_strength']),
            's_fused_confidence': float(latest['s_fused_confidence']),
            'position': str(latest['position']),
            'signal': signal,
            'signal_change': float(latest.get('signal_change', 0)),
            's_ml': float(latest.get('s_ml', 0)),
            's_sent': float(latest.get('s_sent', 0)),
            's_ta': float(latest.get('s_ta', 0)),
            'fear_greed': float(latest.get('fear_greed', 0.5)),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_fusion_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get signal fusion summary statistics.
        
        Args:
            df: DataFrame with fused signals
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}
        
        return {
            'total_signals': len(df),
            'long_positions': len(df[df['position'] == 'LONG']),
            'short_positions': len(df[df['position'] == 'SHORT']),
            'hold_positions': len(df[df['position'] == 'HOLD']),
            'avg_s_fused': float(df['s_fused'].mean()),
            'avg_confidence': float(df['s_fused_confidence'].mean()),
            'avg_signal_strength': float(df['s_fused_strength'].mean()),
            'avg_signal_change': float(df['signal_change'].mean()),
            'hysteresis_enabled': self.hysteresis_enabled,
            'current_position': self.current_position,
            'signal_weights': {
                's_ml': self.s_ml_weight,
                's_sent': self.s_sent_weight,
                's_ta': self.s_ta_weight,
                'fg': self.fg_weight
            }
        }
    
    def reset_hysteresis_state(self):
        """Reset hysteresis state."""
        self.current_position = 'HOLD'
        self.last_signal_strength = 0.0
        self.signal_history = []
        self.logger.info("Hysteresis state reset")


def main():
    """Test the Signal Fusion Module."""
    import sys
    sys.path.append('src')
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = {
        's_ml_weight': 0.45,
        's_sent_weight': 0.20,
        's_ta_weight': 0.25,
        'fg_weight': 0.10,
        'hysteresis_enabled': True,
        'hysteresis_threshold': 0.1,
        'hysteresis_buffer': 0.05,
        'entry_threshold_long': 0.3,
        'entry_threshold_short': -0.3,
        'exit_threshold_long': -0.2,
        'exit_threshold_short': 0.2,
        'min_confidence': 0.6
    }
    
    # Initialize signal fusion module
    fusion_module = SignalFusionModule(config=config)
    
    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open_time': dates,
        's_ml': np.random.uniform(-1, 1, len(dates)),
        's_sent': np.random.uniform(-1, 1, len(dates)),
        's_ta': np.random.uniform(-1, 1, len(dates)),
        'fear_greed': np.random.uniform(0, 1, len(dates))
    })
    
    # Run signal fusion
    result_df = fusion_module.fuse_signals(sample_data)
    
    # Get latest signal
    latest_signal = fusion_module.get_latest_fused_signal(result_df)
    
    print("Signal Fusion Test Results:")
    print(f"Latest s_fused signal: {latest_signal['s_fused']:.4f}")
    print(f"Position: {latest_signal['position']}")
    print(f"Signal: {latest_signal['signal']}")
    print(f"Confidence: {latest_signal['s_fused_confidence']:.4f}")
    print(f"Signal Strength: {latest_signal['s_fused_strength']:.4f}")
    
    # Get summary
    summary = fusion_module.get_fusion_summary(result_df)
    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()
