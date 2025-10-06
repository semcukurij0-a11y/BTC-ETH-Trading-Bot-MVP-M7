/**
 * Signal Fusion Utility
 * Implements the same signal fusion logic as the backend
 */

export interface SignalComponents {
  ml: number;
  technical: number;
  sentiment: number;
  fear_greed: number;
}

export interface SignalFusionConfig {
  s_ml_weight: number;
  s_sent_weight: number;
  s_ta_weight: number;
  fg_weight: number;
  min_confidence: number;
  confidence_weight_signal_strength: number;
  confidence_weight_signal_agreement: number;
}

export class SignalFusionCalculator {
  private config: SignalFusionConfig;

  constructor(config?: Partial<SignalFusionConfig>) {
    this.config = {
      s_ml_weight: 0.45,      // 45% ML influence
      s_sent_weight: 0.20,     // 20% Sentiment influence
      s_ta_weight: 0.25,       // 25% Technical influence
      fg_weight: 0.10,         // 10% Fear & Greed influence
      min_confidence: 0.6,
      confidence_weight_signal_strength: 0.7,
      confidence_weight_signal_agreement: 0.3,
      ...config
    };
  }

  /**
   * Calculate the fused signal using the same formula as backend
   * Formula: s = 0.45*s_ml + 0.20*s_sent + 0.25*s_ta + 0.10*(2*fg-1)
   */
  calculateFusedSignal(components: SignalComponents): number {
    const { ml, technical, sentiment, fear_greed } = components;
    
    // Convert fear_greed from [0,1] to [-1,1] range
    const normalized_fg = (2 * fear_greed) - 1;
    
    // Apply the fusion formula
    const fused_signal = (
      ml * this.config.s_ml_weight +
      sentiment * this.config.s_sent_weight +
      technical * this.config.s_ta_weight +
      normalized_fg * this.config.fg_weight
    );
    
    // Clamp to [-1, 1] range
    return Math.max(-1, Math.min(1, fused_signal));
  }

  /**
   * Calculate signal strength (absolute value of fused signal)
   */
  calculateSignalStrength(components: SignalComponents): number {
    const fused_signal = this.calculateFusedSignal(components);
    return Math.abs(fused_signal);
  }

  /**
   * Calculate signal agreement between different components
   */
  calculateSignalAgreement(components: SignalComponents): number {
    const values = [
      components.ml,
      components.sentiment,
      components.technical,
      (2 * components.fear_greed) - 1  // Convert fear_greed to [-1,1]
    ];
    
    // Count signals in each direction
    const positive_signals = values.filter(v => v > 0.1).length;
    const negative_signals = values.filter(v => v < -0.1).length;
    const neutral_signals = values.length - positive_signals - negative_signals;
    
    // Agreement is the maximum consensus
    const max_consensus = Math.max(positive_signals, negative_signals);
    return max_consensus / values.length;
  }

  /**
   * Calculate confidence score for the fused signal
   */
  calculateConfidence(components: SignalComponents): number {
    // Signal strength component
    const signal_strength = this.calculateSignalStrength(components);
    const strength_confidence = signal_strength;
    
    // Signal agreement component
    const agreement_confidence = this.calculateSignalAgreement(components);
    
    // Combined confidence
    const confidence = (
      strength_confidence * this.config.confidence_weight_signal_strength +
      agreement_confidence * this.config.confidence_weight_signal_agreement
    );
    
    return Math.min(confidence, 1.0);
  }

  /**
   * Process a complete signal with fusion calculation
   */
  processSignal(symbol: string, components: SignalComponents, timestamp?: string): {
    symbol: string;
    signal: number;
    confidence: number;
    components: SignalComponents;
    timestamp: string;
  } {
    const fused_signal = this.calculateFusedSignal(components);
    const confidence = this.calculateConfidence(components);
    
    return {
      symbol,
      signal: Math.round(fused_signal * 1000) / 1000, // Round to 3 decimal places
      confidence: Math.round(confidence * 1000) / 1000,
      components,
      timestamp: timestamp || new Date().toISOString()
    };
  }

  /**
   * Process multiple signals
   */
  processSignals(signals: Array<{ symbol: string; components: SignalComponents; timestamp?: string }>): Array<{
    symbol: string;
    signal: number;
    confidence: number;
    components: SignalComponents;
    timestamp: string;
  }> {
    return signals.map(signal => this.processSignal(signal.symbol, signal.components, signal.timestamp));
  }
}

// Default signal fusion calculator instance
export const defaultSignalFusion = new SignalFusionCalculator();

// Utility function for quick signal fusion
export function calculateFusedSignal(components: SignalComponents): number {
  return defaultSignalFusion.calculateFusedSignal(components);
}

// Utility function for quick confidence calculation
export function calculateSignalConfidence(components: SignalComponents): number {
  return defaultSignalFusion.calculateConfidence(components);
}
