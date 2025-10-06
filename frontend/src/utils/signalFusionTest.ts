/**
 * Simple test for signal fusion calculation
 * This can be run in the browser console to verify the logic works
 */

// Inline signal fusion calculation (same as backend)
export function testSignalFusion(): void {
  console.log('🧪 Testing Signal Fusion Calculation...');
  
  // Test case 1: Strong Bullish Signal
  const test1 = {
    symbol: 'BTCUSDT',
    components: { ml: 0.8, technical: 0.7, sentiment: 0.6, fear_greed: 0.8 }
  };
  
  // Test case 2: Strong Bearish Signal  
  const test2 = {
    symbol: 'ETHUSDT',
    components: { ml: -0.8, technical: -0.7, sentiment: -0.6, fear_greed: 0.2 }
  };
  
  // Test case 3: Mixed Signal
  const test3 = {
    symbol: 'SOLUSDT', 
    components: { ml: 0.3, technical: -0.2, sentiment: 0.1, fear_greed: 0.5 }
  };
  
  const testCases = [test1, test2, test3];
  
  testCases.forEach((testCase, index) => {
    console.log(`\n[TEST ${index + 1}] ${testCase.symbol}`);
    console.log('Components:', testCase.components);
    
    const { ml, technical, sentiment, fear_greed } = testCase.components;
    
    // Apply the fusion formula: s = 0.45*s_ml + 0.20*s_sent + 0.25*s_ta + 0.10*(2*fg-1)
    const normalized_fg = (2 * fear_greed) - 1;
    const fused_signal = (
      ml * 0.45 +
      sentiment * 0.20 +
      technical * 0.25 +
      normalized_fg * 0.10
    );
    
    // Calculate confidence (simplified)
    const confidence = Math.min(Math.abs(fused_signal) * 0.8 + 0.2, 1.0);
    
    console.log(`Fused Signal: ${fused_signal.toFixed(3)}`);
    console.log(`Confidence: ${confidence.toFixed(3)}`);
    console.log(`Normalized FG: ${normalized_fg.toFixed(3)}`);
    
    // Manual calculation breakdown
    console.log('Calculation breakdown:');
    console.log(`  ML component: ${ml} * 0.45 = ${(ml * 0.45).toFixed(3)}`);
    console.log(`  Sentiment component: ${sentiment} * 0.20 = ${(sentiment * 0.20).toFixed(3)}`);
    console.log(`  Technical component: ${technical} * 0.25 = ${(technical * 0.25).toFixed(3)}`);
    console.log(`  Fear/Greed component: ${normalized_fg} * 0.10 = ${(normalized_fg * 0.10).toFixed(3)}`);
    console.log(`  Total: ${fused_signal.toFixed(3)}`);
  });
  
  console.log('\n✅ Signal fusion test completed');
}

// Test the mock signals generation
export function testMockSignals(): any[] {
  console.log('🧪 Testing Mock Signals Generation...');
  
  const rawSignals = [
    {
      symbol: 'BTCUSDT',
      components: { ml: 0.456, technical: 0.234, sentiment: 0.145, fear_greed: 0.089 }
    },
    {
      symbol: 'ETHUSDT',
      components: { ml: -0.123, technical: -0.089, sentiment: -0.067, fear_greed: 0.045 }
    },
    {
      symbol: 'SOLUSDT',
      components: { ml: 0.234, technical: 0.123, sentiment: 0.067, fear_greed: 0.032 }
    }
  ];
  
  const processedSignals = rawSignals.map(signal => {
    const { ml, technical, sentiment, fear_greed } = signal.components;
    const normalized_fg = (2 * fear_greed) - 1;
    const fused_signal = (
      ml * 0.45 +
      sentiment * 0.20 +
      technical * 0.25 +
      normalized_fg * 0.10
    );
    const confidence = Math.min(Math.abs(fused_signal) * 0.8 + 0.2, 1.0);
    
    return {
      symbol: signal.symbol,
      signal: Math.round(fused_signal * 1000) / 1000,
      confidence: Math.round(confidence * 1000) / 1000,
      components: signal.components,
      timestamp: new Date().toISOString()
    };
  });
  
  console.log('Processed signals:', processedSignals);
  console.log('✅ Mock signals test completed');
  
  return processedSignals;
}

// Auto-run tests if in browser
if (typeof window !== 'undefined') {
  console.log('🌐 Running signal fusion tests in browser...');
  testSignalFusion();
  testMockSignals();
}
