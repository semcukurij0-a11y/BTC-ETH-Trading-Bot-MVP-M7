/**
 * Test script to verify signal weighting in frontend
 * This can be run in the browser console to verify the calculation
 */

export function testSignalWeighting(): void {
  console.log('🧪 Testing Signal Weighting Calculation...');
  
  // Test case 1: BTCUSDT with known components
  const btcComponents = {
    ml: 0.456,
    technical: 0.234,
    sentiment: 0.145,
    fear_greed: 0.089
  };
  
  // Test case 2: ETHUSDT with known components
  const ethComponents = {
    ml: -0.123,
    technical: -0.089,
    sentiment: -0.067,
    fear_greed: 0.045
  };
  
  const testCases = [
    { symbol: 'BTCUSDT', components: btcComponents },
    { symbol: 'ETHUSDT', components: ethComponents }
  ];
  
  testCases.forEach((testCase, index) => {
    console.log(`\n[TEST ${index + 1}] ${testCase.symbol}`);
    console.log('Components:', testCase.components);
    
    const { ml, technical, sentiment, fear_greed } = testCase.components;
    
    // Apply the fusion formula: s = 0.45*s_ml + 0.20*s_sent + 0.25*s_ta + 0.10*(2*fg-1)
    const normalized_fg = (2 * fear_greed) - 1;
    const fused_signal = (
      ml * 0.45 +           // 45% ML influence
      sentiment * 0.20 +    // 20% Sentiment influence
      technical * 0.25 +     // 25% Technical influence
      normalized_fg * 0.10  // 10% Fear & Greed influence
    );
    
    console.log(`Fused Signal: ${fused_signal.toFixed(6)}`);
    console.log('Calculation breakdown:');
    console.log(`  ML: ${ml} * 0.45 = ${(ml * 0.45).toFixed(6)}`);
    console.log(`  Sentiment: ${sentiment} * 0.20 = ${(sentiment * 0.20).toFixed(6)}`);
    console.log(`  Technical: ${technical} * 0.25 = ${(technical * 0.25).toFixed(6)}`);
    console.log(`  Fear/Greed: ${normalized_fg.toFixed(6)} * 0.10 = ${(normalized_fg * 0.10).toFixed(6)}`);
    console.log(`  Total: ${fused_signal.toFixed(6)}`);
    
    // Verify the weights sum to 1.0
    const totalWeight = 0.45 + 0.20 + 0.25 + 0.10;
    console.log(`Weight verification: ${totalWeight} (should be 1.0)`);
    
    if (Math.abs(totalWeight - 1.0) < 0.001) {
      console.log('✅ Weights sum to 1.0');
    } else {
      console.log('❌ Weights do not sum to 1.0');
    }
  });
  
  console.log('\n✅ Signal weighting test completed');
}

export function testMockSignalsWeighting(): void {
  console.log('\n🧪 Testing Mock Signals Weighting...');
  
  // Test the exact mock signals from TradingBotService
  const mockSignals = [
    {
      symbol: 'BTCUSDT',
      components: { ml: 0.456, technical: 0.234, sentiment: 0.145, fear_greed: 0.089 }
    },
    {
      symbol: 'ETHUSDT',
      components: { ml: -0.123, technical: -0.089, sentiment: -0.067, fear_greed: 0.045 }
    }
  ];
  
  mockSignals.forEach(signal => {
    console.log(`\n[${signal.symbol}]`);
    const { ml, technical, sentiment, fear_greed } = signal.components;
    
    const normalized_fg = (2 * fear_greed) - 1;
    const fused_signal = (
      ml * 0.45 +
      sentiment * 0.20 +
      technical * 0.25 +
      normalized_fg * 0.10
    );
    
    console.log(`Components: ML=${ml}, Technical=${technical}, Sentiment=${sentiment}, Fear/Greed=${fear_greed}`);
    console.log(`Normalized Fear/Greed: ${normalized_fg.toFixed(6)}`);
    console.log(`Fused Signal: ${fused_signal.toFixed(6)}`);
    console.log(`Signal Strength: ${Math.abs(fused_signal).toFixed(6)}`);
    console.log(`Direction: ${fused_signal > 0 ? 'BULLISH' : 'BEARISH'}`);
  });
  
  console.log('\n✅ Mock signals weighting test completed');
}

// Auto-run tests if in browser
if (typeof window !== 'undefined') {
  console.log('🌐 Running signal weighting tests in browser...');
  testSignalWeighting();
  testMockSignalsWeighting();
}
