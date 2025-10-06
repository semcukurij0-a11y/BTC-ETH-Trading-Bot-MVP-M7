/**
 * Test file for Signal Fusion Calculator
 * This file can be used to verify the frontend signal fusion logic
 */

import { SignalFusionCalculator, calculateFusedSignal, calculateSignalConfidence } from './signalFusion';

// Test cases for signal fusion calculation
const testCases = [
  {
    name: 'Strong Bullish Signal',
    components: { ml: 0.8, technical: 0.7, sentiment: 0.6, fear_greed: 0.8 },
    expectedRange: [0.6, 0.8]
  },
  {
    name: 'Strong Bearish Signal', 
    components: { ml: -0.8, technical: -0.7, sentiment: -0.6, fear_greed: 0.2 },
    expectedRange: [-0.8, -0.6]
  },
  {
    name: 'Mixed Signal',
    components: { ml: 0.3, technical: -0.2, sentiment: 0.1, fear_greed: 0.5 },
    expectedRange: [-0.1, 0.3]
  },
  {
    name: 'Neutral Signal',
    components: { ml: 0.0, technical: 0.0, sentiment: 0.0, fear_greed: 0.5 },
    expectedRange: [-0.1, 0.1]
  }
];

// Test the signal fusion calculator
export function testSignalFusionCalculator(): void {
  console.log('🧪 Testing Signal Fusion Calculator...');
  
  const calculator = new SignalFusionCalculator();
  
  testCases.forEach((testCase, index) => {
    console.log(`\n[TEST ${index + 1}] ${testCase.name}`);
    console.log(`Components:`, testCase.components);
    
    const fusedSignal = calculator.calculateFusedSignal(testCase.components);
    const confidence = calculator.calculateConfidence(testCase.components);
    const strength = calculator.calculateSignalStrength(testCase.components);
    
    console.log(`Fused Signal: ${fusedSignal.toFixed(3)}`);
    console.log(`Confidence: ${confidence.toFixed(3)}`);
    console.log(`Strength: ${strength.toFixed(3)}`);
    
    // Check if result is in expected range
    const [minExpected, maxExpected] = testCase.expectedRange;
    if (fusedSignal >= minExpected && fusedSignal <= maxExpected) {
      console.log(`✅ Signal value ${fusedSignal.toFixed(3)} is in expected range [${minExpected}, ${maxExpected}]`);
    } else {
      console.log(`❌ Signal value ${fusedSignal.toFixed(3)} is outside expected range [${minExpected}, ${maxExpected}]`);
    }
    
    // Verify confidence is reasonable
    if (confidence >= 0 && confidence <= 1) {
      console.log(`✅ Confidence ${confidence.toFixed(3)} is in valid range [0, 1]`);
    } else {
      console.log(`❌ Confidence ${confidence.toFixed(3)} is outside valid range [0, 1]`);
    }
  });
}

// Test the utility functions
export function testUtilityFunctions(): void {
  console.log('\n🧪 Testing Utility Functions...');
  
  const testComponents = { ml: 0.5, technical: 0.3, sentiment: 0.2, fear_greed: 0.7 };
  
  const fusedSignal = calculateFusedSignal(testComponents);
  const confidence = calculateSignalConfidence(testComponents);
  
  console.log(`Test Components:`, testComponents);
  console.log(`Fused Signal: ${fusedSignal.toFixed(3)}`);
  console.log(`Confidence: ${confidence.toFixed(3)}`);
  
  // Manual calculation for verification
  const manualResult = (
    0.5 * 0.45 +      // ml * 0.45
    0.2 * 0.20 +      // sentiment * 0.20
    0.3 * 0.25 +      // technical * 0.25
    (2 * 0.7 - 1) * 0.10  // (2 * fear_greed - 1) * 0.10
  );
  
  console.log(`Manual Calculation: ${manualResult.toFixed(3)}`);
  
  if (Math.abs(fusedSignal - manualResult) < 0.001) {
    console.log('✅ Utility function matches manual calculation');
  } else {
    console.log('❌ Utility function does not match manual calculation');
  }
}

// Test complete signal processing
export function testCompleteSignalProcessing(): void {
  console.log('\n🧪 Testing Complete Signal Processing...');
  
  const calculator = new SignalFusionCalculator();
  
  const testSignals = [
    { symbol: 'BTCUSDT', components: { ml: 0.6, technical: 0.4, sentiment: 0.3, fear_greed: 0.7 } },
    { symbol: 'ETHUSDT', components: { ml: -0.4, technical: -0.3, sentiment: -0.2, fear_greed: 0.3 } },
    { symbol: 'SOLUSDT', components: { ml: 0.2, technical: 0.1, sentiment: 0.1, fear_greed: 0.5 } }
  ];
  
  const processedSignals = calculator.processSignals(testSignals);
  
  console.log('Processed Signals:');
  processedSignals.forEach(signal => {
    console.log(`  ${signal.symbol}: signal=${signal.signal.toFixed(3)}, confidence=${signal.confidence.toFixed(3)}`);
  });
  
  // Verify all signals have required properties
  const allValid = processedSignals.every(signal => 
    typeof signal.signal === 'number' &&
    typeof signal.confidence === 'number' &&
    typeof signal.symbol === 'string' &&
    signal.components &&
    signal.timestamp
  );
  
  if (allValid) {
    console.log('✅ All processed signals have valid structure');
  } else {
    console.log('❌ Some processed signals have invalid structure');
  }
}

// Run all tests
export function runAllTests(): void {
  console.log('🚀 Running Signal Fusion Tests...');
  console.log('=' * 50);
  
  testSignalFusionCalculator();
  testUtilityFunctions();
  testCompleteSignalProcessing();
  
  console.log('\n✅ All Signal Fusion Tests Completed');
}

// Auto-run tests if this file is executed directly
if (typeof window !== 'undefined') {
  // Browser environment
  console.log('🌐 Running in browser environment');
  runAllTests();
} else {
  // Node.js environment
  console.log('🖥️ Running in Node.js environment');
  runAllTests();
}
