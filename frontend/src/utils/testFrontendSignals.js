/**
 * Test script to verify frontend signal calculation
 * Run this in the browser console to test the signal fusion
 */

// Test the current signal values you mentioned
function testCurrentSignalValues() {
    console.log('🧪 Testing Current Signal Values...');
    
    // Your current values
    const currentSignal = {
        fused_signal: -0.784,
        ml: -0.471,
        technical: -0.235,
        sentiment: -0.078,
        fear_greed: 0.763
    };
    
    console.log('Current signal values:', currentSignal);
    
    // Calculate what the weighted result should be
    const ml = currentSignal.ml;
    const technical = currentSignal.technical;
    const sentiment = currentSignal.sentiment;
    const fear_greed = currentSignal.fear_greed;
    
    // Convert fear_greed from [0,1] to [-1,1] range
    const normalized_fg = (2 * fear_greed) - 1;
    
    // Apply the fusion formula: s = 0.45*s_ml + 0.20*s_sent + 0.25*s_ta + 0.10*(2*fg-1)
    const weighted_result = (
        ml * 0.45 +           // 45% ML influence
        sentiment * 0.20 +    // 20% Sentiment influence
        technical * 0.25 +     // 25% Technical influence
        normalized_fg * 0.10  // 10% Fear & Greed influence
    );
    
    console.log('Weighted calculation:');
    console.log(`  ML: ${ml} * 0.45 = ${(ml * 0.45).toFixed(6)}`);
    console.log(`  Sentiment: ${sentiment} * 0.20 = ${(sentiment * 0.20).toFixed(6)}`);
    console.log(`  Technical: ${technical} * 0.25 = ${(technical * 0.25).toFixed(6)}`);
    console.log(`  Fear/Greed: ${normalized_fg.toFixed(6)} * 0.10 = ${(normalized_fg * 0.10).toFixed(6)}`);
    console.log(`  Total weighted result: ${weighted_result.toFixed(6)}`);
    
    console.log('Comparison:');
    console.log(`  Current fused signal: ${currentSignal.fused_signal}`);
    console.log(`  Calculated weighted: ${weighted_result.toFixed(6)}`);
    console.log(`  Difference: ${Math.abs(currentSignal.fused_signal - weighted_result).toFixed(6)}`);
    
    if (Math.abs(currentSignal.fused_signal - weighted_result) < 0.001) {
        console.log('✅ Current signal matches weighted calculation');
    } else {
        console.log('❌ Current signal does NOT match weighted calculation');
        console.log('This means the weighting is not being applied correctly');
    }
}

// Test the mock signals from TradingBotService
function testMockSignals() {
    console.log('\n🧪 Testing Mock Signals...');
    
    // Test the mock signals that should be generated
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
}

// Test the signal fusion calculator
function testSignalFusionCalculator() {
    console.log('\n🧪 Testing Signal Fusion Calculator...');
    
    try {
        // Try to import the signal fusion calculator
        const { defaultSignalFusion } = require('./signalFusion');
        
        const testComponents = {
            ml: 0.456,
            technical: 0.234,
            sentiment: 0.145,
            fear_greed: 0.089
        };
        
        const result = defaultSignalFusion.processSignal('BTCUSDT', testComponents);
        console.log('Signal fusion calculator result:', result);
        
    } catch (error) {
        console.log('Signal fusion calculator not available:', error.message);
    }
}

// Run all tests
function runAllTests() {
    console.log('🚀 Running Frontend Signal Tests...');
    console.log('=' * 50);
    
    testCurrentSignalValues();
    testMockSignals();
    testSignalFusionCalculator();
    
    console.log('\n✅ All Frontend Signal Tests Completed');
}

// Auto-run if in browser
if (typeof window !== 'undefined') {
    console.log('🌐 Running in browser environment');
    runAllTests();
} else {
    console.log('🖥️ Running in Node.js environment');
    runAllTests();
}
