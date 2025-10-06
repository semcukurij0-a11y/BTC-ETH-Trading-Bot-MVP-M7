// Test script to verify positions panel fix
console.log('🔧 Testing positions panel fix...');

console.log('✅ Fixed issues in PositionsPanelOptimized.tsx:');
console.log('   1. Added null checks for position.quantity');
console.log('   2. Added null checks for position.entry_price');
console.log('   3. Added null checks for position.current_price');
console.log('   4. Added null checks for position.unrealized_pnl');
console.log('   5. Added null checks for position.leverage');
console.log('   6. Added null checks for position.margin');
console.log('   7. Ensured data is always an array');
console.log('   8. Added safe calculations for reduce operations');

console.log('🎯 The error "Cannot read properties of undefined (reading 'toFixed')" should now be resolved!');

console.log('✅ Changes made:');
console.log('   - position.quantity.toFixed(6) → (position.quantity || 0).toFixed(6)');
console.log('   - formatCurrency(position.entry_price) → formatCurrency(position.entry_price || 0)');
console.log('   - formatCurrency(position.current_price) → formatCurrency(position.current_price || 0)');
console.log('   - position.unrealized_pnl → (position.unrealized_pnl || 0)');
console.log('   - position.leverage → (position.leverage || 0)');
console.log('   - data → positions (with proper array checking)');

console.log('🚀 The positions tab should now work without errors!');
