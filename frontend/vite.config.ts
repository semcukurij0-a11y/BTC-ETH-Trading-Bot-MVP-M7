import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: '0.0.0.0', // Bind to all interfaces for VPN compatibility
    strictPort: true, // Fail if port is already in use
  },
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
});
