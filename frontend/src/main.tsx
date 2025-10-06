import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import AppSmoothRefresh from './AppSmoothRefresh.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AppSmoothRefresh />
  </StrictMode>
);
