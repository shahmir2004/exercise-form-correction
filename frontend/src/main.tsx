import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

// Note: StrictMode disabled to prevent double-mounting of effects
// which causes issues with MediaPipe WASM initialization
// Re-enable for production to catch potential issues
ReactDOM.createRoot(document.getElementById('root')!).render(
  <App />
)
