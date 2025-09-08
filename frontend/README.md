# 🌾 Agricultural AI Dashboard - Frontend

**FAANG-Level React Frontend** for Organic Agriculture Agentic AI System

## 🚀 **Quick Start**

### **Prerequisites**
- Node.js 16+ 
- npm or yarn
- Backend API running on `http://localhost:8000`

### **Installation & Setup**
```bash
# Install dependencies
npm install

# Start development server
npm start

# Open browser to http://localhost:3000
```

### **Environment Configuration**
Create a `.env` file in the frontend directory:
```env
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000/ws
REACT_APP_MAPBOX_TOKEN=your_mapbox_token_here
```

## 🏗️ **Architecture**

### **Technology Stack**
- **Framework:** React 18 + TypeScript
- **State Management:** React Query + Zustand
- **Styling:** Tailwind CSS + Framer Motion
- **Charts:** D3.js + Recharts
- **AI/ML:** TensorFlow.js
- **Icons:** Heroicons
- **HTTP Client:** Axios

### **Project Structure**
```
src/
├── components/
│   ├── ui/                    # Reusable UI components
│   │   ├── Button.tsx
│   │   ├── Card.tsx
│   │   ├── Badge.tsx
│   │   └── Input.tsx
│   ├── auth/                  # Authentication components
│   │   └── LoginForm.tsx
│   ├── dashboard/             # Dashboard components
│   │   ├── Dashboard.tsx
│   │   ├── Header.tsx
│   │   ├── Sidebar.tsx
│   │   ├── Footer.tsx
│   │   ├── WeatherDashboard.tsx
│   │   ├── MarketIntelligence.tsx
│   │   ├── PestDetection.tsx
│   │   └── YieldPrediction.tsx
│   └── charts/                # Chart components
│       ├── WeatherChart.tsx
│       └── PriceChart.tsx
├── contexts/                  # React contexts
│   ├── AppProvider.tsx
│   └── AuthContext.tsx
├── services/                  # API services
│   ├── api.ts
│   └── websocket.ts
├── types/                     # TypeScript types
│   └── api.ts
├── config/                    # Configuration
│   └── environment.ts
└── App.tsx                    # Main app component
```

## 🎨 **Features**

### **Dashboard Components**
- **Weather Intelligence:** Real-time weather monitoring with D3.js charts
- **Market Analysis:** Price predictions and market trends
- **Pest Detection:** AI-powered pest and disease monitoring
- **Yield Prediction:** ML-based yield forecasting
- **Real-time Updates:** WebSocket integration for live data

### **UI/UX Features**
- **Responsive Design:** Mobile-first approach
- **Smooth Animations:** Framer Motion micro-interactions
- **Dark/Light Theme:** Tailwind CSS theming
- **Accessibility:** WCAG 2.1 compliant
- **Performance:** Optimized with React Query caching

### **Authentication**
- **JWT Token Management:** Secure authentication
- **Auto-refresh:** Token renewal handling
- **Protected Routes:** Route-based access control
- **User Management:** Profile and settings

## 🔌 **API Integration**

### **Available Endpoints**
- **Authentication:** `/auth/login`, `/auth/me`
- **Weather:** `/agents/weather`
- **Market:** `/agents/market`
- **Pest Detection:** `/agents/pest`
- **Yield Prediction:** `/agents/prediction`
- **Decision Support:** `/agents/decision`
- **Metrics:** `/metrics`
- **Health Check:** `/health`

### **Real-time Updates**
- **WebSocket:** `ws://localhost:8000/ws`
- **Message Types:** `weather_update`, `pest_alert`, `market_update`, `system_alert`

## 🎯 **Usage Examples**

### **Weather Dashboard**
```typescript
import { WeatherDashboard } from './components/dashboard/WeatherDashboard';

<WeatherDashboard farmId="farm_123" />
```

### **API Client Usage**
```typescript
import apiClient from './services/api';

// Get weather data
const weatherData = await apiClient.getWeatherAnalysis({
  farm_id: 'farm_123',
  analysis_type: 'current',
  timeframe: '7d'
});

// Get market data
const marketData = await apiClient.getMarketAnalysis({
  crop_type: 'wheat',
  region: 'North America',
  analysis_type: 'price_prediction',
  timeframe: '6m'
});
```

### **WebSocket Integration**
```typescript
import websocketService from './services/websocket';

// Subscribe to weather updates
const unsubscribe = websocketService.subscribe('weather_update', (data) => {
  console.log('Weather update:', data);
});

// Send message
websocketService.send({ type: 'ping', data: {} });
```

## 🎨 **Design System**

### **Color Palette**
- **Primary Green:** `#22c55e` (Agriculture, Growth)
- **Secondary Blue:** `#3b82f6` (Information, Sky)
- **Warning Yellow:** `#eab308` (Caution, Sun)
- **Error Red:** `#ef4444` (Alerts, Critical)
- **Info Purple:** `#8b5cf6` (AI, Technology)

### **Typography**
- **Primary Font:** Inter (Google Fonts)
- **Monospace:** JetBrains Mono
- **Scale:** 12px - 48px (responsive)

### **Components**
- **Buttons:** 5 variants (primary, secondary, outline, ghost, danger)
- **Cards:** Hover effects, shadows, padding variants
- **Badges:** Status indicators with animations
- **Inputs:** Form controls with validation

## 🚀 **Development**

### **Available Scripts**
```bash
npm start          # Start development server
npm run build      # Build for production
npm test           # Run tests
npm run eject      # Eject from Create React App
```

### **Code Quality**
- **TypeScript:** Full type safety
- **ESLint:** Code linting
- **Prettier:** Code formatting
- **Husky:** Git hooks

### **Performance**
- **Code Splitting:** Lazy loading
- **Memoization:** React.memo, useMemo
- **Caching:** React Query
- **Bundle Analysis:** Webpack Bundle Analyzer

## 🔧 **Configuration**

### **Tailwind CSS**
```javascript
// tailwind.config.js
module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        green: { /* Custom green palette */ },
        brown: { /* Earth tones */ },
        blue: { /* Sky colors */ }
      }
    }
  }
}
```

### **React Query**
```typescript
// Query configuration
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    }
  }
});
```

## 🧪 **Testing**

### **Test Setup**
```bash
# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

### **Test Files**
- **Unit Tests:** `*.test.tsx`
- **Integration Tests:** `*.integration.test.tsx`
- **E2E Tests:** `cypress/integration/*.spec.js`

## 📱 **Responsive Design**

### **Breakpoints**
- **Mobile:** 320px - 640px
- **Tablet:** 640px - 1024px
- **Desktop:** 1024px - 1280px
- **Large Desktop:** 1280px+

### **Grid System**
- **Mobile:** 1 column
- **Tablet:** 2 columns
- **Desktop:** 3 columns
- **Large Desktop:** 4 columns

## 🚀 **Deployment**

### **Production Build**
```bash
npm run build
```

### **Environment Variables**
```env
REACT_APP_API_BASE_URL=https://api.yourapp.com
REACT_APP_WS_URL=wss://ws.yourapp.com
REACT_APP_MAPBOX_TOKEN=pk.your_token
```

### **Docker Support**
```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## 🤝 **Contributing**

### **Development Workflow**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### **Code Style**
- Use TypeScript for all new files
- Follow the existing component structure
- Add proper error handling
- Include JSDoc comments

## 📄 **License**

This project is part of the Organic Agriculture Agentic AI System.

## 🆘 **Support**

For support and questions:
- **Documentation:** `/docs/frontend/`
- **API Reference:** `/docs/api/backend_endpoints.md`
- **Issues:** GitHub Issues

---

**Built with ❤️ for the future of agriculture** 🌾