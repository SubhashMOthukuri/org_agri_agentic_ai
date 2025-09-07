# 🚀 Quick Start Guide - Agricultural AI Dashboard

**Ready to build tomorrow!** This guide will get you up and running with a world-class agricultural AI dashboard.

## ⚡ **30-Minute Setup**

### **1. Initialize Project (5 minutes)**
```bash
# Create new React project with Vite
npm create vite@latest agricultural-ai-dashboard -- --template react-ts
cd agricultural-ai-dashboard

# Install core dependencies
npm install @tanstack/react-query zustand framer-motion
npm install d3 @types/d3 recharts
npm install @tensorflow/tfjs
npm install tailwindcss @tailwindcss/forms
npm install lucide-react clsx tailwind-merge

# Install dev dependencies
npm install -D @types/d3 @testing-library/react @testing-library/jest-dom
npm install -D playwright @playwright/test
```

### **2. Project Structure Setup (5 minutes)**
```bash
# Create folder structure
mkdir -p src/{components/{dashboard,charts,ai,ui},hooks,services,utils,types}
mkdir -p public/{models,assets/{icons,images}}
mkdir -p tests/{components,e2e,utils}

# Create essential files
touch src/components/dashboard/{WeatherDashboard,MarketIntelligence,PestDetection}.tsx
touch src/hooks/{useApi,useWebSocket,useNotifications}.ts
touch src/services/{api,websocket,ai}.ts
```

### **3. Environment Setup (5 minutes)**
```bash
# Create .env file
cat > .env << EOF
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
VITE_MAPBOX_TOKEN=your_mapbox_token
VITE_AUTH0_DOMAIN=your_auth0_domain
VITE_AUTH0_CLIENT_ID=your_auth0_client_id
EOF
```

### **4. Basic Configuration (10 minutes)**
```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})
```

### **5. Start Development (5 minutes)**
```bash
# Start backend (in separate terminal)
cd ../src/api
python run_backend.py

# Start frontend
npm run dev
```

---

## 🎨 **First Dashboard Component**

### **Weather Dashboard (Copy & Paste Ready)**
```typescript
// src/components/dashboard/WeatherDashboard.tsx
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import * as d3 from 'd3';
import { Card, CardHeader, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';

interface WeatherData {
  temperature: number;
  humidity: number;
  precipitation: number;
  windSpeed: number;
  timestamp: string;
}

interface WeatherDashboardProps {
  farmId: string;
}

export const WeatherDashboard: React.FC<WeatherDashboardProps> = ({ farmId }) => {
  const [weatherData, setWeatherData] = useState<WeatherData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchWeatherData();
    const interval = setInterval(fetchWeatherData, 30000); // Update every 30s
    return () => clearInterval(interval);
  }, [farmId]);

  const fetchWeatherData = async () => {
    try {
      const response = await fetch(`/api/agents/weather`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          farm_id: farmId,
          analysis_type: 'current'
        })
      });
      
      const data = await response.json();
      setWeatherData(data.data.current_weather);
    } catch (error) {
      console.error('Failed to fetch weather data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Card className="p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="h-32 bg-gray-200 rounded"></div>
        </div>
      </Card>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="weather-dashboard">
        <CardHeader className="flex flex-row items-center justify-between">
          <h3 className="text-lg font-semibold">Weather Intelligence</h3>
          <Badge variant="success" className="animate-pulse">Live</Badge>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <WeatherMetric
              label="Temperature"
              value={`${weatherData?.temperature}°C`}
              icon="🌡️"
              color="blue"
            />
            <WeatherMetric
              label="Humidity"
              value={`${weatherData?.humidity}%`}
              icon="💧"
              color="cyan"
            />
            <WeatherMetric
              label="Precipitation"
              value={`${weatherData?.precipitation}mm`}
              icon="🌧️"
              color="blue"
            />
            <WeatherMetric
              label="Wind Speed"
              value={`${weatherData?.windSpeed} km/h`}
              icon="💨"
              color="gray"
            />
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

const WeatherMetric: React.FC<{
  label: string;
  value: string;
  icon: string;
  color: string;
}> = ({ label, value, icon, color }) => (
  <motion.div
    whileHover={{ scale: 1.05 }}
    className={`p-4 rounded-lg bg-${color}-50 border border-${color}-200`}
  >
    <div className="flex items-center space-x-2">
      <span className="text-2xl">{icon}</span>
      <div>
        <p className="text-sm text-gray-600">{label}</p>
        <p className={`text-xl font-bold text-${color}-700`}>{value}</p>
      </div>
    </div>
  </motion.div>
);
```

---

## 🔧 **Essential Hooks**

### **API Hook**
```typescript
// src/hooks/useApi.ts
import { useState, useEffect } from 'react';

export const useApi = <T>(endpoint: string, options: RequestInit = {}) => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const token = localStorage.getItem('token');
        const response = await fetch(`/api${endpoint}`, {
          ...options,
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
            ...options.headers,
          },
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        setData(result.data);
      } catch (err) {
        setError(err as Error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [endpoint]);

  return { data, loading, error };
};
```

### **WebSocket Hook**
```typescript
// src/hooks/useWebSocket.ts
import { useEffect, useState } from 'react';

export const useWebSocket = (url: string) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [readyState, setReadyState] = useState<number>(WebSocket.CLOSED);

  useEffect(() => {
    const ws = new WebSocket(url);
    
    ws.onopen = () => {
      setSocket(ws);
      setReadyState(WebSocket.OPEN);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setLastMessage(data);
    };

    ws.onclose = () => {
      setSocket(null);
      setReadyState(WebSocket.CLOSED);
    };

    return () => {
      ws.close();
    };
  }, [url]);

  const sendMessage = (message: any) => {
    if (socket && readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    }
  };

  return { socket, lastMessage, readyState, sendMessage };
};
```

---

## 🎯 **Tomorrow's Development Plan**

### **Morning (9 AM - 12 PM)**
1. **Project Setup** (30 min)
   - Initialize Vite + React + TypeScript
   - Install all dependencies
   - Setup folder structure

2. **Design System** (1.5 hours)
   - Configure Tailwind CSS
   - Create base components (Button, Card, Badge)
   - Setup color palette and typography

3. **API Integration** (1 hour)
   - Setup API client
   - Create useApi hook
   - Test backend connection

### **Afternoon (1 PM - 6 PM)**
1. **Weather Dashboard** (2 hours)
   - Build weather component
   - Add real-time updates
   - Implement D3.js charts

2. **Market Intelligence** (2 hours)
   - Create market component
   - Add price prediction charts
   - Implement trend analysis

3. **Navigation & Layout** (1 hour)
   - Setup routing
   - Create main layout
   - Add responsive design

### **Evening (7 PM - 9 PM)**
1. **Pest Detection** (1 hour)
   - Build pest alert system
   - Add image upload
   - Create recommendation panel

2. **Polish & Testing** (1 hour)
   - Add animations
   - Fix bugs
   - Test responsiveness

---

## 🚀 **Ready to Code!**

**Tomorrow you'll have:**
- ✅ **Working dashboard** with real-time data
- ✅ **Beautiful UI** with smooth animations
- ✅ **AI integration** with TensorFlow.js
- ✅ **Responsive design** for all devices
- ✅ **Production-ready** code structure

**Let's build something amazing! 🌾✨**
