# 🚀 Backend API Endpoints Documentation

**FAANG-Level API Documentation** for Organic Agriculture Agentic AI System

## 📋 **Overview**

This document provides comprehensive API endpoint documentation for frontend integration. All endpoints are production-ready and follow RESTful conventions with proper authentication, validation, and error handling.

**Base URL:** `http://localhost:8000`  
**API Version:** v1  
**Authentication:** JWT Bearer Token  

---

## 🔐 **Authentication Endpoints**

### **POST** `/auth/login`
**Purpose:** User authentication and token generation  
**Frontend Usage:** Login form submission  
**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```
**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```
**Frontend Integration:**
```javascript
// Login function
const login = async (username, password) => {
  const response = await fetch('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  });
  const data = await response.json();
  localStorage.setItem('token', data.access_token);
  return data;
};
```

### **GET** `/auth/me`
**Purpose:** Get current user information  
**Frontend Usage:** User profile, dashboard header  
**Headers:** `Authorization: Bearer <token>`  
**Response:**
```json
{
  "user_id": "string",
  "username": "string",
  "email": "string",
  "role": "string",
  "permissions": ["string"]
}
```

---

## 🤖 **AI Agent Endpoints**

### **POST** `/agents/weather`
**Purpose:** Weather analysis and forecasting  
**Frontend Usage:** Weather dashboard, crop planning  
**Headers:** `Authorization: Bearer <token>`  
**Request Body:**
```json
{
  "farm_id": "string",
  "location": {
    "latitude": 40.7128,
    "longitude": -74.0060
  },
  "analysis_type": "forecast|current|historical",
  "timeframe": "7d|30d|90d"
}
```
**Response:**
```json
{
  "status": "success",
  "data": {
    "current_weather": {
      "temperature": 22.5,
      "humidity": 65,
      "precipitation": 0.2,
      "wind_speed": 12.3
    },
    "forecast": [
      {
        "date": "2024-01-01",
        "temperature": 24.1,
        "precipitation_probability": 0.3,
        "recommendations": ["Ideal for planting", "Monitor soil moisture"]
      }
    ],
    "risk_assessment": {
      "drought_risk": "low",
      "flood_risk": "medium",
      "temperature_stress": "none"
    }
  }
}
```

### **POST** `/agents/market`
**Purpose:** Market analysis and price predictions  
**Frontend Usage:** Market dashboard, pricing strategy  
**Request Body:**
```json
{
  "crop_type": "wheat|corn|soybean|tomato",
  "region": "string",
  "analysis_type": "price_prediction|market_trends|supply_demand",
  "timeframe": "1m|3m|6m|1y"
}
```
**Response:**
```json
{
  "status": "success",
  "data": {
    "current_price": 245.50,
    "price_prediction": {
      "1_month": 248.75,
      "3_months": 252.30,
      "6_months": 258.90
    },
    "market_trends": {
      "trend": "increasing",
      "volatility": "medium",
      "confidence": 0.85
    },
    "recommendations": [
      "Consider selling in 2-3 months",
      "Monitor supply chain disruptions"
    ]
  }
}
```

### **POST** `/agents/pest`
**Purpose:** Pest and disease detection  
**Frontend Usage:** Pest monitoring dashboard, alert system  
**Request Body:**
```json
{
  "farm_id": "string",
  "crop_type": "string",
  "images": ["base64_encoded_image"],
  "sensor_data": {
    "temperature": 25.5,
    "humidity": 70,
    "soil_moisture": 0.6
  }
}
```
**Response:**
```json
{
  "status": "success",
  "data": {
    "pest_detection": {
      "detected": true,
      "pest_type": "aphids",
      "severity": "medium",
      "confidence": 0.87
    },
    "disease_detection": {
      "detected": false,
      "disease_type": null,
      "confidence": 0.95
    },
    "recommendations": [
      "Apply neem oil treatment",
      "Increase monitoring frequency",
      "Consider beneficial insects"
    ],
    "risk_level": "medium"
  }
}
```

### **POST** `/agents/prediction`
**Purpose:** Crop yield and supply chain predictions  
**Frontend Usage:** Yield forecasting, planning dashboard  
**Request Body:**
```json
{
  "farm_id": "string",
  "crop_type": "string",
  "planting_date": "2024-03-15",
  "expected_harvest": "2024-08-15",
  "historical_data": {
    "previous_yields": [1200, 1350, 1280],
    "weather_conditions": ["normal", "drought", "normal"]
  }
}
```
**Response:**
```json
{
  "status": "success",
  "data": {
    "yield_prediction": {
      "predicted_yield": 1420.5,
      "confidence_interval": [1380, 1460],
      "factors": {
        "weather_impact": 0.15,
        "soil_quality": 0.25,
        "pest_pressure": -0.05
      }
    },
    "supply_chain_risk": {
      "overall_risk": "low",
      "logistics_risk": "medium",
      "market_risk": "low"
    },
    "recommendations": [
      "Optimize irrigation schedule",
      "Prepare for 15% yield increase",
      "Secure transportation early"
    ]
  }
}
```

### **POST** `/agents/decision`
**Purpose:** Strategic decision making and recommendations  
**Frontend Usage:** Decision support system, strategic planning  
**Request Body:**
```json
{
  "farm_id": "string",
  "decision_type": "planting|harvesting|irrigation|pest_control",
  "context": {
    "current_conditions": "string",
    "goals": ["maximize_yield", "minimize_costs"],
    "constraints": ["budget", "time", "resources"]
  }
}
```
**Response:**
```json
{
  "status": "success",
  "data": {
    "decision": {
      "recommended_action": "Plant corn in 2 weeks",
      "confidence": 0.92,
      "reasoning": "Optimal soil temperature and moisture levels",
      "expected_outcome": "15% yield increase"
    },
    "alternatives": [
      {
        "action": "Wait 1 week",
        "pros": ["Lower risk"],
        "cons": ["Reduced yield potential"]
      }
    ],
    "implementation_plan": [
      "Prepare soil",
      "Order seeds",
      "Schedule planting equipment"
    ]
  }
}
```

### **POST** `/agents/analyze`
**Purpose:** Run all agents for comprehensive analysis  
**Frontend Usage:** Complete farm analysis, dashboard overview  
**Request Body:**
```json
{
  "farm_id": "string",
  "analysis_scope": "comprehensive|quick|detailed"
}
```
**Response:**
```json
{
  "status": "success",
  "data": {
    "weather_analysis": { /* weather data */ },
    "market_analysis": { /* market data */ },
    "pest_analysis": { /* pest data */ },
    "prediction_analysis": { /* prediction data */ },
    "decision_analysis": { /* decision data */ },
    "overall_score": 8.5,
    "recommendations": ["string"],
    "alerts": ["string"]
  }
}
```

---

## 📊 **Data Access Endpoints**

### **GET** `/data/farms`
**Purpose:** Get farm information  
**Frontend Usage:** Farm selection, farm management  
**Headers:** `Authorization: Bearer <token>`  
**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 10)
- `search`: Search term
- `status`: Filter by status

**Response:**
```json
{
  "status": "success",
  "data": {
    "farms": [
      {
        "farm_id": "string",
        "name": "string",
        "location": {
          "latitude": 40.7128,
          "longitude": -74.0060,
          "address": "string"
        },
        "crops": ["wheat", "corn"],
        "status": "active",
        "last_updated": "2024-01-01T00:00:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 10,
      "total": 100,
      "pages": 10
    }
  }
}
```

### **GET** `/data/farms/{farm_id}`
**Purpose:** Get specific farm details  
**Frontend Usage:** Farm detail page, farm management  

### **GET** `/data/crops`
**Purpose:** Get crop information  
**Frontend Usage:** Crop selection, crop management  

---

## 📈 **Metrics Endpoints**

### **GET** `/metrics`
**Purpose:** Get system metrics  
**Frontend Usage:** System monitoring dashboard  
**Response:**
```json
{
  "status": "success",
  "data": {
    "data_quality": {
      "completeness": 99.85,
      "consistency": 98.5,
      "accuracy": 95.0,
      "status": "EXCELLENT"
    },
    "performance": {
      "records_per_second": 86000,
      "memory_usage": 4.1,
      "cpu_usage": 45.0,
      "status": "GOOD"
    },
    "business": {
      "enterprise_readiness": 98.84,
      "total_records": 8931933,
      "active_farms": 100,
      "status": "EXCELLENT"
    }
  }
}
```

### **GET** `/metrics/critical`
**Purpose:** Get critical metrics and alerts  
**Frontend Usage:** Alert system, critical monitoring  

---

## 🏥 **Health Check Endpoints**

### **GET** `/health`
**Purpose:** System health check  
**Frontend Usage:** System status indicator  
**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0",
  "uptime": 86400,
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "agents": "healthy"
  }
}
```

---

## 🔌 **WebSocket Endpoints**

### **WS** `/ws`
**Purpose:** Real-time updates and live data streaming  
**Frontend Usage:** Real-time dashboards, live monitoring  
**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Handle real-time updates
};
```

**Message Types:**
- `weather_update`: Real-time weather data
- `pest_alert`: Pest detection alerts
- `market_update`: Market price changes
- `system_alert`: System notifications

---

## 🛠️ **Frontend Integration Examples**

### **React Hook for API Calls**
```javascript
import { useState, useEffect } from 'react';

const useApi = (endpoint, options = {}) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`/api${endpoint}`, {
        ...options,
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
          ...options.headers
        }
      });
      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [endpoint]);

  return { data, loading, error, refetch: fetchData };
};
```

### **Dashboard Component Example**
```javascript
const Dashboard = () => {
  const { data: weatherData } = useApi('/agents/weather', {
    method: 'POST',
    body: JSON.stringify({
      farm_id: 'farm_123',
      analysis_type: 'current'
    })
  });

  const { data: metricsData } = useApi('/metrics');

  return (
    <div className="dashboard">
      <WeatherCard data={weatherData} />
      <MetricsCard data={metricsData} />
    </div>
  );
};
```

---

## 🔒 **Security & Authentication**

### **Token Management**
```javascript
// Add token to all requests
const apiClient = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Authorization': `Bearer ${localStorage.getItem('token')}`
  }
});

// Handle token expiration
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);
```

---

## 📝 **Error Handling**

### **Standard Error Response**
```json
{
  "error": "VALIDATION_ERROR",
  "message": "Invalid input data",
  "details": {
    "field": "farm_id",
    "value": null
  },
  "status_code": 400
}
```

### **Frontend Error Handling**
```javascript
const handleApiError = (error) => {
  if (error.response?.data?.error === 'VALIDATION_ERROR') {
    // Show validation errors
    showValidationErrors(error.response.data.details);
  } else if (error.response?.status === 401) {
    // Redirect to login
    redirectToLogin();
  } else {
    // Show generic error
    showErrorMessage('An error occurred. Please try again.');
  }
};
```

---

## 🚀 **Quick Start for Frontend**

1. **Install dependencies:**
   ```bash
   npm install axios react-query
   ```

2. **Setup API client:**
   ```javascript
   import axios from 'axios';
   
   const api = axios.create({
     baseURL: 'http://localhost:8000',
     headers: {
       'Content-Type': 'application/json'
     }
   });
   ```

3. **Start with authentication:**
   ```javascript
   const login = async (credentials) => {
     const response = await api.post('/auth/login', credentials);
     localStorage.setItem('token', response.data.access_token);
     return response.data;
   };
   ```

4. **Use agent endpoints:**
   ```javascript
   const getWeatherAnalysis = async (farmId) => {
     const response = await api.post('/agents/weather', {
       farm_id: farmId,
       analysis_type: 'current'
     });
     return response.data;
   };
   ```

---

**📋 Complete API Reference:** All endpoints are documented with OpenAPI/Swagger at `http://localhost:8000/docs` when the backend is running.
