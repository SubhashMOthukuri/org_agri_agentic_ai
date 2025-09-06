# API Specifications

## 🌐 API Overview
This document defines the complete API specification for the Organic Agriculture Agentic AI system.

**Base URL**: `https://api.agri-risk-ai.com/v1`  
**Authentication**: Bearer Token  
**Content-Type**: `application/json`

---

## 🔐 Authentication

### **Headers Required**
```http
Authorization: Bearer <your-token>
Content-Type: application/json
```

### **Token Management**
- Tokens expire after 24 hours
- Refresh tokens available
- Rate limiting: 1000 requests/hour per token

---

## 📊 Data Models

### **Weather Data**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "location": {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "farm_id": "farm_001"
  },
  "temperature": 30.5,
  "rainfall": 50.2,
  "humidity": 70.0,
  "wind_speed": 15.3,
  "storm_alert": true,
  "forecast_hours": 24
}
```

### **Market Data**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "crop": "Tomato",
  "region": "California",
  "price_current": 1.50,
  "price_forecast": 1.65,
  "demand_trend": "upward",
  "supply_level": "moderate",
  "supply_disruption_alert": false,
  "market_volatility": 0.15
}
```

### **Pest & Disease Data**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "farm_id": "farm_001",
  "crop": "Tomato",
  "pest_risk": 0.7,
  "disease_alert": true,
  "pest_types": ["aphids", "whiteflies"],
  "disease_types": ["blight"],
  "severity": "high",
  "confidence": 0.85
}
```

### **Risk Prediction**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "farm_id": "farm_001",
  "crop": "Tomato",
  "risk_score": 0.82,
  "risk_type": "High Pest & Weather Risk",
  "confidence": 0.9,
  "factors": {
    "weather_impact": 0.7,
    "pest_impact": 0.8,
    "market_impact": 0.3
  },
  "prediction_horizon": "7_days"
}
```

### **Action Recommendation**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "farm_id": "farm_001",
  "crop": "Tomato",
  "risk_score": 0.82,
  "actions": [
    {
      "action_type": "harvest",
      "description": "Harvest 50% of crops immediately",
      "priority": "high",
      "deadline": "2024-01-16T18:00:00Z",
      "estimated_impact": "reduce_loss_by_30%"
    },
    {
      "action_type": "pest_control",
      "description": "Apply organic pest control measures",
      "priority": "medium",
      "deadline": "2024-01-17T12:00:00Z",
      "estimated_impact": "prevent_spread_by_40%"
    }
  ],
  "total_estimated_savings": "$15,000"
}
```

---

## 🔌 API Endpoints

### **Weather Data Agent**

#### **GET /weather/current**
Get current weather data for a specific location.

**Parameters:**
- `farm_id` (string, required): Farm identifier
- `hours` (integer, optional): Forecast hours (default: 24)

**Response:**
```json
{
  "status": "success",
  "data": {
    "current_weather": { /* Weather Data object */ },
    "forecast": [
      { /* Weather Data object */ }
    ]
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### **GET /weather/historical**
Get historical weather data for analysis.

**Parameters:**
- `farm_id` (string, required): Farm identifier
- `start_date` (string, required): Start date (ISO 8601)
- `end_date` (string, required): End date (ISO 8601)

**Response:**
```json
{
  "status": "success",
  "data": {
    "historical_data": [
      { /* Weather Data object */ }
    ],
    "summary": {
      "avg_temperature": 25.5,
      "total_rainfall": 150.2,
      "storm_days": 5
    }
  }
}
```

### **Market Data Agent**

#### **GET /market/prices**
Get current and forecasted crop prices.

**Parameters:**
- `crop` (string, required): Crop type
- `region` (string, optional): Geographic region
- `days` (integer, optional): Forecast days (default: 30)

**Response:**
```json
{
  "status": "success",
  "data": {
    "current_prices": { /* Market Data object */ },
    "price_forecast": [
      { /* Market Data object */ }
    ],
    "trend_analysis": {
      "trend": "upward",
      "volatility": 0.15,
      "confidence": 0.8
    }
  }
}
```

#### **GET /market/demand**
Get demand analysis for specific crops.

**Parameters:**
- `crop` (string, required): Crop type
- `region` (string, optional): Geographic region

**Response:**
```json
{
  "status": "success",
  "data": {
    "demand_analysis": {
      "current_demand": "high",
      "demand_trend": "increasing",
      "seasonal_factor": 1.2,
      "market_size": 1000000
    }
  }
}
```

### **Pest & Disease Agent**

#### **GET /pest/risk-assessment**
Get pest and disease risk assessment.

**Parameters:**
- `farm_id` (string, required): Farm identifier
- `crop` (string, required): Crop type

**Response:**
```json
{
  "status": "success",
  "data": {
    "risk_assessment": { /* Pest & Disease Data object */ },
    "recommendations": [
      {
        "type": "prevention",
        "description": "Apply organic pest control",
        "urgency": "medium"
      }
    ]
  }
}
```

#### **POST /pest/report**
Report pest or disease observation.

**Request Body:**
```json
{
  "farm_id": "farm_001",
  "crop": "Tomato",
  "observation_type": "pest",
  "description": "Aphids observed on leaves",
  "severity": "medium",
  "location": {
    "latitude": 40.7128,
    "longitude": -74.0060
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "report_id": "report_123",
    "risk_score": 0.6,
    "recommendations": [
      {
        "action": "Apply neem oil treatment",
        "priority": "high"
      }
    ]
  }
}
```

### **Risk Prediction Agent**

#### **POST /predict/risk-score**
Get comprehensive risk prediction.

**Request Body:**
```json
{
  "farm_id": "farm_001",
  "crop": "Tomato",
  "prediction_horizon": "7_days",
  "include_factors": true
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "risk_prediction": { /* Risk Prediction object */ },
    "model_info": {
      "model_version": "v1.2.0",
      "training_date": "2024-01-01T00:00:00Z",
      "accuracy": 0.87
    }
  }
}
```

#### **GET /predict/batch**
Get risk predictions for multiple farms.

**Parameters:**
- `farm_ids` (array, required): List of farm IDs
- `crop` (string, required): Crop type
- `horizon` (string, optional): Prediction horizon (default: "7_days")

**Response:**
```json
{
  "status": "success",
  "data": {
    "predictions": [
      {
        "farm_id": "farm_001",
        "risk_prediction": { /* Risk Prediction object */ }
      }
    ],
    "summary": {
      "total_farms": 10,
      "high_risk_farms": 3,
      "avg_risk_score": 0.45
    }
  }
}
```

### **Action Recommendation Agent**

#### **POST /recommend/actions**
Get actionable recommendations based on risk assessment.

**Request Body:**
```json
{
  "farm_id": "farm_001",
  "crop": "Tomato",
  "risk_score": 0.82,
  "risk_type": "High Pest & Weather Risk",
  "budget_constraints": 5000,
  "time_constraints": "urgent"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "recommendations": { /* Action Recommendation object */ },
    "alternative_options": [
      {
        "action_type": "alternative_supplier",
        "description": "Switch to backup supplier",
        "cost": 2000,
        "effectiveness": 0.7
      }
    ]
  }
}
```

#### **POST /recommend/validate**
Validate and get feedback on recommended actions.

**Request Body:**
```json
{
  "farm_id": "farm_001",
  "action_id": "action_123",
  "status": "implemented",
  "feedback": "Action was effective",
  "actual_outcome": "reduced_loss_by_25%"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "feedback_id": "feedback_456",
    "model_update": "scheduled",
    "next_recommendation": "Consider applying additional pest control"
  }
}
```

### **Notification Agent**

#### **GET /notifications**
Get notifications for a specific farm or user.

**Parameters:**
- `farm_id` (string, optional): Farm identifier
- `user_id` (string, optional): User identifier
- `status` (string, optional): Notification status (unread, read, archived)
- `limit` (integer, optional): Number of notifications (default: 50)

**Response:**
```json
{
  "status": "success",
  "data": {
    "notifications": [
      {
        "id": "notif_123",
        "type": "risk_alert",
        "title": "High Risk Detected",
        "message": "Tomato crops at high risk due to pest outbreak",
        "priority": "high",
        "timestamp": "2024-01-15T10:30:00Z",
        "status": "unread",
        "actions": [
          {
            "action": "view_details",
            "url": "/dashboard/risk-details/123"
          }
        ]
      }
    ],
    "unread_count": 5
  }
}
```

#### **POST /notifications/send**
Send notification to users.

**Request Body:**
```json
{
  "recipients": ["user_123", "farm_001"],
  "type": "risk_alert",
  "title": "Urgent Action Required",
  "message": "Immediate harvest recommended",
  "priority": "high",
  "channels": ["email", "sms", "dashboard"]
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "notification_id": "notif_456",
    "delivery_status": {
      "email": "sent",
      "sms": "sent",
      "dashboard": "delivered"
    }
  }
}
```

---

## 🔄 WebSocket Endpoints

### **Real-time Updates**
**Connection**: `wss://api.agri-risk-ai.com/v1/ws`

#### **Subscribe to Farm Updates**
```json
{
  "action": "subscribe",
  "farm_id": "farm_001",
  "channels": ["weather", "pest", "market", "risk"]
}
```

#### **Real-time Data Format**
```json
{
  "type": "weather_update",
  "farm_id": "farm_001",
  "data": { /* Weather Data object */ },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## 📊 Error Handling

### **Error Response Format**
```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid farm_id provided",
    "details": {
      "field": "farm_id",
      "value": "invalid_id",
      "constraint": "Must be a valid UUID"
    }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### **Error Codes**
- `VALIDATION_ERROR`: Invalid input parameters
- `AUTHENTICATION_ERROR`: Invalid or missing authentication
- `AUTHORIZATION_ERROR`: Insufficient permissions
- `NOT_FOUND`: Resource not found
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INTERNAL_ERROR`: Server error
- `SERVICE_UNAVAILABLE`: External service unavailable

---

## 📈 Rate Limiting

### **Limits**
- **General API**: 1000 requests/hour per token
- **ML Predictions**: 100 requests/hour per token
- **WebSocket**: 10 concurrent connections per user
- **File Uploads**: 10MB per request, 100MB per hour

### **Headers**
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
```

---

## 🔧 SDK Examples

### **Python SDK**
```python
from agri_risk_ai import AgriRiskClient

client = AgriRiskClient(api_key="your-api-key")

# Get weather data
weather = client.weather.get_current(farm_id="farm_001")

# Get risk prediction
risk = client.predict.get_risk_score(
    farm_id="farm_001",
    crop="Tomato"
)

# Get recommendations
actions = client.recommend.get_actions(
    farm_id="farm_001",
    risk_score=risk.risk_score
)
```

### **JavaScript SDK**
```javascript
import { AgriRiskClient } from 'agri-risk-ai-js';

const client = new AgriRiskClient('your-api-key');

// Get weather data
const weather = await client.weather.getCurrent('farm_001');

// Get risk prediction
const risk = await client.predict.getRiskScore({
  farm_id: 'farm_001',
  crop: 'Tomato'
});

// Get recommendations
const actions = await client.recommend.getActions({
  farm_id: 'farm_001',
  risk_score: risk.risk_score
});
```

---

**Last Updated**: [Auto-updated by system]  
**Version**: 1.0  
**Maintainer**: API Team
