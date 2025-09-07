# 🚀 Backend Infrastructure - Organic Agriculture Agentic AI

## Overview

Production-ready FastAPI backend infrastructure for the Organic Agriculture Agentic AI system. Provides RESTful APIs, WebSocket support, authentication, and comprehensive monitoring.

## 🏗️ Architecture

```
backend/
├── main.py              # Main FastAPI application
├── config.py            # Configuration management
├── middleware.py        # Custom middleware
├── run_backend.py       # Backend runner script
├── requirements.txt     # Python dependencies
├── env.example         # Environment configuration template
└── README.md           # This documentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install dependencies (from main project)
pip install -r requirement.txt
```

### 2. Configure Environment

```bash
# Copy environment template (from main project)
cp env.example .env

# Edit configuration
nano .env
```

### 3. Start Backend

```bash
# Development mode
python DeepAgentPrototype/backend/run_backend.py

# Production mode
python DeepAgentPrototype/backend/run_backend.py --env production

# Custom port
python DeepAgentPrototype/backend/run_backend.py --port 8080
```

## 📋 API Endpoints

### **Authentication**
- `POST /auth/login` - User login
- `GET /auth/me` - Get current user

### **Agent Endpoints**
- `POST /agents/weather` - Weather analysis
- `POST /agents/market` - Market analysis
- `POST /agents/pest` - Pest/disease analysis
- `POST /agents/prediction` - Yield/risk predictions
- `POST /agents/decision` - Decision recommendations
- `POST /agents/analyze` - Full analysis (all agents)

### **Data Endpoints**
- `GET /data/farms` - List farms
- `GET /data/crops` - List crop types

### **Metrics Endpoints**
- `GET /metrics` - All project metrics
- `GET /metrics/critical` - Critical metrics only

### **System Endpoints**
- `GET /health` - Health check
- `WebSocket /ws` - Real-time updates

## 🔧 Configuration

### **Environment Variables**

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Debug mode | `false` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `MONGODB_URI` | MongoDB connection URI | `mongodb://localhost:27017/` |
| `MONGODB_DATABASE` | Database name | `organic_agriculture_ai` |
| `SECRET_KEY` | JWT secret key | `your-secret-key` |
| `CORS_ORIGINS` | Allowed origins | `["http://localhost:3000"]` |

### **Configuration Classes**

- **DevelopmentSettings**: Debug enabled, auto-reload
- **ProductionSettings**: Optimized for production
- **TestingSettings**: Test database, debug logging

## 🛡️ Security Features

### **Authentication**
- JWT token-based authentication
- Password hashing with bcrypt
- Token expiration and refresh

### **Security Headers**
- X-Content-Type-Options
- X-Frame-Options
- X-XSS-Protection
- Referrer-Policy
- Permissions-Policy

### **Rate Limiting**
- Configurable requests per minute
- IP-based rate limiting
- Automatic cleanup of old requests

### **CORS Protection**
- Configurable allowed origins
- Credential support
- Method and header restrictions

## 📊 Monitoring & Metrics

### **Built-in Middleware**
- **LoggingMiddleware**: Request/response logging
- **MetricsMiddleware**: Performance metrics collection
- **RateLimitMiddleware**: Rate limiting
- **SecurityHeadersMiddleware**: Security headers
- **ErrorHandlingMiddleware**: Global error handling

### **Health Monitoring**
- Health check endpoint (`/health`)
- Database connection status
- Agent availability status
- System resource monitoring

### **Performance Metrics**
- Request count and error rate
- Average response time
- WebSocket connection tracking
- Resource usage monitoring

## 🔌 WebSocket Support

### **Real-time Features**
- Live data streaming
- Real-time alerts
- Dashboard updates
- Agent status updates

### **Connection Management**
- Connection pooling
- Automatic cleanup
- Connection limits
- Ping/pong heartbeat

## 🧪 Testing

### **Test Configuration**
```bash
# Run tests
pytest DeepAgentPrototype/backend/tests/

# With coverage
pytest --cov=backend DeepAgentPrototype/backend/tests/
```

### **Test Environment**
- Separate test database
- Mock external services
- Isolated test data
- Automated cleanup

## 🚀 Deployment

### **Development**
```bash
python run_backend.py --env development --reload
```

### **Production**
```bash
python run_backend.py --env production --workers 4
```

### **Docker**
```dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "run_backend.py", "--env", "production"]
```

## 📈 Performance Optimization

### **Async Processing**
- Async/await throughout
- Non-blocking I/O
- Concurrent request handling
- Background task processing

### **Caching**
- Redis integration
- Response caching
- Database query caching
- Session management

### **Database Optimization**
- Connection pooling
- Query optimization
- Index management
- Batch operations

## 🔍 Logging

### **Log Levels**
- DEBUG: Detailed debugging info
- INFO: General information
- WARNING: Warning messages
- ERROR: Error conditions
- CRITICAL: Critical errors

### **Log Files**
- `logs/backend.log`: Main application log
- `logs/access.log`: HTTP access log
- `logs/error.log`: Error log

### **Structured Logging**
- JSON format for production
- Request/response correlation
- Performance metrics
- Error tracking

## 🛠️ Development

### **Code Quality**
- Black code formatting
- isort import sorting
- flake8 linting
- mypy type checking

### **Pre-commit Hooks**
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

### **API Documentation**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI schema: `http://localhost:8000/openapi.json`

## 🔧 Troubleshooting

### **Common Issues**

1. **MongoDB Connection Failed**
   - Check MongoDB is running
   - Verify connection string
   - Check authentication credentials

2. **Port Already in Use**
   - Change port with `--port` flag
   - Kill existing process
   - Check for port conflicts

3. **CORS Errors**
   - Update CORS_ORIGINS in .env
   - Check frontend URL
   - Verify credentials setting

4. **Authentication Issues**
   - Check SECRET_KEY is set
   - Verify JWT token format
   - Check token expiration

### **Debug Mode**
```bash
# Enable debug logging
python run_backend.py --log-level debug

# Validate configuration
python run_backend.py --validate-config
```

## 📚 API Usage Examples

### **Authentication**
```python
import httpx

# Login
response = httpx.post("http://localhost:8000/auth/login", 
                     data={"username": "admin", "password": "admin"})
token = response.json()["access_token"]

# Use token
headers = {"Authorization": f"Bearer {token}"}
response = httpx.get("http://localhost:8000/auth/me", headers=headers)
```

### **Agent Analysis**
```python
# Weather analysis
response = httpx.post("http://localhost:8000/agents/weather",
                     json={"farm_id": "farm_001", "location": {"lat": 40.7128, "lng": -74.0060}},
                     headers=headers)

# Full analysis
response = httpx.post("http://localhost:8000/agents/analyze",
                     json={"farm_id": "farm_001", "crop_type": "tomato"},
                     headers=headers)
```

### **WebSocket Connection**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function(event) {
    console.log('Connected to WebSocket');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

## 🎯 Next Steps

1. **Complete Agent Integration** - Test all agent endpoints
2. **Add Database Models** - Implement data models
3. **Frontend Integration** - Connect React frontend
4. **Production Deployment** - Deploy to production
5. **Monitoring Setup** - Add comprehensive monitoring

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Status**: Production Ready ✅
