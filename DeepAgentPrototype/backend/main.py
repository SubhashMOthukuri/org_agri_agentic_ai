"""
FastAPI Backend - Organic Agriculture Agentic AI

Main FastAPI application with all API endpoints for the agentic AI system.
Includes agent endpoints, WebSocket support, authentication, and monitoring.

Author: Principal AI Engineer
Version: 1.0.0
Date: December 2024
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn

# JWT and security
import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt

# Pydantic models
from pydantic import BaseModel, Field
from typing import Union

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Project imports
from data_layer.database.mongodb.mongodb_setup import MongoDBManager
from agents.weather_agent import WeatherAgent
from agents.market_agent import MarketAgent
from agents.pest_agent import PestAgent
from agents.prediction_agent import PredictionAgent
from agents.decision_agent import DecisionAgent
from metrics.project_metrics import ProjectMetrics, MetricCategory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Organic Agriculture Agentic AI",
    description="Production-grade AI system for organic agriculture optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Import configuration
from backend.config import get_settings
settings = get_settings()

# Security configuration
SECRET_KEY = settings.secret_key
ALGORITHM = settings.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.jwt_expiration_hours * 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.trusted_hosts
)

# Global variables
mongo_manager = None
agents = {}
metrics = None
websocket_connections = []

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool = True

class UserInDB(User):
    hashed_password: str

class AgentRequest(BaseModel):
    farm_id: Optional[str] = None
    crop_type: Optional[str] = None
    location: Optional[Dict[str, float]] = None
    parameters: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime
    agent_name: str

class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime

# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Dependency to get current user
async def get_current_user(token: str = Depends(verify_token)):
    """Get current authenticated user"""
    # In production, this would query the database
    # For now, return a mock user
    return User(username=token, email=f"{token}@example.com", full_name=token)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global mongo_manager, agents, metrics
    
    logger.info("Starting Organic Agriculture Agentic AI Backend...")
    
    try:
        # Initialize MongoDB
        mongo_manager = MongoDBManager()
        if not mongo_manager.connect():
            logger.error("Failed to connect to MongoDB")
            raise Exception("MongoDB connection failed")
        logger.info("MongoDB connected successfully")
        
        # Initialize agents
        agents = {
            "weather": WeatherAgent(mongo_manager),
            "market": MarketAgent(mongo_manager),
            "pest": PestAgent(mongo_manager),
            "prediction": PredictionAgent(mongo_manager),
            "decision": DecisionAgent(mongo_manager)
        }
        logger.info(f"Initialized {len(agents)} agents")
        
        # Initialize metrics
        metrics = ProjectMetrics()
        metrics.load_enterprise_metrics()
        logger.info("Metrics system initialized")
        
        logger.info("Backend startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global mongo_manager
    
    logger.info("Shutting down backend...")
    
    if mongo_manager:
        mongo_manager.close()
        logger.info("MongoDB connection closed")
    
    logger.info("Backend shutdown completed")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "agents": len(agents) if agents else 0,
        "database": "connected" if mongo_manager else "disconnected"
    }

# Authentication endpoints
@app.post("/auth/login", response_model=Token)
async def login(username: str, password: str):
    """Login endpoint (mock implementation)"""
    # In production, verify against database
    if username == "admin" and password == "admin":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

@app.get("/auth/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return current_user

# Agent endpoints
@app.post("/agents/weather", response_model=AgentResponse)
async def weather_analysis(
    request: AgentRequest,
    current_user: User = Depends(get_current_user)
):
    """Get weather analysis and forecasts"""
    try:
        if not agents.get("weather"):
            raise HTTPException(status_code=500, detail="Weather agent not available")
        
        # Process weather request
        result = await agents["weather"].analyze_weather(
            farm_id=request.farm_id,
            location=request.location,
            parameters=request.parameters
        )
        
        return AgentResponse(
            success=True,
            data=result,
            timestamp=datetime.utcnow(),
            agent_name="weather"
        )
        
    except Exception as e:
        logger.error(f"Weather agent error: {e}")
        return AgentResponse(
            success=False,
            error=str(e),
            timestamp=datetime.utcnow(),
            agent_name="weather"
        )

@app.post("/agents/market", response_model=AgentResponse)
async def market_analysis(
    request: AgentRequest,
    current_user: User = Depends(get_current_user)
):
    """Get market analysis and price predictions"""
    try:
        if not agents.get("market"):
            raise HTTPException(status_code=500, detail="Market agent not available")
        
        result = await agents["market"].analyze_market(
            crop_type=request.crop_type,
            farm_id=request.farm_id,
            parameters=request.parameters
        )
        
        return AgentResponse(
            success=True,
            data=result,
            timestamp=datetime.utcnow(),
            agent_name="market"
        )
        
    except Exception as e:
        logger.error(f"Market agent error: {e}")
        return AgentResponse(
            success=False,
            error=str(e),
            timestamp=datetime.utcnow(),
            agent_name="market"
        )

@app.post("/agents/pest", response_model=AgentResponse)
async def pest_analysis(
    request: AgentRequest,
    current_user: User = Depends(get_current_user)
):
    """Get pest and disease analysis"""
    try:
        if not agents.get("pest"):
            raise HTTPException(status_code=500, detail="Pest agent not available")
        
        result = await agents["pest"].analyze_pest_risk(
            farm_id=request.farm_id,
            crop_type=request.crop_type,
            parameters=request.parameters
        )
        
        return AgentResponse(
            success=True,
            data=result,
            timestamp=datetime.utcnow(),
            agent_name="pest"
        )
        
    except Exception as e:
        logger.error(f"Pest agent error: {e}")
        return AgentResponse(
            success=False,
            error=str(e),
            timestamp=datetime.utcnow(),
            agent_name="pest"
        )

@app.post("/agents/prediction", response_model=AgentResponse)
async def prediction_analysis(
    request: AgentRequest,
    current_user: User = Depends(get_current_user)
):
    """Get yield and risk predictions"""
    try:
        if not agents.get("prediction"):
            raise HTTPException(status_code=500, detail="Prediction agent not available")
        
        result = await agents["prediction"].predict_yield_and_risk(
            farm_id=request.farm_id,
            crop_type=request.crop_type,
            parameters=request.parameters
        )
        
        return AgentResponse(
            success=True,
            data=result,
            timestamp=datetime.utcnow(),
            agent_name="prediction"
        )
        
    except Exception as e:
        logger.error(f"Prediction agent error: {e}")
        return AgentResponse(
            success=False,
            error=str(e),
            timestamp=datetime.utcnow(),
            agent_name="prediction"
        )

@app.post("/agents/decision", response_model=AgentResponse)
async def decision_analysis(
    request: AgentRequest,
    current_user: User = Depends(get_current_user)
):
    """Get decision recommendations"""
    try:
        if not agents.get("decision"):
            raise HTTPException(status_code=500, detail="Decision agent not available")
        
        result = await agents["decision"].make_decision(
            farm_id=request.farm_id,
            crop_type=request.crop_type,
            parameters=request.parameters
        )
        
        return AgentResponse(
            success=True,
            data=result,
            timestamp=datetime.utcnow(),
            agent_name="decision"
        )
        
    except Exception as e:
        logger.error(f"Decision agent error: {e}")
        return AgentResponse(
            success=False,
            error=str(e),
            timestamp=datetime.utcnow(),
            agent_name="decision"
        )

# Combined analysis endpoint
@app.post("/agents/analyze", response_model=Dict[str, AgentResponse])
async def full_analysis(
    request: AgentRequest,
    current_user: User = Depends(get_current_user)
):
    """Run all agents for comprehensive analysis"""
    try:
        results = {}
        
        # Run all agents in parallel
        tasks = []
        agent_names = ["weather", "market", "pest", "prediction", "decision"]
        
        for agent_name in agent_names:
            if agents.get(agent_name):
                task = asyncio.create_task(
                    _run_agent(agent_name, request)
                )
                tasks.append((agent_name, task))
        
        # Wait for all agents to complete
        for agent_name, task in tasks:
            try:
                result = await task
                results[agent_name] = result
            except Exception as e:
                results[agent_name] = AgentResponse(
                    success=False,
                    error=str(e),
                    timestamp=datetime.utcnow(),
                    agent_name=agent_name
                )
        
        return results
        
    except Exception as e:
        logger.error(f"Full analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _run_agent(agent_name: str, request: AgentRequest) -> AgentResponse:
    """Helper function to run individual agents"""
    agent = agents[agent_name]
    
    if agent_name == "weather":
        result = await agent.analyze_weather(
            farm_id=request.farm_id,
            location=request.location,
            parameters=request.parameters
        )
    elif agent_name == "market":
        result = await agent.analyze_market(
            crop_type=request.crop_type,
            farm_id=request.farm_id,
            parameters=request.parameters
        )
    elif agent_name == "pest":
        result = await agent.analyze_pest_risk(
            farm_id=request.farm_id,
            crop_type=request.crop_type,
            parameters=request.parameters
        )
    elif agent_name == "prediction":
        result = await agent.predict_yield_and_risk(
            farm_id=request.farm_id,
            crop_type=request.crop_type,
            parameters=request.parameters
        )
    elif agent_name == "decision":
        result = await agent.make_decision(
            farm_id=request.farm_id,
            crop_type=request.crop_type,
            parameters=request.parameters
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
    
    return AgentResponse(
        success=True,
        data=result,
        timestamp=datetime.utcnow(),
        agent_name=agent_name
    )

# Metrics endpoints
@app.get("/metrics")
async def get_metrics(current_user: User = Depends(get_current_user)):
    """Get all project metrics"""
    try:
        if not metrics:
            raise HTTPException(status_code=500, detail="Metrics not available")
        
        summary = metrics.get_metrics_summary()
        return {
            "summary": summary,
            "metrics": {name: {
                "value": metric.value,
                "status": metric.status.value,
                "unit": metric.threshold.unit,
                "description": metric.description
            } for name, metric in metrics.metrics.items()}
        }
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/critical")
async def get_critical_metrics(current_user: User = Depends(get_current_user)):
    """Get critical metrics only"""
    try:
        if not metrics:
            raise HTTPException(status_code=500, detail="Metrics not available")
        
        critical_metrics = metrics.get_critical_metrics()
        return {
            "count": len(critical_metrics),
            "metrics": [{
                "name": metric.name,
                "value": metric.value,
                "unit": metric.threshold.unit,
                "description": metric.description
            } for metric in critical_metrics]
        }
        
    except Exception as e:
        logger.error(f"Critical metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back the message (in production, process the message)
            await manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Data endpoints
@app.get("/data/farms")
async def get_farms(current_user: User = Depends(get_current_user)):
    """Get list of farms"""
    try:
        if not mongo_manager:
            raise HTTPException(status_code=500, detail="Database not available")
        
        # Query farms from database
        farms = list(mongo_manager.db.farms.find({}, {"_id": 0}))
        return {"farms": farms}
        
    except Exception as e:
        logger.error(f"Get farms error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/crops")
async def get_crops(current_user: User = Depends(get_current_user)):
    """Get list of crop types"""
    try:
        if not mongo_manager:
            raise HTTPException(status_code=500, detail="Database not available")
        
        # Query crops from database
        crops = list(mongo_manager.db.crops.find({}, {"_id": 0}))
        return {"crops": crops}
        
    except Exception as e:
        logger.error(f"Get crops error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
