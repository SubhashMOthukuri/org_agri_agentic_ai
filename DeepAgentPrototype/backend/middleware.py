"""
Backend Middleware - Organic Agriculture Agentic AI

Custom middleware for logging, rate limiting, and monitoring.

Author: Principal AI Engineer
Version: 1.0.0
Date: December 2024
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict, deque
import asyncio

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Log request
        start_time = time.time()
        logger.info(f"Request: {request.method} {request.url}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting"""
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(deque)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client IP
        client_ip = request.client.host
        
        # Clean old requests
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Remove requests older than 1 minute
        while self.requests[client_ip] and self.requests[client_ip][0] < minute_ago:
            self.requests[client_ip].popleft()
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"}
            )
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting metrics"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.response_times = deque(maxlen=1000)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        try:
            response = await call_next(request)
            self.request_count += 1
            
            # Track response time
            process_time = time.time() - start_time
            self.response_times.append(process_time)
            
            # Track errors
            if response.status_code >= 400:
                self.error_count += 1
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Request error: {e}")
            raise
    
    def get_metrics(self) -> dict:
        """Get collected metrics"""
        if not self.response_times:
            avg_response_time = 0
        else:
            avg_response_time = sum(self.response_times) / len(self.response_times)
        
        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "avg_response_time": avg_response_time,
            "recent_response_times": list(self.response_times)[-10:]  # Last 10 response times
        }


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware for health check optimization"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip middleware processing for health checks
        if request.url.path == "/health":
            return await call_next(request)
        
        # Add health check headers
        response = await call_next(request)
        response.headers["X-Health-Check"] = "enabled"
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for global error handling"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "error_type": type(e).__name__
                }
            )


class WebSocketMiddleware:
    """Middleware for WebSocket connections"""
    
    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self.active_connections = 0
        self.connection_times = {}
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            # Check connection limit
            if self.active_connections >= self.max_connections:
                await send({
                    "type": "websocket.close",
                    "code": 1013,  # Try again later
                    "reason": "Server overloaded"
                })
                return
            
            # Track connection
            self.active_connections += 1
            connection_id = id(scope)
            self.connection_times[connection_id] = time.time()
            
            logger.info(f"WebSocket connected. Active: {self.active_connections}")
            
            try:
                await self.app(scope, receive, send)
            finally:
                # Clean up connection
                self.active_connections -= 1
                if connection_id in self.connection_times:
                    del self.connection_times[connection_id]
                logger.info(f"WebSocket disconnected. Active: {self.active_connections}")
        else:
            await self.app(scope, receive, send)
    
    def get_connection_stats(self) -> dict:
        """Get WebSocket connection statistics"""
        return {
            "active_connections": self.active_connections,
            "max_connections": self.max_connections,
            "connection_times": self.connection_times
        }


# Performance monitoring decorator
def monitor_performance(func_name: str = None):
    """Decorator for monitoring function performance"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func_name or func.__name__} executed in {execution_time:.3f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func_name or func.__name__} failed after {execution_time:.3f}s: {e}")
                raise
        return wrapper
    return decorator


# Async context manager for resource management
class ResourceManager:
    """Context manager for resource cleanup"""
    
    def __init__(self, resources: list):
        self.resources = resources
    
    async def __aenter__(self):
        for resource in self.resources:
            if hasattr(resource, '__aenter__'):
                await resource.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for resource in reversed(self.resources):
            if hasattr(resource, '__aexit__'):
                await resource.__aexit__(exc_type, exc_val, exc_tb)


# Circuit breaker pattern for external services
class CircuitBreaker:
    """Circuit breaker for external service calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e
    
    def get_state(self) -> dict:
        """Get circuit breaker state"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time
        }
