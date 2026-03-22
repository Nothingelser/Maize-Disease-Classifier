"""
API middleware for authentication, rate limiting, and logging
"""
from functools import wraps
from flask import request, jsonify, g
import time
import redis
import logging
from app.database.supabase_client import supabase_client

logger = logging.getLogger(__name__)

# Redis connection for rate limiting
try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    redis_available = True
except Exception:
    redis_client = None
    redis_available = False
    logger.warning("Redis not available, rate limiting disabled")

class RateLimiter:
    """Rate limiting middleware"""
    
    def __init__(self, limit=100, window=3600):
        self.limit = limit
        self.window = window
    
    def __call__(self, f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not redis_available:
                return f(*args, **kwargs)

            try:
                # Get client identifier
                client_id = request.remote_addr
                if hasattr(g, 'user') and g.user:
                    client_id = f"user_{g.user.id}"

                key = f"rate_limit:{client_id}:{request.endpoint}"

                # Get current count
                current = redis_client.get(key)

                if current and int(current) >= self.limit:
                    return jsonify({
                        'error': 'Rate limit exceeded',
                        'limit': self.limit,
                        'window': self.window,
                        'retry_after': redis_client.ttl(key)
                    }), 429

                # Increment counter
                pipe = redis_client.pipeline()
                pipe.incr(key)
                pipe.expire(key, self.window)
                pipe.execute()
            except Exception as exc:
                logger.warning("Rate limiting unavailable, continuing without Redis: %s", exc)

            return f(*args, **kwargs)
        
        return decorated

class RequestLogger:
    """Request logging middleware"""
    
    def __call__(self, f):
        @wraps(f)
        def decorated(*args, **kwargs):
            start_time = time.time()
            
            # Log request
            logger.info(f"Request: {request.method} {request.path}")
            
            try:
                response = f(*args, **kwargs)
                
                # Log response
                elapsed_time = (time.time() - start_time) * 1000
                status_code = response[1] if isinstance(response, tuple) else 200
                logger.info(f"Response: {status_code} - {elapsed_time:.2f}ms")
                
                return response
                
            except Exception as e:
                elapsed_time = (time.time() - start_time) * 1000
                logger.error(f"Exception: {str(e)} - {elapsed_time:.2f}ms")
                raise
        
        return decorated

class AuthenticationMiddleware:
    """Authentication middleware backed by Supabase auth."""
    
    def __init__(self, exempt_routes=None):
        self.exempt_routes = exempt_routes or [
            '/api/login',
            '/api/register',
            '/api/health',
            '/api/model/info'
        ]
    
    def __call__(self, f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # Check if route is exempt
            if request.path in self.exempt_routes:
                return f(*args, **kwargs)
            
            # Get token from header
            auth_header = request.headers.get('Authorization')
            
            if not auth_header:
                return jsonify({'error': 'Missing authorization header'}), 401
            
            try:
                # Extract token
                token = auth_header.split(' ')[1] if ' ' in auth_header else auth_header

                if not supabase_client.is_connected():
                    return jsonify({'error': 'Supabase is not configured'}), 503

                auth_result = supabase_client.get_auth_user(token)
                if not auth_result.get('success'):
                    return jsonify({'error': 'Invalid token'}), 401

                user_result = supabase_client.get_user_record(auth_result['user'].id)
                if not user_result.get('success'):
                    return jsonify({'error': 'Invalid user'}), 401
                user = user_result['user']

                if not user:
                    return jsonify({'error': 'Invalid user'}), 401

                # Set user in global context
                g.user = user
            except Exception as e:
                logger.error(f"Authentication error: {str(e)}")
                return jsonify({'error': 'Authentication failed'}), 401

            return f(*args, **kwargs)
        
        return decorated
