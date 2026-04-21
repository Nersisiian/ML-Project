from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import time
from typing import Dict
import redis.asyncio as redis

class RateLimiter:
    """Rate limiting middleware"""
    
    def __init__(self, redis_client: redis.Redis, rate: int = 100, window: int = 60):
        self.redis = redis_client
        self.rate = rate
        self.window = window
    
    async def __call__(self, request: Request):
        client_ip = request.client.host
        key = f"ratelimit:{client_ip}"
        
        now = time.time()
        window_start = now - self.window
        
        # Use Lua script for atomic operation
        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        
        redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
        local count = redis.call('ZCARD', key)
        
        if count < limit then
            redis.call('ZADD', key, now, now)
            redis.call('EXPIRE', key, window)
            return {1, limit - count - 1, now + window}
        else
            return {0, 0, now + window}
        end
        """
        
        result = await self.redis.eval(lua_script, 1, key, now, self.window, self.rate)
        
        if result[0] == 0:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(self.rate),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(result[2]))
                }
            )
        
        # Add rate limit headers
        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(self.rate),
            "X-RateLimit-Remaining": str(result[1]),
            "X-RateLimit-Reset": str(int(result[2]))
        }