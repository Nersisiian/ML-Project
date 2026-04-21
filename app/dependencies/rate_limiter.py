import time
from typing import Dict, Tuple
import redis.asyncio as redis
from app.core.config import settings

class RateLimiter:
    """Redis-based rate limiter implementation"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.rate = settings.API_RATE_LIMIT
        self.window = settings.API_RATE_LIMIT_WINDOW
    
    async def is_allowed(self, key: str) -> Tuple[bool, Dict]:
        """Check if request is allowed"""
        now = time.time()
        window_start = now - self.window
        
        # Use Redis sorted set for rate limiting
        redis_key = f"rate_limit:{key}"
        
        # Remove old entries
        await self.redis.zremrangebyscore(redis_key, 0, window_start)
        
        # Count requests in current window
        count = await self.redis.zcard(redis_key)
        
        if count < self.rate:
            # Add current request
            await self.redis.zadd(redis_key, {str(now): now})
            await self.redis.expire(redis_key, self.window)
            
            return True, {
                "limit": self.rate,
                "remaining": self.rate - count - 1,
                "reset": int(window_start + self.window)
            }
        
        return False, {
            "limit": self.rate,
            "remaining": 0,
            "reset": int(window_start + self.window)
        }