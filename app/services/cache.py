import json
from typing import Any, Optional
import redis.asyncio as redis

class CacheService:
    """Redis cache service"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        value = await self.redis.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        try:
            await self.redis.setex(key, ttl, json.dumps(value))
            return True
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        return await self.redis.delete(key) > 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        return await self.redis.exists(key) > 0
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        return await self.redis.incrby(key, amount)