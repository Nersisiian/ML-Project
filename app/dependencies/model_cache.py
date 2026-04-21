import asyncio
import json
import hashlib
from typing import Dict, Any, Optional
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)

class ModelCache:
    """Model prediction cache with TTL and invalidation"""
    
    def __init__(self, redis_client: redis.Redis, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl
        self.local_cache: Dict[str, Any] = {}
        self.local_ttl: Dict[str, float] = {}
    
    def _generate_key(self, features: Dict[str, Any]) -> str:
        """Generate cache key from features"""
        feature_str = json.dumps(features, sort_keys=True)
        return f"pred:{hashlib.md5(feature_str.encode()).hexdigest()}"
    
    async def get(self, features: Dict[str, Any]) -> Optional[Any]:
        """Get cached prediction"""
        
        key = self._generate_key(features)
        
        # Check local cache first (L1)
        if key in self.local_cache:
            if asyncio.get_event_loop().time() < self.local_ttl.get(key, 0):
                logger.debug(f"L1 cache hit for {key}")
                return self.local_cache[key]
            else:
                # Remove expired
                del self.local_cache[key]
                del self.local_ttl[key]
        
        # Check Redis cache (L2)
        try:
            cached = await self.redis.get(key)
            if cached:
                logger.debug(f"L2 cache hit for {key}")
                result = json.loads(cached)
                
                # Update L1 cache
                self.local_cache[key] = result
                self.local_ttl[key] = asyncio.get_event_loop().time() + 60  # 1 min L1 TTL
                
                return result
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
        
        return None
    
    async def set(self, features: Dict[str, Any], value: Any):
        """Cache prediction result"""
        
        key = self._generate_key(features)
        
        # Store in Redis
        try:
            await self.redis.setex(
                key, 
                self.ttl, 
                json.dumps(value)
            )
            logger.debug(f"Cached result for {key}")
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
        
        # Also store in L1 cache
        self.local_cache[key] = value
        self.local_ttl[key] = asyncio.get_event_loop().time() + 60
    
    async def invalidate(self, features: Optional[Dict[str, Any]] = None):
        """Invalidate cache entries"""
        
        if features:
            # Invalidate specific key
            key = self._generate_key(features)
            await self.redis.delete(key)
            self.local_cache.pop(key, None)
            self.local_ttl.pop(key, None)
            logger.info(f"Invalidated cache for {key}")
        else:
            # Invalidate all
            await self.redis.flushdb()
            self.local_cache.clear()
            self.local_ttl.clear()
            logger.info("Invalidated entire cache")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        try:
            redis_keys = await self.redis.keys("pred:*")
            return {
                "redis_keys_count": len(redis_keys),
                "local_keys_count": len(self.local_cache),
                "ttl_seconds": self.ttl,
                "local_ttl_seconds": 60
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    async def warmup(self, sample_features: list):
        """Warm up cache with common requests"""
        
        logger.info(f"Warming up cache with {len(sample_features)} samples")
        
        for features in sample_features:
            await self.get(features)  # This will cache if not present
        
        logger.info("Cache warmup complete")

class ModelVersionCache:
    """Cache for model versions"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.current_version: Optional[str] = None
    
    async def get_current_version(self) -> Optional[str]:
        """Get current model version"""
        
        if self.current_version:
            return self.current_version
        
        try:
            version = await self.redis.get("model:current_version")
            if version:
                self.current_version = version.decode()
                return self.current_version
        except Exception as e:
            logger.warning(f"Failed to get model version: {e}")
        
        return None
    
    async def set_current_version(self, version: str):
        """Update current model version"""
        
        self.current_version = version
        try:
            await self.redis.set("model:current_version", version)
            logger.info(f"Model version updated to {version}")
        except Exception as e:
            logger.warning(f"Failed to set model version: {e}")