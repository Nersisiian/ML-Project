# pipelines/data_pipeline/ingestion/kafka_consumer.py
import asyncio
import json
from typing import Dict, Any
from aiokafka import AIOKafkaConsumer
import orjson
from structlog import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential

logger = get_logger()

class RealEstateKafkaConsumer:
    """High-throughput Kafka consumer with exactly-once semantics"""
    
    def __init__(
        self,
        bootstrap_servers: list[str],
        topic: str,
        group_id: str,
        max_batch_size: int = 1000,
        batch_timeout_ms: int = 5000
    ):
        self.consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            enable_auto_commit=False,  # Manual commit for exactly-once
            max_poll_records=max_batch_size,
            auto_offset_reset='earliest',
            value_deserializer=lambda x: orjson.loads(x),
            session_timeout_ms=30000,
            max_poll_interval_ms=600000
        )
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        
    async def start(self):
        await self.consumer.start()
        logger.info("Kafka consumer started", topic=self.consumer._topics)
        
    async def consume_batches(self):
        """Consume messages in batches for efficient processing"""
        batch = []
        last_commit = asyncio.get_event_loop().time()
        
        try:
            async for msg in self.consumer:
                batch.append({
                    'value': msg.value,
                    'partition': msg.partition,
                    'offset': msg.offset,
                    'timestamp': msg.timestamp
                })
                
                current_time = asyncio.get_event_loop().time()
                if (len(batch) >= self.max_batch_size or 
                    current_time - last_commit >= self.batch_timeout_ms / 1000):
                    
                    # Process batch with retry
                    await self._process_batch_with_retry(batch)
                    
                    # Commit offsets after successful processing
                    await self.consumer.commit()
                    logger.info(f"Committed {len(batch)} messages")
                    
                    batch = []
                    last_commit = current_time
                    
        except Exception as e:
            logger.error("Consumer error", error=str(e))
            raise
        finally:
            await self.consumer.stop()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def _process_batch_with_retry(self, batch: list):
        """Process batch with exponential backoff retry"""
        # Validate schema
        validated = await self._validate_schema(batch)
        
        # Write to raw storage
        await self._write_to_raw_storage(validated)
        
        # Send to feature pipeline
        await self._send_to_feature_pipeline(validated)
    
    async def _validate_schema(self, batch: list):
        from great_expectations.core import ExpectationSuite
        # Implementation with Great Expectations
        pass