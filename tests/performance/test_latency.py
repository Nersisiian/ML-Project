import pytest
import time
import statistics
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestLatencyPerformance:
    """Latency performance tests"""
    
    @pytest.fixture
    def valid_payload(self):
        return {
            "square_feet": 2200,
            "bedrooms": 4,
            "bathrooms": 2.5,
            "year_built": 2010,
            "zipcode": "94105",
            "pool": True,
            "fireplace": True
        }
    
    @pytest.fixture
    def headers(self):
        return {"X-API-Key": "test-key-123"}
    
    def test_single_prediction_latency(self, valid_payload, headers):
        """Test single prediction latency"""
        
        latencies = []
        
        for _ in range(100):
            start = time.time()
            response = client.post("/api/v1/predict", json=valid_payload, headers=headers)
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)
            
            assert response.status_code == 200
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"\nSingle prediction latency (ms):")
        print(f"  Average: {avg_latency:.2f}")
        print(f"  p95: {p95_latency:.2f}")
        print(f"  p99: {p99_latency:.2f}")
        
        # Assert thresholds
        assert avg_latency < 100  # Average < 100ms
        assert p95_latency < 200  # p95 < 200ms
    
    def test_cached_prediction_latency(self, valid_payload, headers):
        """Test cached prediction latency"""
        
        # First request (cache miss)
        start = time.time()
        response1 = client.post("/api/v1/predict", json=valid_payload, headers=headers)
        miss_latency = (time.time() - start) * 1000
        
        # Second request (cache hit)
        start = time.time()
        response2 = client.post("/api/v1/predict", json=valid_payload, headers=headers)
        hit_latency = (time.time() - start) * 1000
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        print(f"\nCached vs uncached latency:")
        print(f"  Cache miss: {miss_latency:.2f}ms")
        print(f"  Cache hit: {hit_latency:.2f}ms")
        
        # Cache hit should be faster
        assert hit_latency < miss_latency
    
    def test_batch_prediction_latency(self, valid_payload, headers):
        """Test batch prediction latency"""
        
        batch_sizes = [1, 10, 25, 50, 100]
        
        for batch_size in batch_sizes:
            batch_payload = {"requests": [valid_payload] * batch_size}
            
            start = time.time()
            response = client.post("/api/v1/predict/batch", json=batch_payload, headers=headers)
            total_latency = (time.time() - start) * 1000
            
            assert response.status_code == 200
            
            avg_latency_per_request = total_latency / batch_size
            
            print(f"\nBatch size {batch_size}:")
            print(f"  Total: {total_latency:.2f}ms")
            print(f"  Per request: {avg_latency_per_request:.2f}ms")
            
            # Per-request latency should decrease with batching
            if batch_size > 1:
                assert avg_latency_per_request < 50  # <50ms per request