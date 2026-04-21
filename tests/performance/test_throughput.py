import pytest
import time
import threading
import concurrent.futures
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestThroughputPerformance:
    """Throughput performance tests"""
    
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
    
    def test_sequential_throughput(self, valid_payload, headers):
        """Test sequential request throughput"""
        
        n_requests = 100
        start = time.time()
        
        for _ in range(n_requests):
            response = client.post("/api/v1/predict", json=valid_payload, headers=headers)
            assert response.status_code == 200
        
        total_time = time.time() - start
        throughput = n_requests / total_time
        
        print(f"\nSequential throughput:")
        print(f"  Requests: {n_requests}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")
        
        assert throughput > 10  # At least 10 req/s sequentially
    
    def test_concurrent_throughput(self, valid_payload, headers):
        """Test concurrent request throughput"""
        
        def make_request():
            response = client.post("/api/v1/predict", json=valid_payload, headers=headers)
            return response.status_code
        
        n_requests = 200
        n_concurrent = 20
        
        start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as executor:
            futures = [executor.submit(make_request) for _ in range(n_requests)]
            results = [f.result() for f in futures]
        
        total_time = time.time() - start
        throughput = n_requests / total_time
        
        success_count = sum(1 for r in results if r == 200)
        
        print(f"\nConcurrent throughput:")
        print(f"  Requests: {n_requests}")
        print(f"  Concurrency: {n_concurrent}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")
        print(f"  Success rate: {success_count/n_requests*100:.1f}%")
        
        assert throughput > 50  # At least 50 req/s concurrent
        assert success_count > n_requests * 0.95  # >95% success rate
    
    def test_sustained_load(self, valid_payload, headers):
        """Test sustained load over time"""
        
        duration = 30  # seconds
        start = time.time()
        request_count = 0
        
        while time.time() - start < duration:
            response = client.post("/api/v1/predict", json=valid_payload, headers=headers)
            assert response.status_code in [200, 429]  # Rate limited is acceptable
            request_count += 1
        
        sustained_throughput = request_count / duration
        
        print(f"\nSustained load ({duration}s):")
        print(f"  Total requests: {request_count}")
        print(f"  Sustained throughput: {sustained_throughput:.2f} req/s")
        
        # Should maintain reasonable throughput
        assert sustained_throughput > 30
    
    def test_peak_load(self, valid_payload, headers):
        """Test peak load handling"""
        
        def burst_requests(n_burst, delay=0):
            time.sleep(delay)
            for _ in range(n_burst):
                response = client.post("/api/v1/predict", json=valid_payload, headers=headers)
        
        # Simulate 5 bursts of 50 requests
        threads = []
        for i in range(5):
            t = threading.Thread(target=burst_requests, args=(50, i * 0.1))
            threads.append(t)
            t.start()
        
        start = time.time()
        for t in threads:
            t.join()
        total_time = time.time() - start
        
        peak_throughput = 250 / total_time  # 5 * 50 = 250 requests
        
        print(f"\nPeak load test:")
        print(f"  Total requests: 250")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Peak throughput: {peak_throughput:.2f} req/s")
        
        # Should handle peak load without crashing
        assert peak_throughput > 100