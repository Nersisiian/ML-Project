#!/usr/bin/env python3
"""
Stress testing script for API endpoints
"""

import asyncio
import aiohttp
import json
import time
import statistics
from typing import List, Dict
import argparse

class StressTester:
    """Stress testing for ML API"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.results = []
    
    def generate_payload(self) -> Dict:
        """Generate random payload"""
        import random
        return {
            "square_feet": random.randint(1000, 5000),
            "bedrooms": random.randint(1, 6),
            "bathrooms": random.uniform(1, 4),
            "year_built": random.randint(1950, 2024),
            "zipcode": random.choice(["94105", "94103", "94123", "94107"]),
            "pool": random.choice([True, False]),
            "fireplace": random.choice([True, False]),
            "garage_spaces": random.randint(0, 3),
            "condition_score": random.uniform(1, 10)
        }
    
    async def send_request(self, session: aiohttp.ClientSession, payload: Dict) -> Dict:
        """Send single request and measure latency"""
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.base_url}/api/v1/predict",
                json=payload,
                headers={"X-API-Key": self.api_key}
            ) as response:
                latency = (time.time() - start_time) * 1000  # ms
                return {
                    "success": response.status == 200,
                    "latency": latency,
                    "status": response.status
                }
        except Exception as e:
            return {
                "success": False,
                "latency": 0,
                "error": str(e)
            }
    
    async def run_stress_test(self, n_requests: int, concurrency: int):
        """Run stress test with specified concurrency"""
        print(f"\n🚀 Starting stress test: {n_requests} requests, {concurrency} concurrent")
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request(session, payload):
            async with semaphore:
                return await self.send_request(session, payload)
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for _ in range(n_requests):
                payload = self.generate_payload()
                tasks.append(bounded_request(session, payload))
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            self.results = results
            self.analyze_results(total_time, n_requests, concurrency)
    
    def analyze_results(self, total_time: float, n_requests: int, concurrency: int):
        """Analyze and display results"""
        successful = [r for r in self.results if r.get("success")]
        failed = [r for r in self.results if not r.get("success")]
        latencies = [r["latency"] for r in successful if "latency" in r]
        
        print("\n" + "="*60)
        print("📊 STRESS TEST RESULTS")
        print("="*60)
        print(f"Total requests: {n_requests}")
        print(f"Concurrency: {concurrency}")
        print(f"Success rate: {len(successful)/n_requests*100:.2f}%")
        print(f"Failed requests: {len(failed)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {n_requests/total_time:.2f} req/s")
        
        if latencies:
            print(f"\nLatency statistics (ms):")
            print(f"  Min: {min(latencies):.2f}")
            print(f"  Max: {max(latencies):.2f}")
            print(f"  Mean: {statistics.mean(latencies):.2f}")
            print(f"  Median: {statistics.median(latencies):.2f}")
            print(f"  p95: {statistics.quantiles(latencies, n=100)[94]:.2f}")
            print(f"  p99: {statistics.quantiles(latencies, n=100)[98]:.2f}")
        
        # Status code breakdown
        status_codes = {}
        for r in self.results:
            status = r.get("status", "error")
            status_codes[status] = status_codes.get(status, 0) + 1
        
        print(f"\nStatus code breakdown:")
        for code, count in status_codes.items():
            print(f"  {code}: {count} ({count/n_requests*100:.1f}%)")

async def main():
    parser = argparse.ArgumentParser(description="Stress test ML API")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--requests", type=int, default=1000, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=50, help="Concurrent requests")
    parser.add_argument("--api-key", default="dev-key-1", help="API key")
    
    args = parser.parse_args()
    
    tester = StressTester(args.url, args.api_key)
    await tester.run_stress_test(args.requests, args.concurrency)

if __name__ == "__main__":
    asyncio.run(main())