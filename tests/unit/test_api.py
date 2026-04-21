import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestAPI:
    """Unit tests for API endpoints"""
    
    @pytest.fixture
    def valid_payload(self):
        return {
            "square_feet": 2200,
            "bedrooms": 4,
            "bathrooms": 2.5,
            "year_built": 2010,
            "zipcode": "94105",
            "pool": True,
            "fireplace": True,
            "garage_spaces": 2,
            "condition_score": 8
        }
    
    @pytest.fixture
    def headers(self):
        return {"X-API-Key": "dev-key-1"}
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert "status" in response.json()
    
    def test_predict_endpoint_success(self, valid_payload, headers):
        """Test successful prediction"""
        response = client.post(
            "/api/v1/predict",
            json=valid_payload,
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_price" in data
        assert "confidence_interval_lower" in data
        assert "confidence_interval_upper" in data
        assert data["predicted_price"] > 0
    
    def test_predict_endpoint_missing_api_key(self, valid_payload):
        """Test prediction without API key"""
        response = client.post("/api/v1/predict", json=valid_payload)
        assert response.status_code == 401
    
    def test_predict_endpoint_invalid_api_key(self, valid_payload):
        """Test prediction with invalid API key"""
        response = client.post(
            "/api/v1/predict",
            json=valid_payload,
            headers={"X-API-Key": "invalid-key"}
        )
        assert response.status_code == 401
    
    def test_predict_endpoint_invalid_input(self, headers):
        """Test prediction with invalid input"""
        invalid_payload = {
            "square_feet": -100,  # Negative square feet
            "bedrooms": 4,
            "bathrooms": 2.5,
            "year_built": 2010,
            "zipcode": "94105"
        }
        
        response = client.post(
            "/api/v1/predict",
            json=invalid_payload,
            headers=headers
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_endpoint(self, valid_payload, headers):
        """Test batch prediction"""
        batch_payload = {
            "requests": [valid_payload] * 5
        }
        
        response = client.post(
            "/api/v1/predict/batch",
            json=batch_payload,
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5
        assert all("predicted_price" in item for item in data)
    
    def test_rate_limiting(self, valid_payload, headers):
        """Test rate limiting"""
        # Make many requests quickly
        for _ in range(150):  # Exceed rate limit
            response = client.post(
                "/api/v1/predict",
                json=valid_payload,
                headers=headers
            )
            if response.status_code == 429:
                break
        
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
    
    def test_readiness_probe(self):
        """Test readiness endpoint"""
        response = client.get("/api/v1/ready")
        # Should be 200 if model loaded, 503 if not
        assert response.status_code in [200, 503]