from fastapi import FastAPI

app = FastAPI(title="ML Project API", version="1.0.0")

@app.get("/")
def root():
    return {"message": "ML Project API is running!", "status": "online"}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "ml-project-api"}

@app.get("/api/v1/health")
def health_v1():
    return {"status": "healthy", "model_loaded": False, "message": "ML Project API - Demo Mode"}

@app.get("/demo")
def demo():
    return {
        "name": "ML Project API",
        "version": "1.0.0",
        "status": "running on Render",
        "endpoints": [
            "/ - Root",
            "/health - Health check",
            "/api/v1/health - API health",
            "/demo - Demo info"
        ]
    }
