#!/bin/bash

# Deployment script for production

set -e

echo "🚀 Starting deployment process..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-staging}
REGION="us-west-2"
PROJECT_NAME="real-estate-ml"

echo "📦 Deploying to environment: $ENVIRONMENT"

# Load environment variables
if [ -f ".env.${ENVIRONMENT}" ]; then
    source .env.${ENVIRONMENT}
    echo "✅ Loaded environment variables from .env.${ENVIRONMENT}"
else
    echo -e "${RED}❌ .env.${ENVIRONMENT} not found${NC}"
    exit 1
fi

# Build Docker images
echo "🏗️ Building Docker images..."
docker build -f docker/Dockerfile.api -t ${PROJECT_NAME}-api:latest .
docker build -f docker/Dockerfile.trainer -t ${PROJECT_NAME}-trainer:latest .

# Tag images for ECR
if [ "$ENVIRONMENT" == "production" ]; then
    echo "🔐 Logging to AWS ECR..."
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com
    
    # Tag and push
    docker tag ${PROJECT_NAME}-api:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${PROJECT_NAME}-api:${GITHUB_SHA}
    docker tag ${PROJECT_NAME}-api:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${PROJECT_NAME}-api:latest
    
    docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${PROJECT_NAME}-api:${GITHUB_SHA}
    docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${PROJECT_NAME}-api:latest
    
    echo "✅ Images pushed to ECR"
fi

# Deploy to Kubernetes
if [ "$ENVIRONMENT" == "production" ]; then
    echo "☸️ Updating Kubernetes deployment..."
    
    # Update API deployment
    kubectl set image deployment/api api=${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${PROJECT_NAME}-api:${GITHUB_SHA} -n production
    
    # Wait for rollout
    kubectl rollout status deployment/api -n production --timeout=5m
    
    # Run smoke tests
    echo "🧪 Running smoke tests..."
    kubectl port-forward service/api 8000:8000 -n production &
    sleep 10
    
    if curl -f http://localhost:8000/api/v1/health; then
        echo -e "${GREEN}✅ Smoke tests passed${NC}"
    else
        echo -e "${RED}❌ Smoke tests failed${NC}"
        exit 1
    fi
    
    # Cleanup port-forward
    pkill -f "kubectl port-forward"
fi

# Run database migrations
echo "🗄️ Running database migrations..."
python scripts/migrate_db.py --environment $ENVIRONMENT

# Update model to production
if [ "$ENVIRONMENT" == "production" ]; then
    echo "🤖 Promoting model to production..."
    python -c "
import mlflow
client = mlflow.tracking.MlflowClient()
model_version = client.get_latest_versions('real_estate_predictor', stages=['Staging'])[0].version
client.transition_model_version_stage('real_estate_predictor', model_version, 'Production')
print(f'Promoted version {model_version} to Production')
    "
fi

echo -e "${GREEN}✅ Deployment completed successfully!${NC}"

# Print access information
if [ "$ENVIRONMENT" == "staging" ]; then
    echo ""
    echo "📊 Access Information:"
    echo "  API: http://staging.real-estate-ml.com"
    echo "  MLflow: http://mlflow.staging.real-estate-ml.com"
    echo "  Grafana: http://monitoring.staging.real-estate-ml.com"
elif [ "$ENVIRONMENT" == "production" ]; then
    echo ""
    echo "📊 Access Information:"
    echo "  API: https://api.real-estate-ml.com"
    echo "  MLflow: https://mlflow.real-estate-ml.com"
    echo "  Grafana: https://monitoring.real-estate-ml.com"
fi