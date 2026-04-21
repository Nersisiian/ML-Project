.PHONY: help install test lint format clean docker-up docker-down deploy

# Colors
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m

help:
@echo "$(GREEN)ML Project Commands:$(NC)"
@echo "  make install    - Install dependencies"
@echo "  make test       - Run all tests"
@echo "  make lint       - Run linters"
@echo "  make docker-up  - Start all services"
@echo "  make deploy     - Deploy to production"

install:
pip install -r requirements.txt
@echo "$(GREEN)✅ Dependencies installed$(NC)"

test:
pytest tests/ -v --tb=short
@echo "$(GREEN)✅ All tests passed$(NC)"

lint:
ruff check . --fix
black . --check
@echo "$(GREEN)✅ Linting passed$(NC)"

docker-up:
docker-compose -f docker/docker-compose.yml up -d
@echo "$(GREEN)✅ Services started$(NC)"

docker-down:
docker-compose -f docker/docker-compose.yml down

deploy:
@echo "$(GREEN)🚀 Deploying...$(NC)"
git push origin main
@echo "$(GREEN)✅ Deployed!$(NC)"

all: install test lint
@echo "$(GREEN)🎉 All checks passed!$(NC)"
