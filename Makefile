.PHONY: help install dev test lint format train serve docker-build docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Run development server"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linters"
	@echo "  format      - Format code"
	@echo "  train       - Run training pipeline"
	@echo "  serve       - Run API server"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-up   - Start Docker services"
	@echo "  docker-down - Stop Docker services"
	@echo "  clean       - Clean cache files"

install:
	poetry install

dev:
	poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	poetry run pytest tests/ -v --cov=app --cov=ml

lint:
	poetry run ruff check .
	poetry run mypy app/ ml/

format:
	poetry run black app/ ml/ tests/
	poetry run ruff check --fix .

train:
	poetry run python -m ml.training.train

serve:
	poetry run gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +