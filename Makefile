# PulmoGuard - common development operations.
# Run `make help` to list targets.

.PHONY: help ml-test ml-install backend-install backend-dev backend-test \
        frontend-install frontend-dev frontend-build frontend-test \
        docker-up docker-down lint

help:
	@echo "PulmoGuard Makefile targets:"
	@echo "  ml-install         Install the ml/ package and its dependencies"
	@echo "  ml-test            Run ml/ unit tests"
	@echo "  backend-install    Install backend dependencies (incl. ml/ as a library)"
	@echo "  backend-dev        Run the backend API locally with hot reload"
	@echo "  backend-test       Run backend unit tests"
	@echo "  frontend-install   Install frontend dependencies"
	@echo "  frontend-dev       Run the frontend dev server"
	@echo "  frontend-build     Build the frontend production bundle"
	@echo "  docker-up          Build and start the full system via docker compose"
	@echo "  docker-down        Stop the docker compose stack"
	@echo "  lint               Run linters across ml/ and backend/"

ml-install:
	pip install -r ml/requirements.txt
	pip install ./ml

ml-test:
	cd ml && pytest tests/ -v

backend-install:
	pip install ./ml
	pip install -r backend/requirements.txt

backend-dev:
	cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

backend-test:
	cd backend && pytest tests/ -v

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

docker-up:
	docker compose up --build

docker-down:
	docker compose down

lint:
	ruff check ml/pulmoguard/ backend/app/
