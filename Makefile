PY := python
NPM := npm

.PHONY: help install models warmup dev backend frontend build clean

help:
	@echo "FinLens — make targets:"
	@echo "  install   install backend + frontend deps"
	@echo "  models    verify local model artefacts are present"
	@echo "  warmup    precompute caches (timeline grid, topic×ticker, seasonality)"
	@echo "  dev       run FastAPI (:8000) + Vite (:3000)"
	@echo "  backend   run FastAPI only"
	@echo "  frontend  run Vite only"
	@echo "  build     build frontend for production"

install:
	$(PY) -m pip install -e .
	cd frontend && $(NPM) install

models:
	$(PY) -m backend.scripts.check_models

warmup:
	$(PY) -m backend.scripts.warmup

backend:
	$(PY) -m uvicorn backend.main:app --reload --port 8000

frontend:
	cd frontend && $(NPM) run dev

dev:
	@echo "Launching backend on :8000 and frontend on :3000 …"
	@($(PY) -m uvicorn backend.main:app --port 8000 &) ; \
	 (cd frontend && $(NPM) run dev)

build:
	cd frontend && $(NPM) run build

clean:
	rm -rf cache/*.parquet cache/topics.db frontend/dist frontend/node_modules
