install:
	pip install fastapi "uvicorn[standard]" polars pyarrow sumy nltk statsmodels scipy numpy
	python -m nltk.downloader punkt punkt_tab
	cd frontend && npm install

dev:
	(cd backend && uvicorn main:app --reload --port 8000) & \
	(cd frontend && npm run dev)

backend:
	cd backend && uvicorn main:app --reload --port 8000

frontend:
	cd frontend && npm run dev
