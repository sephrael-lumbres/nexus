# Process definitions for Nexus deployment

# Web process - FastAPI server
web: python -m nexus.api --host 0.0.0.0 --port ${PORT:-8000}

# Worker process - Job processing workers
worker: python -m nexus.worker ${WORKER_COUNT:-3}

# Release process - Run migrations on deploy
release: alembic upgrade head