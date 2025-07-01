#!/bin/bash
set -e

# Apply database migrations
python manage.py migrate --noinput

# Collect static files
python manage.py collectstatic --noinput

# Start Uvicorn server (ASGI)
exec uvicorn config.asgi:application --host 0.0.0.0 --port 8000 --workers 3
