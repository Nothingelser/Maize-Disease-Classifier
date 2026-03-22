#!/bin/bash

echo "🚀 Starting deployment of Maize Disease Classifier"

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose is required but not installed. Aborting." >&2; exit 1; }

# Create necessary directories
mkdir -p models logs data/uploads

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Pull latest code
echo "📦 Pulling latest code..."
git pull origin main

# Build and start containers
echo "🐳 Building Docker containers..."
docker-compose build

echo "🔄 Starting services..."
docker-compose up -d

# Run database migrations
echo "🗄️ Running database migrations..."
docker-compose exec web flask db upgrade

# Health check
echo "🏥 Performing health check..."
sleep 10
curl -f http://localhost:5000/api/health || { echo "Health check failed!"; exit 1; }

echo "✅ Deployment completed successfully!"
echo "📊 Application is running at http://localhost:5000"