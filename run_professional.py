#!/usr/bin/env python
"""
Professional entry point for the Multi-Crop Disease Classifier.
Loads configuration and starts the Flask application.
"""
import logging
import logging.config
import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the current environment
env = os.environ.get('FLASK_ENV', 'development')
print(f"Starting application in {env} mode")

# Import configuration
from config.settings import DevelopmentConfig, ProductionConfig, TestingConfig

# Get the appropriate configuration based on environment
if env == 'production':
    CurrentConfig = ProductionConfig
elif env == 'testing':
    CurrentConfig = TestingConfig
else:
    CurrentConfig = DevelopmentConfig

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
if os.path.exists('config/logging.conf'):
    logging.config.fileConfig('config/logging.conf')
else:
    logging.basicConfig(
        level=CurrentConfig.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler(sys.stdout),
        ],
    )

logger = logging.getLogger(__name__)
try:
    from app import create_app
    app = create_app(CurrentConfig)

    if __name__ == '__main__':
        port = int(os.environ.get('PORT', 5000))
        logger.info("Multi-Crop Disease Classifier started in %s mode on port %s", env, port)
        app.run(
            host='0.0.0.0',
            port=port,
            debug=CurrentConfig.DEBUG,
        )
except Exception as exc:
    logger.error("Failed to start application: %s", exc)
    sys.exit(1)
