"""
Application configuration settings for different environments
Loads environment variables and provides configuration classes
"""
import os
import re
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import quote

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

BASE_DIR = Path(__file__).parent.parent
IS_VERCEL = os.environ.get('VERCEL') == '1'

def normalize_database_url(url):
    """
    Normalize database URLs from .env, including bracket-wrapped passwords
    such as postgresql://user:[p@ss]@host:5432/db.
    """
    if not url:
        return url

    bracketed_password = re.match(r'^(?P<prefix>[^:]+://[^:]+:)\[(?P<password>.*)\](?P<suffix>@.*)$', url)
    if bracketed_password:
        password = quote(bracketed_password.group('password'), safe='')
        return f"{bracketed_password.group('prefix')}{password}{bracketed_password.group('suffix')}"

    return url

def resolve_database_url(*env_keys, default=None):
    """Return the first configured database URL after normalization."""
    for key in env_keys:
        value = os.environ.get(key)
        if value:
            return normalize_database_url(value)
    return normalize_database_url(default)

def validate_database_url(url):
    """Catch common .env mistakes before SQLAlchemy fails with cryptic errors."""
    if not url:
        return

    if '@https://' in url or '@http://' in url:
        raise ValueError(
            "DATABASE_URL is malformed. Use a database host like "
            "'db.<project-ref>.supabase.co', not an HTTPS URL."
        )

class Config:
    """
    Base configuration class with common settings
    """
    # Application
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production-2024-longer-key')
    DEBUG = False
    TESTING = False
    
    # Database
    _DEFAULT_SQLITE_PATH = Path('/tmp/app.db') if IS_VERCEL else (BASE_DIR / 'app.db')
    SQLALCHEMY_DATABASE_URI = resolve_database_url(
        'DATABASE_URL',
        default='sqlite:///' + str(_DEFAULT_SQLITE_PATH)
    )
    validate_database_url(SQLALCHEMY_DATABASE_URI)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    
    # Redis (for rate limiting and caching)
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # File Uploads
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = str(Path('/tmp/uploads') if IS_VERCEL else (BASE_DIR / 'data' / 'uploads'))
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    
    # Create upload folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Model Configuration
    MODEL_PATH = os.environ.get(
        'MODEL_PATH',
        str(Path(__file__).parent.parent / 'models' / 'maize_disease_classifier.pkl')
    )
    MODEL_VERSION = '2.0.0'
    
    # Image Processing
    IMG_SIZE = (128, 128)  # Height, Width
    IMG_CHANNELS = 3
    
    # API Configuration
    API_RATE_LIMIT = int(os.environ.get('API_RATE_LIMIT', 100))  # requests per hour
    API_RATE_WINDOW = 3600  # seconds
    MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', 10))
    
    # JWT Authentication
    JWT_EXPIRATION_HOURS = int(os.environ.get('JWT_EXPIRATION_HOURS', 24))
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    _LOG_DIR = Path('/tmp/logs') if IS_VERCEL else (BASE_DIR / 'logs')
    LOG_FILE = str(_LOG_DIR / 'app.log')
    ERROR_LOG_FILE = str(_LOG_DIR / 'error.log')
    
    # Create logs directory
    os.makedirs(_LOG_DIR, exist_ok=True)
    
    # Email Configuration (for notifications)
    SMTP_HOST = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
    SMTP_USER = os.environ.get('SMTP_USER', '')
    SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
    SMTP_USE_TLS = True
    
    # Admin Email
    ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'admin@maizeclassifier.com')
    
    # Monitoring
    SENTRY_DSN = os.environ.get('SENTRY_DSN', '')
    NEW_RELIC_LICENSE_KEY = os.environ.get('NEW_RELIC_LICENSE_KEY', '')
    
    # CORS
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:5000,http://localhost:3000').split(',')
    
    # Security
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    REMEMBER_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Feature Flags
    ENABLE_CACHE = os.environ.get('ENABLE_CACHE', 'True').lower() == 'true'
    ENABLE_RATE_LIMITING = os.environ.get('ENABLE_RATE_LIMITING', 'True').lower() == 'true'
    ENABLE_ASYNC_PREDICTIONS = os.environ.get('ENABLE_ASYNC_PREDICTIONS', 'False').lower() == 'true'
    
    # Class Names (for the model)
    CLASS_NAMES = ['Blight', 'Gray Leaf Spot', 'Healthy', 'Maize Rust']
    CLASS_COLORS = ['#dc3545', '#ffc107', '#28a745', '#17a2b8']
    
    # Supabase Configuration
    SUPABASE_URL = os.environ.get('SUPABASE_URL')
    SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
    SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_KEY')
    
    # Database (using Supabase PostgreSQL when configured)
    SQLALCHEMY_DATABASE_URI = resolve_database_url(
        'DATABASE_URL',
        default='sqlite:///' + str(_DEFAULT_SQLITE_PATH)
    )
    validate_database_url(SQLALCHEMY_DATABASE_URI)

    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        pass

class DevelopmentConfig(Config):
    """
    Development configuration - for local development
    """
    DEBUG = True
    TESTING = False
    
    # Prefer an explicitly configured development DB, then Supabase/Postgres, then local SQLite.
    SQLALCHEMY_DATABASE_URI = resolve_database_url(
        'DEV_DATABASE_URL',
        'DATABASE_POOLER_URL',
        'SUPABASE_POOLER_URL',
        'DATABASE_URL',
        default='sqlite:///' + str(Path('/tmp/dev.db') if IS_VERCEL else (BASE_DIR / 'dev.db'))
    )
    validate_database_url(SQLALCHEMY_DATABASE_URI)
    SQLALCHEMY_ECHO = True  # Log SQL queries
    
    # Development logging
    LOG_LEVEL = 'DEBUG'
    
    # Disable rate limiting in development
    ENABLE_RATE_LIMITING = False
    
    # Disable cache in development
    ENABLE_CACHE = False
    
    # Security (relaxed for development)
    SESSION_COOKIE_SECURE = False
    REMEMBER_COOKIE_SECURE = False
    
    # CORS for development
    CORS_ORIGINS = ['http://localhost:5000', 'http://localhost:3000', 'http://127.0.0.1:5000']

class TestingConfig(Config):
    """
    Testing configuration - for running tests
    """
    TESTING = True
    DEBUG = True
    
    # Use separate test database
    SQLALCHEMY_DATABASE_URI = resolve_database_url(
        'TEST_DATABASE_URL',
        default='sqlite:///' + str(Path('/tmp/test.db') if IS_VERCEL else (BASE_DIR / 'test.db'))
    )
    validate_database_url(SQLALCHEMY_DATABASE_URI)
    
    # Disable rate limiting for tests
    ENABLE_RATE_LIMITING = False
    
    # Disable cache for tests
    ENABLE_CACHE = False

    # Use a longer test key to satisfy JWT SHA256 recommendations.
    SECRET_KEY = os.environ.get('TEST_SECRET_KEY', 'testing-secret-key-change-in-production-2024')
    
    # Shorter JWT expiration for tests
    JWT_EXPIRATION_HOURS = 1
    
    # Don't create logs during tests
    LOG_LEVEL = 'WARNING'
    
    # Use in-memory database for faster tests (optional)
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

class ProductionConfig(Config):
    """
    Production configuration - for deployment
    """
    DEBUG = False
    TESTING = False
    
    # Use PostgreSQL in production (must be set in environment)
    SQLALCHEMY_DATABASE_URI = resolve_database_url(
        'DATABASE_POOLER_URL',
        'SUPABASE_POOLER_URL',
        'DATABASE_URL'
    )
    validate_database_url(SQLALCHEMY_DATABASE_URI)
    
    if not SQLALCHEMY_DATABASE_URI:
        raise ValueError("DATABASE_URL must be set in production")
    
    # Security settings for production
    SESSION_COOKIE_SECURE = True  # Only send over HTTPS
    REMEMBER_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'
    
    # Enable all features in production
    ENABLE_RATE_LIMITING = True
    ENABLE_CACHE = True
    
    # Production logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Strong secret key must be set in environment
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY must be set in production")
    
    # SSL/HTTPS
    PREFERRED_URL_SCHEME = 'https'
    
    # Proxy settings (if behind nginx)
    USE_X_FORWARDED_HOST = True
    USE_X_FORWARDED_PORT = True
    
    @staticmethod
    def init_app(app):
        """Initialize production-specific settings"""
        Config.init_app(app)
        
        # Configure logging for production
        import logging
        from logging.handlers import RotatingFileHandler
        
        # Create file handler for production logs
        file_handler = RotatingFileHandler(
            Config.LOG_FILE,
            maxBytes=10485760,  # 10MB
            backupCount=10
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s '
            '[in %(pathname)s:%(lineno)d]'
        ))
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Application startup')

class VercelConfig(Config):
    """
    Vercel-specific configuration.
    Keeps production-safe defaults without forcing local-only env requirements.
    """
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

class DockerConfig(ProductionConfig):
    """
    Docker-specific configuration
    """
    # Use environment variables for Docker
    SQLALCHEMY_DATABASE_URI = resolve_database_url(
        'DATABASE_POOLER_URL',
        'SUPABASE_POOLER_URL',
        'DATABASE_URL',
        default='postgresql://user:password@db:5432/maize_db'
    )
    validate_database_url(SQLALCHEMY_DATABASE_URI)
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://redis:6379/0')

# Configuration dictionary for easy access
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'vercel': VercelConfig,
    'docker': DockerConfig,
    'default': DevelopmentConfig
}

def get_config():
    """
    Get configuration based on FLASK_ENV environment variable
    """
    env = os.environ.get('FLASK_ENV')
    if env:
        return config.get(env, config['default'])
    if IS_VERCEL:
        return config['vercel']
    return config['default']
