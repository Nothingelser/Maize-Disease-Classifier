"""
Application factory for Plant Disease Classifier.
"""
from flask import Flask, jsonify
from flask_cors import CORS
import logging
import os

logger = logging.getLogger(__name__)

def create_app(config_object=None):
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder='templates', static_folder='static')

    if config_object:
        app.config.from_object(config_object)
        config_name = config_object if isinstance(config_object, str) else config_object.__name__
        logger.info("Loaded configuration from %s", config_name)
    else:
        from config.settings import get_config
        resolved_config = get_config()
        app.config.from_object(resolved_config)
        logger.info("Loaded configuration from environment")

    if app.config['DEBUG']:
        logger.debug("Supabase URL configured: %s", bool(app.config.get('SUPABASE_URL')))
        logger.debug("Upload folder: %s", app.config['UPLOAD_FOLDER'])
        logger.debug("Model path: %s", app.config['MODEL_PATH'])

    CORS(app, resources={r"/api/*": {"origins": app.config.get('CORS_ORIGINS', '*')}})

    from app.api.routes import api_bp
    from app.main_routes import main_bp

    app.register_blueprint(api_bp)
    app.register_blueprint(main_bp)

    register_error_handlers(app)
    register_context_processors(app)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    from app.services.prediction_service import PredictionService

    try:
        prediction_service = PredictionService(model_path=app.config['MODEL_PATH'])
        model_info = prediction_service.get_model_info()
        logger.info(
            "Model loaded: %s with %s features",
            model_info['model_type'],
            model_info['features']
        )
    except Exception as exc:
        logger.warning("Model not loaded: %s", exc)

    logger.info("Application initialized successfully")
    return app

def register_error_handlers(app):
    """Register API-friendly error handlers."""

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Resource not found', 'status': 404}), 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error("Internal server error: %s", error)
        return jsonify({'error': 'Internal server error', 'status': 500}), 500

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': 'Bad request', 'status': 400}), 400

    @app.errorhandler(413)
    def too_large(error):
        return jsonify({
            'error': f'File too large. Maximum size: {app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)}MB',
            'status': 413
        }), 413

    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        return jsonify({
            'error': 'Rate limit exceeded. Please try again later.',
            'status': 429
        }), 429

def register_context_processors(app):
    """Register template helpers."""

    @app.context_processor
    def utility_processor():
        from datetime import datetime

        def current_year():
            return datetime.now().year

        def get_config_value(key, default=None):
            return app.config.get(key, default)

        return dict(
            current_year=current_year,
            get_config_value=get_config_value,
            app_name='Plant Disease Classifier',
            app_version=app.config.get('MODEL_VERSION', '2.0.0')
        )
