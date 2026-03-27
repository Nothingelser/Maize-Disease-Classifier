"""
Configuration package for Multi-Crop Disease Classifier
Provides environment-specific configuration settings
"""

from config.settings import (
    Config,
    DevelopmentConfig,
    TestingConfig,
    ProductionConfig,
    config
)

__all__ = [
    'Config',
    'DevelopmentConfig',
    'TestingConfig',
    'ProductionConfig',
    'config'
]