"""Vercel serverless entrypoint for the Flask app."""
from dotenv import load_dotenv

from app import create_app
from config.settings import get_config

# Load environment variables configured in Vercel project settings.
load_dotenv()

app = create_app(get_config())
