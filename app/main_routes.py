"""
Main routes for server-rendered application pages.
"""
from flask import Blueprint, render_template, jsonify

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    """Landing page."""
    return render_template('professional_index.html')

@main_bp.route('/health')
def health():
    """Simple app health page."""
    return jsonify({"status": "healthy", "message": "Maize Disease Classifier is running"}), 200

@main_bp.route('/dashboard')
def dashboard():
    """Admin dashboard page."""
    return render_template('dashboard.html')

@main_bp.route('/analytics')
def analytics():
    """Analytics page."""
    return render_template('analytics.html')

@main_bp.route('/reports')
def reports():
    """Reports page."""
    return render_template('reports.html')

@main_bp.route('/login')
def login():
    """Login page."""
    return render_template('login.html')

@main_bp.route('/register')
def register():
    """Registration page."""
    return render_template('register.html')

@main_bp.route('/profile')
def profile():
    """User profile page."""
    return render_template('profile.html')

@main_bp.route('/forgot-password')
def forgot_password():
    """Password reset request page."""
    return render_template('forgot_password.html')

@main_bp.route('/auth/callback')
def auth_callback():
    """Supabase auth callback landing page for confirm/recovery links."""
    return render_template('auth_callback.html')

@main_bp.route('/reset-password')
def reset_password():
    """Password reset completion page after Supabase recovery link callback."""
    return render_template('reset_password.html')

@main_bp.route('/about')
def about():
    """About page."""
    return render_template('about.html')

@main_bp.route('/privacy')
def privacy():
    """Privacy page."""
    return render_template('privacy.html')

@main_bp.route('/support')
def support():
    """Support and feedback page."""
    return render_template('support.html')
