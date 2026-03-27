"""
Main routes for server-rendered application pages.
"""
from flask import Blueprint, render_template, jsonify, abort

main_bp = Blueprint('main', __name__)

CROP_APPS = [
    {'key': 'maize', 'name': 'Maize', 'subtitle': 'Blight, rust, gray leaf spot analysis'},
    {'key': 'cassava', 'name': 'Cassava', 'subtitle': 'Mosaic, streak, and blight diagnostics'},
    {'key': 'rice', 'name': 'Rice', 'subtitle': 'Leaf blast and bacterial disease checks'},
    {'key': 'tomato', 'name': 'Tomato', 'subtitle': 'Multi-disease tomato leaf workflow'},
    {'key': 'potato', 'name': 'Potato', 'subtitle': 'Early and late blight monitoring'},
    {'key': 'pepper_bell', 'name': 'Pepper Bell', 'subtitle': 'Healthy vs bacterial spot detection'},
]


def _render_detection_workspace(active_crop=None):
    """Render the main detection workspace with optional crop scoping."""
    crop_title = None
    if active_crop:
        matching = next((crop for crop in CROP_APPS if crop['key'] == active_crop), None)
        if not matching:
            abort(404)
        crop_title = matching['name']

    return render_template(
        'professional_index.html',
        active_crop=active_crop,
        crop_title=crop_title,
        crop_apps=CROP_APPS,
    )

@main_bp.route('/')
def app_hub():
    """Crop app hub page."""
    return render_template('app_hub.html', crop_apps=CROP_APPS)


@main_bp.route('/workspace')
def home():
    """All-crop workspace page."""
    return _render_detection_workspace()


@main_bp.route('/apps/<crop>')
def crop_app(crop):
    """Crop-specific detection workspace route."""
    return _render_detection_workspace(active_crop=crop)

@main_bp.route('/health')
def health():
    """Simple app health page."""
    return jsonify({"status": "healthy", "message": "Plant Disease Classifier is running"}), 200

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
