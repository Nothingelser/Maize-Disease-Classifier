"""
API routes for Maize Disease Classifier.
"""
from flask import Blueprint, request, jsonify, current_app, g, render_template, send_file
from app.services.prediction_service import PredictionService
from app.services.analytics_service import AnalyticsService
from app.services.export_service import ExportService
from app.database.supabase_client import supabase_client
from app.api.middleware import RateLimiter, RequestLogger, AuthenticationMiddleware
import datetime
import logging
import smtplib
from email.message import EmailMessage

logger = logging.getLogger(__name__)

def send_system_email(subject: str, body: str, recipient: str) -> bool:
    """Send notification emails when SMTP is configured."""
    smtp_host = current_app.config.get('SMTP_HOST')
    smtp_port = current_app.config.get('SMTP_PORT')
    smtp_user = current_app.config.get('SMTP_USER')
    smtp_password = current_app.config.get('SMTP_PASSWORD')
    use_tls = current_app.config.get('SMTP_USE_TLS', True)

    if not smtp_host or not smtp_port or not smtp_user or not smtp_password or not recipient:
        return False

    message = EmailMessage()
    message['Subject'] = subject
    message['From'] = smtp_user
    message['To'] = recipient
    message.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            if use_tls:
                server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(message)
        return True
    except Exception as exc:
        logger.warning('Email notification failed: %s', exc)
        return False

def utc_now():
    """Return the current timezone-aware UTC datetime."""
    return datetime.datetime.now(datetime.timezone.utc)

def use_supabase_auth():
    """Determine whether Supabase is available for auth and data storage."""
    return supabase_client.is_connected()

def require_supabase():
    """Return an API error response when Supabase is not configured."""
    if use_supabase_auth():
        return None
    return jsonify({'error': 'Supabase is not configured for this application'}), 503

api_bp = Blueprint('api', __name__, url_prefix='/api')

prediction_service = PredictionService()
analytics_service = AnalyticsService()
export_service = ExportService()

rate_limiter = RateLimiter(limit=100, window=3600)
request_logger = RequestLogger()
auth_middleware = AuthenticationMiddleware(
    exempt_routes=[
        '/api/login',
        '/api/register',
        '/api/health',
        '/api/model/info',
        '/api/predict/public',
        '/'
    ]
)

@api_bp.route('/')
def home():
    """Serve the main application page through the API namespace if needed."""
    return render_template('professional_index.html')

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'version': '2.0.0'
    }), 200

@api_bp.route('/model/info', methods=['GET'])
def get_model_info():
    """Return the deployed model metadata."""
    return jsonify(prediction_service.get_model_info()), 200

@api_bp.route('/login', methods=['POST'])
def login():
    """User login endpoint."""
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    supabase_error = require_supabase()
    if supabase_error:
        return supabase_error

    profile_result = supabase_client.find_user_profile(username)
    if not profile_result.get('success'):
        return jsonify({'error': 'Invalid credentials'}), 401

    profile = profile_result['profile']
    auth_result = supabase_client.login_user(profile['email'], password)
    if not auth_result.get('success'):
        error_text = auth_result.get('error', 'Invalid credentials')
        if 'confirm' in error_text.lower() or 'not confirmed' in error_text.lower():
            return jsonify({'error': 'Please confirm your email before signing in.'}), 403
        return jsonify({'error': 'Invalid credentials'}), 401

    user_result = supabase_client.get_user_record(auth_result['user'].id)
    if not user_result.get('success'):
        return jsonify({'error': 'Unable to load user profile'}), 500
    user = user_result['user']

    return jsonify({
        'token': auth_result['session'].access_token,
        'refresh_token': auth_result['session'].refresh_token,
        'expires_in': auth_result['session'].expires_in,
        'user': user.to_dict()
    }), 200

@api_bp.route('/register', methods=['POST'])
def register():
    """User registration endpoint."""
    supabase_error = require_supabase()
    if supabase_error:
        return supabase_error

    try:
        data = request.get_json() or {}
        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        full_name = data.get('full_name', '').strip() or None

        if not username or not email or not password:
            return jsonify({'error': 'Username, email, and password are required'}), 400

        if supabase_client.get_user_by_username(username).get('success'):
            return jsonify({'error': 'Username already exists'}), 400

        if supabase_client.get_user_by_email(email).get('success'):
            return jsonify({'error': 'Email already exists'}), 400

        count_result = supabase_client.count_users()
        is_first_user = count_result.get('count', 0) == 0 if count_result.get('success') else False

        auth_result = supabase_client.register_auth_user(
            email=email,
            password=password,
            username=username,
            full_name=full_name,
            is_admin=is_first_user,
        )
        if not auth_result.get('success'):
            return jsonify({'error': auth_result.get('error', 'Registration failed')}), 400

        user_result = supabase_client.get_user_record(auth_result['user'].id)
        if not user_result.get('success'):
            return jsonify({'error': 'User created, but profile could not be loaded'}), 500
        user = user_result['user']

        return jsonify({
            'success': True,
            'message': 'Account created. Check your email to confirm your account before signing in.',
            'user': user.to_dict()
        }), 201
    except Exception as exc:
        logger.exception("Registration failed")
        return jsonify({'error': f'Registration failed: {exc}'}), 500

@api_bp.route('/predict/public', methods=['POST'])
@rate_limiter
@request_logger
def predict_public():
    """Public prediction endpoint for the landing page."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    result = prediction_service.predict_sync(file)
    if not result.get('success'):
        return jsonify({'error': result.get('error', 'Prediction failed')}), 500

    result['saved'] = False
    result['auth_required_for_history'] = True
    return jsonify(result), 200

@api_bp.route('/predict', methods=['POST'])
@auth_middleware
@rate_limiter
@request_logger
def predict():
    """Authenticated single-image prediction with history persistence."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    result = prediction_service.predict_sync(file, g.user.id)
    if not result.get('success'):
        return jsonify({'error': result.get('error', 'Prediction failed')}), 500

    save_result = supabase_client.create_prediction(
        user_id=g.user.id,
        image_name=file.filename,
        prediction=result['prediction'],
        confidence=result['confidence'],
        probabilities=result['probabilities'],
        processing_time=result['processing_time'],
        ip_address=request.remote_addr,
        user_agent=request.headers.get('User-Agent'),
    )
    if not save_result.get('success'):
        return jsonify({'error': save_result.get('error', 'Prediction could not be saved')}), 500

    prediction = save_result['prediction']

    result['prediction_id'] = prediction.id
    result['saved'] = True
    return jsonify(result), 200

@api_bp.route('/batch_predict', methods=['POST'])
@auth_middleware
@rate_limiter
@request_logger
def batch_predict():
    """Authenticated batch prediction endpoint."""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files[]')
    if len(files) > current_app.config.get('MAX_BATCH_SIZE', 10):
        return jsonify({'error': f'Maximum batch size is {current_app.config.get("MAX_BATCH_SIZE")}'}), 400

    results = prediction_service.batch_predict(files, g.user.id)
    return jsonify({
        'success': True,
        'total': len(results),
        'results': results
    }), 200

@api_bp.route('/predictions', methods=['GET'])
@auth_middleware
def get_predictions():
    """Return paginated predictions for the current user."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    days = request.args.get('days', '30')
    disease = request.args.get('disease', 'all')
    confidence = request.args.get('confidence', 'all')

    predictions_result = supabase_client.get_predictions(
        g.user.id,
        is_admin=g.user.is_admin,
        page=page,
        per_page=per_page,
        days=days,
        disease=disease,
        confidence=confidence,
    )
    if not predictions_result.get('success'):
        return jsonify({'error': predictions_result.get('error', 'Unable to fetch predictions')}), 500

    return jsonify({
        'total': predictions_result['total'],
        'pages': predictions_result['pages'],
        'current_page': predictions_result['current_page'],
        'predictions': [prediction.to_dict() for prediction in predictions_result['predictions']]
    }), 200

@api_bp.route('/predictions/<int:prediction_id>', methods=['GET'])
@auth_middleware
def get_prediction(prediction_id):
    """Return a single prediction for the current user."""
    prediction_result = supabase_client.get_prediction(prediction_id, g.user.id, is_admin=g.user.is_admin)
    if not prediction_result.get('success'):
        return jsonify({'error': 'Prediction not found'}), 404
    return jsonify(prediction_result['prediction'].to_dict()), 200

@api_bp.route('/analytics', methods=['GET'])
@auth_middleware
def get_analytics():
    """Return per-user analytics."""
    period = request.args.get('period', 'week')
    return jsonify(analytics_service.get_user_analytics(g.user.id, period)), 200

@api_bp.route('/analytics/system', methods=['GET'])
@auth_middleware
def get_system_analytics():
    """Return system-wide analytics for admin users."""
    if not g.user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    return jsonify(analytics_service.get_system_analytics()), 200

@api_bp.route('/export/<int:prediction_id>', methods=['GET'])
@auth_middleware
def export_prediction(prediction_id):
    """Export a single prediction."""
    prediction_result = supabase_client.get_prediction(prediction_id, g.user.id, is_admin=g.user.is_admin)
    if not prediction_result.get('success'):
        return jsonify({'error': 'Prediction not found'}), 404
    prediction = prediction_result['prediction']

    export_format = request.args.get('format', 'pdf')
    if export_format == 'pdf':
        pdf = export_service.generate_pdf(prediction)
        return send_file(pdf, mimetype='application/pdf', as_attachment=True, download_name=f'prediction_{prediction_id}.pdf')
    if export_format == 'csv':
        csv = export_service.generate_csv([prediction])
        return send_file(csv, mimetype='text/csv', as_attachment=True, download_name=f'prediction_{prediction_id}.csv')
    if export_format == 'json':
        return jsonify(prediction.to_dict())

    return jsonify({'error': 'Unsupported format'}), 400

@api_bp.route('/export/batch', methods=['POST'])
@auth_middleware
def export_batch():
    """Export multiple predictions for the current user."""
    data = request.get_json() or {}
    prediction_ids = data.get('prediction_ids', [])
    export_format = data.get('format', 'csv')

    predictions_result = supabase_client.get_predictions_by_ids(g.user.id, prediction_ids, is_admin=g.user.is_admin)
    if not predictions_result.get('success'):
        return jsonify({'error': predictions_result.get('error', 'Unable to export predictions')}), 500
    predictions = predictions_result['predictions']

    if export_format == 'csv':
        csv = export_service.generate_csv(predictions)
        return send_file(csv, mimetype='text/csv', as_attachment=True, download_name='predictions_export.csv')
    if export_format == 'pdf':
        pdf_bundle = export_service.generate_pdf_bundle(predictions)
        return send_file(
            pdf_bundle,
            mimetype='application/zip',
            as_attachment=True,
            download_name='predictions_export_pdf_bundle.zip'
        )
    if export_format == 'excel':
        excel = export_service.generate_excel(predictions)
        return send_file(
            excel,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='predictions_export.xlsx'
        )

    return jsonify({'error': 'Unsupported format'}), 400

@api_bp.route('/user/profile', methods=['GET'])
@auth_middleware
def get_profile():
    """Return the authenticated user's profile."""
    profile = g.user.to_dict()
    predictions_result = supabase_client.count_user_predictions(g.user.id)
    profile['predictions_count'] = predictions_result.get('total', 0) if predictions_result.get('success') else 0
    return jsonify(profile), 200

@api_bp.route('/user/profile', methods=['PUT'])
@auth_middleware
def update_profile():
    """Update the authenticated user's profile."""
    data = request.get_json() or {}

    username = data.get('username', '').strip()
    email = data.get('email', '').strip().lower()
    full_name = data.get('full_name', '').strip()

    if username and username != g.user.username:
        existing = supabase_client.get_user_by_username(username)
        if existing.get('success') and existing['profile'].get('id') != g.user.id:
            return jsonify({'error': 'Username already in use'}), 400

    if email and email != g.user.email:
        existing = supabase_client.get_user_by_email(email)
        if existing.get('success') and existing['profile'].get('id') != g.user.id:
            return jsonify({'error': 'Email already in use'}), 400

    result = supabase_client.update_auth_user(
        user_id=g.user.id,
        email=email or g.user.email,
        username=username or g.user.username,
        full_name=full_name if full_name else g.user.full_name,
        is_admin=g.user.is_admin,
    )
    if not result.get('success'):
        return jsonify({'error': result.get('error', 'Profile update failed')}), 400

    user_result = supabase_client.get_user_record(g.user.id)
    if not user_result.get('success'):
        return jsonify({'error': 'Profile updated, but refreshed profile could not be loaded'}), 500

    return jsonify({'success': True, 'message': 'Profile updated', 'user': user_result['user'].to_dict()}), 200

@api_bp.route('/password/reset-request', methods=['POST'])
def request_password_reset():
    """Request a password reset email through Supabase."""
    supabase_error = require_supabase()
    if supabase_error:
        return supabase_error

    data = request.get_json() or {}
    email = (data.get('email') or '').strip().lower()
    if not email:
        return jsonify({'error': 'Email is required'}), 400

    result = supabase_client.send_password_reset_email(email)
    if not result.get('success'):
        logger.warning("Password reset email could not be sent for %s: %s", email, result.get('error'))

    # Return a generic success response to avoid account enumeration.
    return jsonify({'success': True, 'message': 'If an account exists for this email, a reset link has been sent.'}), 200

@api_bp.route('/password/reset-confirm', methods=['POST'])
def confirm_password_reset():
    """Confirm and apply a new password using Supabase recovery access token."""
    supabase_error = require_supabase()
    if supabase_error:
        return supabase_error

    data = request.get_json() or {}
    access_token = (data.get('access_token') or '').strip()
    new_password = data.get('password') or ''

    if not access_token:
        return jsonify({'error': 'Recovery token is required'}), 400
    if len(new_password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters long'}), 400

    result = supabase_client.reset_password_with_access_token(access_token, new_password)
    if not result.get('success'):
        error_message = result.get('error', 'Password reset failed')
        if 'expired' in error_message.lower() or 'invalid' in error_message.lower():
            return jsonify({'error': 'Reset link is invalid or expired. Request a new password reset email.'}), 400
        return jsonify({'error': error_message}), 400

    return jsonify({'success': True, 'message': 'Password has been reset. You can now sign in.'}), 200

@api_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """Capture customer feedback."""
    supabase_error = require_supabase()
    if supabase_error:
        return supabase_error

    data = request.get_json() or {}
    message = (data.get('message') or '').strip()
    email = (data.get('email') or '').strip().lower() or None
    name = (data.get('name') or '').strip() or None
    category = (data.get('category') or '').strip() or None

    if not message:
        return jsonify({'error': 'Feedback message is required'}), 400

    user_id = None
    auth_header = request.headers.get('Authorization')
    if auth_header:
        try:
            token = auth_header.split(' ')[1] if ' ' in auth_header else auth_header
            auth_result = supabase_client.get_auth_user(token)
            if auth_result.get('success') and auth_result.get('user'):
                user_id = auth_result['user'].id
        except Exception:
            user_id = None

    result = supabase_client.create_feedback(
        message=message,
        email=email,
        name=name,
        category=category,
        user_id=user_id,
    )
    if not result.get('success'):
        return jsonify({'error': result.get('error', 'Unable to submit feedback')}), 500

    feedback_item = result.get('feedback', {})
    admin_email = current_app.config.get('ADMIN_EMAIL')
    send_system_email(
        subject='[MaizeGuard] New customer feedback received',
        body=(
            f"Feedback ID: {feedback_item.get('id')}\n"
            f"Category: {feedback_item.get('category') or 'General'}\n"
            f"Name: {feedback_item.get('name') or 'Not provided'}\n"
            f"Email: {feedback_item.get('email') or 'Not provided'}\n"
            f"Message:\n{feedback_item.get('message', '')}\n"
        ),
        recipient=admin_email,
    )

    return jsonify({'success': True, 'message': 'Thanks for your feedback.'}), 201

@api_bp.route('/admin/feedback', methods=['GET'])
@auth_middleware
def admin_feedback_list():
    """Return paginated feedback items for admin users."""
    if not g.user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403

    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    status = request.args.get('status', 'all')

    result = supabase_client.get_feedback(page=page, per_page=per_page, status=status)
    if not result.get('success'):
        return jsonify({'error': result.get('error', 'Unable to fetch feedback')}), 500

    return jsonify({
        'success': True,
        'feedback': result.get('feedback', []),
        'total': result.get('total', 0),
        'pages': result.get('pages', 0),
        'current_page': result.get('current_page', page),
    }), 200

@api_bp.route('/admin/feedback/<int:feedback_id>/reply', methods=['POST'])
@auth_middleware
def admin_feedback_reply(feedback_id):
    """Save an admin reply for a feedback item and send an email when possible."""
    if not g.user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json() or {}
    reply_text = (data.get('reply') or '').strip()
    if not reply_text:
        return jsonify({'error': 'Reply message is required'}), 400

    current_item = supabase_client.get_feedback_by_id(feedback_id)
    if not current_item.get('success'):
        return jsonify({'error': 'Feedback not found'}), 404

    result = supabase_client.reply_feedback(
        feedback_id=feedback_id,
        admin_response=reply_text,
        responded_by=g.user.id,
    )
    if not result.get('success'):
        return jsonify({'error': result.get('error', 'Unable to save reply')}), 500

    feedback_item = current_item.get('feedback', {})
    destination_email = feedback_item.get('email')
    if destination_email:
        send_system_email(
            subject='[MaizeGuard Support] Response to your feedback',
            body=(
                'Thank you for contacting MaizeGuard AI support.\n\n'
                f"Your message:\n{feedback_item.get('message', '')}\n\n"
                'Support response:\n'
                f"{reply_text}\n"
            ),
            recipient=destination_email,
        )

    return jsonify({'success': True, 'message': 'Reply sent and feedback updated.'}), 200
