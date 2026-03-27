"""
Analytics service backed by Supabase data.
"""
from datetime import datetime, timedelta, timezone
import logging

from app.database.supabase_client import supabase_client

logger = logging.getLogger(__name__)

def utc_now():
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)

class AnalyticsService:
    """Service for generating analytics and insights."""

    def _build_class_health(self, predictions, confidence_threshold: float):
        class_totals = {}
        class_conf_sums = {}
        class_low_conf = {}

        for pred in predictions:
            label = pred.prediction
            conf = float(pred.confidence or 0.0)

            class_totals[label] = class_totals.get(label, 0) + 1
            class_conf_sums[label] = class_conf_sums.get(label, 0.0) + conf
            if conf < confidence_threshold:
                class_low_conf[label] = class_low_conf.get(label, 0) + 1

        class_health = []
        for label, total in class_totals.items():
            avg_conf = class_conf_sums[label] / total if total else 0.0
            low_count = class_low_conf.get(label, 0)
            low_ratio = (low_count / total) if total else 0.0

            class_health.append({
                'class_name': label,
                'total_predictions': total,
                'average_confidence': avg_conf,
                'low_confidence_count': low_count,
                'low_confidence_ratio': low_ratio,
                'improvement_priority': round((1.0 - avg_conf) * low_ratio * total, 6),
            })

        class_health.sort(
            key=lambda item: (
                item['improvement_priority'],
                item['low_confidence_ratio'],
                -item['average_confidence'],
            ),
            reverse=True,
        )
        return class_health

    def _build_daily_predictions(self, predictions):
        daily_counts = {}
        for pred in predictions:
            if not pred.created_at:
                continue
            day = pred.created_at.strftime('%Y-%m-%d')
            daily_counts[day] = daily_counts.get(day, 0) + 1

        return [
            {'date': date, 'count': count}
            for date, count in sorted(daily_counts.items())
        ]

    def _range_for_period(self, period: str):
        end_date = utc_now()
        if period == 'day':
            start_date = end_date - timedelta(days=1)
        elif period == 'week':
            start_date = end_date - timedelta(days=7)
        elif period == 'month':
            start_date = end_date - timedelta(days=30)
        elif period == 'year':
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=7)
        return start_date, end_date

    def get_user_analytics(self, user_id: str, period: str = 'week') -> dict:
        """Get analytics for a specific user."""
        start_date, end_date = self._range_for_period(period)
        predictions_result = supabase_client.list_predictions_since(user_id, start_date.isoformat())
        predictions = predictions_result.get('predictions', []) if predictions_result.get('success') else []

        total_predictions = len(predictions)
        if total_predictions == 0:
            return {
                'total_predictions': 0,
                'disease_distribution': {},
                'average_confidence': 0,
                'most_common_disease': None,
                'period': period,
                'daily_predictions': [],
                'percentage_change': 0
            }

        disease_counts = {}
        for pred in predictions:
            disease_counts[pred.prediction] = disease_counts.get(pred.prediction, 0) + 1

        avg_confidence = sum(p.confidence for p in predictions) / total_predictions
        most_common = max(disease_counts.items(), key=lambda item: item[1])[0] if disease_counts else None

        previous_start = start_date - (end_date - start_date)
        previous_result = supabase_client.list_predictions_since(
            user_id,
            previous_start.isoformat(),
            start_date.isoformat(),
        )
        previous_total = len(previous_result.get('predictions', [])) if previous_result.get('success') else 0

        if previous_total == 0:
            percentage_change = 100.0 if total_predictions > 0 else 0.0
        else:
            percentage_change = ((total_predictions - previous_total) / previous_total) * 100

        return {
            'total_predictions': total_predictions,
            'disease_distribution': disease_counts,
            'average_confidence': avg_confidence,
            'most_common_disease': most_common,
            'period': period,
            'daily_predictions': self._build_daily_predictions(predictions),
            'percentage_change': percentage_change
        }

    def get_class_monitoring(self, user_id: str, period: str = 'month', confidence_threshold: float = 0.60) -> dict:
        """Return class-level confidence health indicators for data collection planning."""
        start_date, _ = self._range_for_period(period)
        predictions_result = supabase_client.list_predictions_since(user_id, start_date.isoformat())
        predictions = predictions_result.get('predictions', []) if predictions_result.get('success') else []

        class_health = self._build_class_health(predictions, confidence_threshold)
        weak_classes = [
            item for item in class_health
            if item['average_confidence'] < confidence_threshold or item['low_confidence_ratio'] >= 0.25
        ]

        return {
            'period': period,
            'confidence_threshold': confidence_threshold,
            'total_predictions': len(predictions),
            'classes_tracked': len(class_health),
            'class_health': class_health,
            'weak_classes': weak_classes,
        }

    def build_data_collection_template_rows(self, monitoring_data: dict):
        """Build CSV rows for data collection of weak classes."""
        rows = []
        weak_classes = monitoring_data.get('weak_classes', [])
        threshold = monitoring_data.get('confidence_threshold', 0.60)

        for rank, item in enumerate(weak_classes, start=1):
            rows.append({
                'priority_rank': rank,
                'class_name': item.get('class_name', ''),
                'target_new_images': 250,
                'min_capture_resolution': '1024x1024',
                'lighting_condition': 'Natural light preferred',
                'leaf_stage': 'young|mature|mixed',
                'region_or_farm': '',
                'capture_device': '',
                'symptom_severity': 'mild|moderate|severe',
                'expert_verified_label': 'yes|no',
                'current_avg_confidence': round(float(item.get('average_confidence', 0.0)), 4),
                'low_confidence_ratio': round(float(item.get('low_confidence_ratio', 0.0)), 4),
                'confidence_threshold': threshold,
                'notes': 'Collect varied backgrounds and angles for this class',
            })

        return rows

    def get_system_analytics(self) -> dict:
        """Get system-wide analytics."""
        users_result = supabase_client.count_users()
        predictions_result = supabase_client.count_predictions()
        distribution_result = supabase_client.get_prediction_distribution()
        errors_result = supabase_client.count_system_logs_since(
            'ERROR',
            (utc_now() - timedelta(days=7)).isoformat(),
        )

        total_users = users_result.get('count', 0) if users_result.get('success') else 0
        total_predictions = predictions_result.get('count', 0) if predictions_result.get('success') else 0
        disease_distribution = distribution_result.get('distribution', []) if distribution_result.get('success') else []
        error_count = errors_result.get('count', 0)

        return {
            'users': {
                'total': total_users
            },
            'predictions': {
                'total': total_predictions,
                'daily_avg': total_predictions / 30 if total_predictions > 0 else 0
            },
            'disease_distribution': disease_distribution,
            'errors_last_7_days': error_count,
            'system_health': self.check_system_health()
        }

    def check_system_health(self) -> dict:
        """Check system health status."""
        health_status = {
            'status': 'healthy',
            'checks': {}
        }

        db_result = supabase_client.ping()
        if db_result.get('success'):
            health_status['checks']['database'] = 'healthy'
        else:
            health_status['checks']['database'] = f"unhealthy: {db_result.get('error', 'unknown error')}"
            health_status['status'] = 'degraded'

        try:
            from app.services.prediction_service import PredictionService
            service = PredictionService()
            if service.model:
                health_status['checks']['model'] = 'healthy'
            else:
                health_status['checks']['model'] = 'unhealthy'
                health_status['status'] = 'degraded'
        except Exception as exc:
            logger.error("Model health check failed: %s", exc)
            health_status['checks']['model'] = f'unhealthy: {str(exc)}'
            health_status['status'] = 'degraded'

        return health_status
