"""
Prediction service for handling image classification
"""
import os
import sys
import json
import cv2
import numpy as np
import joblib
import time
import logging
from typing import Dict, List, Any, Optional

# Add the project root to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for making predictions with the Random Forest model"""
    
    def __init__(self, model_path=None, labels_path=None):
        """
        Initialize the prediction service
        
        Args:
            model_path: Path to the trained model (optional)
        """
        # Set model path
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'models', 'maize_disease_classifier.pkl'
            )
        
        self.model_path = model_path
        self.labels_path = labels_path or os.path.join(
            os.path.dirname(self.model_path),
            'class_labels.json'
        )
        self.model = None
        self.preprocessor = None
        
        # Default class names - will be overwritten from metadata/model classes when available.
        # Starts empty for multi-crop support; dynamically loaded at initialization.
        self.class_names = []
        self.class_colors = []
        self.crop_class_map = {}
        self.img_size = (128, 128)
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessor"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"âœ… Model loaded from {self.model_path}")
            else:
                logger.error(f"âŒ Model not found at {self.model_path}")
                raise FileNotFoundError(f"Model not found at {self.model_path}")

            self.class_names = self._load_class_names()
            self.class_colors = self._build_class_colors(len(self.class_names))
            self.crop_class_map = self._build_crop_class_map(self.class_names)
            
            # Import preprocessor from src - supports LeafPreprocessor for multi-crop
            try:
                # Try to import multi-crop LeafPreprocessor first
                try:
                    from src.data_preprocessing import LeafPreprocessor
                    self.preprocessor = LeafPreprocessor(img_size=self.img_size)
                    logger.info("✅ Multi-crop LeafPreprocessor initialized")
                except ImportError:
                    # Fall back to MaizeLeafPreprocessor for backward compatibility
                    from src.data_preprocessing import MaizeLeafPreprocessor
                    self.preprocessor = MaizeLeafPreprocessor(img_size=self.img_size)
                    logger.info("✅ Legacy MaizeLeafPreprocessor initialized")
            except ImportError as e:
                logger.error(f"âŒ Failed to import preprocessor: {e}")
                logger.info("Creating fallback preprocessor...")
                self._create_fallback_preprocessor()
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _load_class_names(self):
        """Load class names from metadata, then model classes, then defaults."""
        if os.path.exists(self.labels_path):
            try:
                with open(self.labels_path, 'r', encoding='utf-8') as labels_file:
                    payload = json.load(labels_file)
                    classes = payload.get('classes', [])
                    if classes:
                        logger.info(f"Loaded {len(classes)} classes from metadata")
                        return classes
            except Exception as exc:
                logger.warning(f"Could not read labels metadata: {exc}")

        model_classes = getattr(self.model, 'classes_', None)
        if model_classes is not None and len(model_classes) > 0:
            if all(isinstance(class_name, (int, np.integer)) for class_name in model_classes):
                if len(model_classes) == len(self.class_names):
                    logger.info("Model classes are numeric; using default legacy class names")
                    return self.class_names

            resolved = [str(class_name) for class_name in model_classes]
            logger.info(f"Loaded {len(resolved)} classes from model")
            return resolved

        logger.warning("Using built-in default classes because no metadata/model classes were found")
        return self.class_names

    @staticmethod
    def _build_class_colors(num_classes):
        """Generate a deterministic color palette sized to the class count."""
        palette = [
            '#dc3545', '#ffc107', '#28a745', '#17a2b8', '#6f42c1', '#fd7e14', '#20c997',
            '#0d6efd', '#e83e8c', '#198754', '#6c757d', '#6610f2'
        ]
        return [palette[i % len(palette)] for i in range(num_classes)]

    @staticmethod
    def _format_label(label):
        """Convert machine-friendly labels to display labels."""
        return str(label).replace('___', ' / ').replace('_', ' ')

    @staticmethod
    def _normalize_crop_key(crop: str) -> str:
        """Normalize crop keys from user input, URL segments, and label prefixes."""
        return str(crop or '').strip().lower().replace('-', '_').replace(' ', '_')

    @staticmethod
    def _format_crop_name(crop_key: str) -> str:
        """Format crop key for user-facing display."""
        return str(crop_key or '').replace('_', ' ').title()

    def _build_crop_class_map(self, classes: List[str]) -> Dict[str, List[int]]:
        """Build class-index groups for each crop from class labels."""
        crop_map: Dict[str, List[int]] = {}

        for index, class_name in enumerate(classes):
            class_name = str(class_name)
            if '___' in class_name:
                crop_prefix = class_name.split('___', 1)[0]
            else:
                # Legacy maize-only labels without crop prefix.
                crop_prefix = 'maize'

            crop_key = self._normalize_crop_key(crop_prefix)
            crop_map.setdefault(crop_key, []).append(index)

        return crop_map

    def _resolve_crop_key(self, crop: Optional[str]) -> Optional[str]:
        """Resolve a crop alias to a normalized crop key known by the model."""
        if not crop:
            return None

        crop_key = self._normalize_crop_key(crop)
        aliases = {
            'pepper': 'pepper_bell',
            'bell_pepper': 'pepper_bell',
            'pepper_bell': 'pepper_bell',
        }
        crop_key = aliases.get(crop_key, crop_key)

        if crop_key in self.crop_class_map:
            return crop_key
        return None

    def get_available_crops(self) -> List[Dict[str, Any]]:
        """List crops detected in model class metadata."""
        available = []
        for crop_key in sorted(self.crop_class_map.keys()):
            available.append({
                'key': crop_key,
                'name': self._format_crop_name(crop_key),
                'class_count': len(self.crop_class_map[crop_key]),
            })
        return available
    
    def _create_fallback_preprocessor(self):
        """Create a simple fallback preprocessor if import fails"""
        class SimplePreprocessor:
            def __init__(self, img_size):
                self.img_size = img_size
            
            def _load_and_preprocess_image(self, img_path):
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        return None
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    img = cv2.GaussianBlur(img, (5, 5), 0)
                    img = img.astype(np.float32) / 255.0
                    return img
                except:
                    return None
            
            def extract_features(self, images):
                """Simple feature extraction"""
                features = []
                for img in images:
                    # Simple color histogram features
                    features.append([
                        np.mean(img[:,:,0]), np.std(img[:,:,0]),
                        np.mean(img[:,:,1]), np.std(img[:,:,1]),
                        np.mean(img[:,:,2]), np.std(img[:,:,2])
                    ])
                return np.array(features)
        
        self.preprocessor = SimplePreprocessor(self.img_size)
        logger.warning("âš ï¸ Using fallback preprocessor - model may have reduced accuracy")
    
    def predict_sync(self, file, user_id=None, crop: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous prediction"""
        start_time = time.time()
        
        try:
            # Read image
            img_bytes = file.read()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize and preprocess
            img = cv2.resize(img, self.img_size)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img_normalized = img.astype(np.float32) / 255.0
            
            # Extract features
            features = self.preprocessor.extract_features([img_normalized])
            
            # Make prediction and align probabilities to resolved labels.
            predicted_raw = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]

            model_classes = getattr(self.model, 'classes_', None)
            model_classes_are_numeric = (
                model_classes is not None
                and len(model_classes) == len(probabilities)
                and all(isinstance(class_name, (int, np.integer)) for class_name in model_classes)
            )

            if self.class_names and len(self.class_names) == len(probabilities):
                probability_labels = [str(class_name) for class_name in self.class_names]
            elif model_classes is not None and len(model_classes) == len(probabilities):
                probability_labels = [str(class_name) for class_name in model_classes]
            else:
                probability_labels = [f'class_{i}' for i in range(len(probabilities))]

            predicted_index = None
            if model_classes_are_numeric:
                try:
                    candidate_index = int(predicted_raw)
                    if 0 <= candidate_index < len(probabilities):
                        predicted_index = candidate_index
                except (TypeError, ValueError):
                    predicted_index = None
            elif model_classes is not None and len(model_classes) > 0:
                matching = np.where(np.asarray(model_classes) == predicted_raw)[0]
                if len(matching) > 0:
                    predicted_index = int(matching[0])

            if predicted_index is None:
                predicted_index = int(np.argmax(probabilities))

            crop_key = self._resolve_crop_key(crop)
            if crop and not crop_key:
                return {
                    'success': False,
                    'error': f"Unsupported crop '{crop}'",
                    'available_crops': self.get_available_crops(),
                }

            selected_indices = list(range(len(probabilities)))
            selected_probabilities = probabilities
            selected_labels = probability_labels

            if crop_key:
                selected_indices = self.crop_class_map.get(crop_key, [])
                if not selected_indices:
                    return {
                        'success': False,
                        'error': f"No classes available for crop '{crop}'",
                        'available_crops': self.get_available_crops(),
                    }

                selected_probabilities = probabilities[selected_indices]
                selected_labels = [probability_labels[index] for index in selected_indices]

                selected_sum = float(np.sum(selected_probabilities))
                if selected_sum > 0:
                    selected_probabilities = selected_probabilities / selected_sum

                predicted_index = int(np.argmax(selected_probabilities))

            prediction_label = selected_labels[predicted_index]
            confidence = float(selected_probabilities[predicted_index])
            
            processing_time = (time.time() - start_time) * 1000
            
            result = {
                'success': True,
                'prediction': self._format_label(prediction_label),
                'prediction_label': prediction_label,
                'crop': crop_key,
                'confidence': confidence,
                'probabilities': [
                    {
                        'class': selected_labels[i],
                        'display_class': self._format_label(selected_labels[i]),
                        'probability': float(selected_probabilities[i]),
                        'color': self.class_colors[i % len(self.class_colors)] if self.class_colors else '#2f6b42'
                    }
                    for i in range(len(selected_labels))
                ],
                'top_predictions': [
                    {
                        'class': selected_labels[i],
                        'display_class': self._format_label(selected_labels[i]),
                        'probability': float(selected_probabilities[i]),
                    }
                    for i in np.argsort(selected_probabilities)[::-1][:3]
                ],
                'processing_time': processing_time
            }
            
            logger.info(f"Prediction: {result['prediction']} with {result['confidence']:.2%} confidence")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def batch_predict(self, files, user_id=None, crop: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run prediction for multiple uploaded files."""
        results = []
        for file in files:
            result = self.predict_sync(file, user_id=user_id, crop=crop)
            result['filename'] = getattr(file, 'filename', 'unknown')
            results.append(result)
        return results
    
    def get_model_info(self, crop: Optional[str] = None) -> Dict[str, Any]:
        """Return model information"""
        feature_count = getattr(self.model, 'n_features_in_', None) if self.model is not None else None
        crop_key = self._resolve_crop_key(crop)

        selected_classes = self.class_names
        if crop_key:
            selected_classes = [self.class_names[index] for index in self.crop_class_map.get(crop_key, [])]

        return {
            'algorithm': 'Random Forest',
            'model_type': 'Random Forest Classifier',
            'n_estimators': self.model.n_estimators if self.model else None,
            'crop': crop_key,
            'available_crops': self.get_available_crops(),
            'classes': selected_classes,
            'display_classes': [self._format_label(class_name) for class_name in selected_classes],
            'features': feature_count,
            'image_size': self.img_size,
            'version': '2.0.0',
            'accuracy': 0.917,
            'max_file_size': '16MB'
        }
