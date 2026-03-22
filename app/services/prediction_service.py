"""
Prediction service for handling image classification
"""
import os
import sys
import cv2
import numpy as np
import joblib
import time
import logging
from typing import Dict, List, Any

# Add the project root to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for making predictions with the Random Forest model"""
    
    def __init__(self, model_path=None):
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
        self.model = None
        self.preprocessor = None
        
        # Class names
        self.class_names = ['Blight', 'Gray Leaf Spot', 'Healthy', 'Maize Rust']
        self.class_colors = ['#dc3545', '#ffc107', '#28a745', '#17a2b8']
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
            
            # Import preprocessor from src
            try:
                from src.data_preprocessing import MaizeLeafPreprocessor
                self.preprocessor = MaizeLeafPreprocessor(img_size=self.img_size)
                logger.info("âœ… Preprocessor initialized")
            except ImportError as e:
                logger.error(f"âŒ Failed to import preprocessor: {e}")
                logger.info("Creating fallback preprocessor...")
                self._create_fallback_preprocessor()
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
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
    
    def predict_sync(self, file, user_id=None) -> Dict[str, Any]:
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
            
            # Make prediction
            prediction_idx = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            processing_time = (time.time() - start_time) * 1000
            
            result = {
                'success': True,
                'prediction': self.class_names[prediction_idx],
                'confidence': float(probabilities[prediction_idx]),
                'probabilities': [
                    {
                        'class': self.class_names[i],
                        'probability': float(probabilities[i]),
                        'color': self.class_colors[i]
                    }
                    for i in range(len(self.class_names))
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

    def batch_predict(self, files, user_id=None) -> List[Dict[str, Any]]:
        """Run prediction for multiple uploaded files."""
        results = []
        for file in files:
            result = self.predict_sync(file, user_id=user_id)
            result['filename'] = getattr(file, 'filename', 'unknown')
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information"""
        feature_count = getattr(self.model, 'n_features_in_', None) if self.model is not None else None
        return {
            'algorithm': 'Random Forest',
            'model_type': 'Random Forest Classifier',
            'n_estimators': self.model.n_estimators if self.model else None,
            'classes': self.class_names,
            'features': feature_count,
            'image_size': self.img_size,
            'version': '2.0.0',
            'accuracy': 0.917,
            'max_file_size': '16MB'
        }
