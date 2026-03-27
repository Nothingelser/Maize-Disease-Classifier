"""
Model training module for Random Forest classifier
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MaizeDiseaseClassifier:
    """
    Random Forest classifier for maize disease detection
    """
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42, class_weight=None):
        """
        Initialize the classifier
        
        Args:
            n_estimators: number of trees in the forest
            max_depth: maximum depth of trees
            random_state: seed for reproducibility
            class_weight: class weighting strategy for imbalanced datasets
        """
        self.class_weight = class_weight
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1  # Use all available CPU cores
        )
        self.classes = None
        self.feature_importances_ = None
    
    def train(self, X_train, y_train):
        """
        Train the Random Forest model
        
        Args:
            X_train: training features
            y_train: training labels
        """
        logger.info("Training Random Forest Classifier...")
        logger.info(f"Number of trees: {self.model.n_estimators}")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Class weight: {self.model.class_weight}")
        
        self.model.fit(X_train, y_train)
        self.classes = self.model.classes_
        self.feature_importances_ = self.model.feature_importances_
        
        train_score = self.model.score(X_train, y_train)
        logger.info(f"âœ… Training completed!")
        logger.info(f"Training accuracy: {train_score:.4f}")
    
    def optimize_hyperparameters(self, X_train, y_train):
        """
        Find best hyperparameters using grid search
        
        Args:
            X_train: training features
            y_train: training labels
            
        Returns:
            best_params: dictionary of best hyperparameters
        """
        logger.info("Optimizing hyperparameters...")
        
        # Define parameters to try
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create grid search
        grid_search = GridSearchCV(
            RandomForestClassifier(
                random_state=42,
                class_weight=self.class_weight
            ),
            param_grid,
            cv=5,  # 5-fold cross validation
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Run grid search
        grid_search.fit(X_train, y_train)
        
        logger.info(f"\nâœ… Best parameters found:")
        for param, value in grid_search.best_params_.items():
            logger.info(f"  {param}: {value}")
        logger.info(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
        
        # Use the best model
        self.model = grid_search.best_estimator_
        self.classes = self.model.classes_
        self.feature_importances_ = self.model.feature_importances_
        
        return grid_search.best_params_
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: features to predict
            
        Returns:
            predicted class indices
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: features to predict
            
        Returns:
            probability for each class
        """
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance scores
        
        Args:
            feature_names: list of feature names (optional)
            
        Returns:
            sorted list of (feature_name, importance) tuples
        """
        if self.feature_importances_ is None:
            return None
        
        importances = self.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        if feature_names:
            return [(feature_names[i], importances[i]) for i in indices]
        else:
            return [(f"Feature_{i}", importances[i]) for i in indices]
    
    def save_model(self, filepath):
        """
        Save trained model to disk
        
        Args:
            filepath: path where to save the model
        """
        joblib.dump(self.model, filepath)
        logger.info(f"âœ… Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from disk
        
        Args:
            filepath: path to the saved model
        """
        self.model = joblib.load(filepath)
        self.classes = self.model.classes_
        self.feature_importances_ = self.model.feature_importances_
        logger.info(f"âœ… Model loaded from {filepath}")