"""
Module for training Random Forest classifier
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np


class MaizeDiseaseClassifier:
    """
    Random Forest classifier for maize disease detection
    """
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize the classifier
        
        Args:
            n_estimators: number of trees in the forest
            max_depth: maximum depth of trees
            random_state: seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # Use all available CPU cores
        )
        self.classes = None
    
    def train(self, X_train, y_train):
        """
        Train the Random Forest model
        
        Args:
            X_train: training features
            y_train: training labels
        """
        print("Training Random Forest Classifier...")
        print(f"Number of trees: {self.model.n_estimators}")
        
        self.model.fit(X_train, y_train)
        self.classes = self.model.classes_
        
        print("✅ Training completed!")
        print(f"Training accuracy: {self.model.score(X_train, y_train):.4f}")
    
    def optimize_hyperparameters(self, X_train, y_train):
        """
        Find best hyperparameters using grid search
        
        Args:
            X_train: training features
            y_train: training labels
        """
        print("Optimizing hyperparameters...")
        
        # Define parameters to try
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create grid search
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,  # 5-fold cross validation
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Run grid search
        grid_search.fit(X_train, y_train)
        
        print(f"\n✅ Best parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
        
        # Use the best model
        self.model = grid_search.best_estimator_
        self.classes = self.model.classes_
        
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
    
    def save_model(self, filepath):
        """
        Save trained model to disk
        
        Args:
            filepath: path where to save the model
        """
        joblib.dump(self.model, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from disk
        
        Args:
            filepath: path to the saved model
        """
        self.model = joblib.load(filepath)
        self.classes = self.model.classes_
        print(f"✅ Model loaded from {filepath}")