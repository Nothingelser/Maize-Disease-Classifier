"""
Model evaluation module
Provides metrics and visualizations for model performance
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluates and visualizes model performance
    """
    
    def __init__(self, model, class_names):
        """
        Initialize evaluator
        
        Args:
            model: trained model
            class_names: list of class names
        """
        self.model = model
        self.class_names = class_names
    
    def evaluate(self, X_test, y_test):
        """
        Comprehensive model evaluation
        
        Args:
            X_test: test features
            y_test: true labels
            
        Returns:
            dictionary with all evaluation metrics
        """
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Generate classification report
        report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_prob
        }
    
    def plot_confusion_matrix(self, cm, save_path=None, figsize=(10, 8)):
        """
        Plot confusion matrix as heatmap
        
        Args:
            cm: confusion matrix
            save_path: optional path to save the plot
            figsize: figure size tuple
        """
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            annot_kws={'size': 12}
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"âœ… Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names, top_n=20, save_path=None):
        """
        Plot feature importance from Random Forest
        
        Args:
            feature_names: list of feature names
            top_n: number of top features to show
            save_path: optional path to save the plot
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        plt.bar(range(top_n), importances[indices], color='#2ecc71', alpha=0.7)
        plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        
        # Set x-tick labels
        if feature_names and len(feature_names) > 0:
            plt.xticks(range(top_n), [feature_names[i] for i in indices], 
                       rotation=45, ha='right', fontsize=8)
        else:
            plt.xticks(range(top_n), [f"Feature_{i}" for i in indices], 
                       rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"âœ… Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def print_metrics(self, metrics):
        """
        Print evaluation metrics in a nice format
        
        Args:
            metrics: dictionary from evaluate() method
        """
        print("\n" + "="*50)
        print("ðŸ“Š MODEL EVALUATION METRICS")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print("="*50)
    
    def print_classification_report(self, report):
        """
        Print classification report
        
        Args:
            report: classification report dictionary
        """
        print("\n" + "="*50)
        print("ðŸ“‹ CLASSIFICATION REPORT")
        print("="*50)
        
        for class_name in self.class_names:
            if class_name in report:
                print(f"\n{class_name}:")
                print(f"  Precision: {report[class_name]['precision']:.4f}")
                print(f"  Recall:    {report[class_name]['recall']:.4f}")
                print(f"  F1-Score:  {report[class_name]['f1-score']:.4f}")
                print(f"  Support:   {report[class_name]['support']}")
        
        print("\n" + "="*50)
        print(f"Macro Avg:  P={report['macro avg']['precision']:.4f} "
              f"R={report['macro avg']['recall']:.4f} "
              f"F1={report['macro avg']['f1-score']:.4f}")
        print(f"Weighted Avg: P={report['weighted avg']['precision']:.4f} "
              f"R={report['weighted avg']['recall']:.4f} "
              f"F1={report['weighted avg']['f1-score']:.4f}")
        print("="*50)