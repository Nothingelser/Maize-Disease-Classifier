"""
Main pipeline for multi-crop plant disease classification project
"""
import os
import json
from src.data_preprocessing import LeafPreprocessor
from src.model_training import MaizeDiseaseClassifier
from src.evaluation import ModelEvaluator


def save_class_metadata(classes, output_path):
    """Persist class labels so inference can decode model outputs correctly."""
    payload = {
        'classes': [str(class_name) for class_name in classes]
    }
    with open(output_path, 'w', encoding='utf-8') as metadata_file:
        json.dump(payload, metadata_file, indent=2)

def main():
    """
    Main execution function
    """
    print("\n" + "="*60)
    print("ðŸŒ± PLANT DISEASE CLASSIFICATION PROJECT")
    print("="*60 + "\n")
    
    # Configuration
    DATA_PATH = "data/raw"  # Adjust this path!
    MODEL_PATH = "models/maize_disease_classifier.pkl"
    IMG_SIZE = (128, 128)
    
    # Check if data path exists
    if not os.path.exists(DATA_PATH):
        print(f"âŒ ERROR: Data path not found: {DATA_PATH}")
        print("\nPlease update the DATA_PATH in main.py to point to your dataset.")
        print("Common locations:")
        print("  - Downloads folder: C:/Users/YOURNAME/Downloads/PlantVillage")
        print("  - Project folder: data/raw/PlantVillage")
        return
    
    # Step 1: Preprocess data
    print("\n" + "-"*40)
    print("ðŸ“ STEP 1: Data Preprocessing")
    print("-"*40)
    
    preprocessor = LeafPreprocessor(img_size=IMG_SIZE)
    data = preprocessor.prepare_dataset(DATA_PATH, test_size=0.3)
    
    # Step 2: Train model
    print("\n" + "-"*40)
    print("ðŸ¤– STEP 2: Model Training")
    print("-"*40)
    
    classifier = MaizeDiseaseClassifier(n_estimators=100)
    
    # Ask if user wants hyperparameter optimization
    optimize = input("\nDo you want to optimize hyperparameters? (y/n, default: n): ").lower()
    
    if optimize == 'y':
        classifier.optimize_hyperparameters(data['X_train'], data['y_train'])
    else:
        classifier.train(data['X_train'], data['y_train'])
    
    # Save model
    os.makedirs('models', exist_ok=True)
    classifier.save_model(MODEL_PATH)
    save_class_metadata(data['classes'], 'models/class_labels.json')
    
    # Step 3: Evaluate model
    print("\n" + "-"*40)
    print("ðŸ“Š STEP 3: Model Evaluation")
    print("-"*40)
    
    evaluator = ModelEvaluator(classifier, data['classes'])
    results = evaluator.evaluate(data['X_test'], data['y_test'])
    
    # Print metrics
    evaluator.print_metrics(results['metrics'])
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        results['confusion_matrix'],
        save_path='reports/confusion_matrix.png'
    )
    
    print("\n" + "="*60)
    print("âœ… PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nResults saved in:")
    print("  - Trained model: models/maize_disease_classifier.pkl")
    print("  - Class metadata: models/class_labels.json")
    print("  - Confusion matrix: reports/confusion_matrix.png")
    if 'feature_importance' in results:
        print("  - Feature importance: reports/feature_importance.png")

if __name__ == "__main__":
    main()