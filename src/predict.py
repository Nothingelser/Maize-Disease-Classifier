"""
Module for making predictions on new images
"""
import os
import json
import cv2
import numpy as np
import joblib
from src.data_preprocessing import LeafPreprocessor


def load_classes(model, model_path):
    """Resolve class labels from metadata first, then model classes as fallback."""
    labels_path = os.path.join(os.path.dirname(model_path), 'class_labels.json')

    if os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as labels_file:
            payload = json.load(labels_file)
            classes = payload.get('classes', [])
            if classes:
                return classes

    model_classes = getattr(model, 'classes_', None)
    if model_classes is not None and len(model_classes) > 0:
        return [str(class_name) for class_name in model_classes]

    return ['Maize___Blight', 'Maize___Gray_Leaf_Spot', 'Maize___Healthy', 'Maize___Rust']

def predict_single_image(image_path, model_path, classes=None):
    """
    Predict disease for a single leaf image.
    
    Args:
        image_path: path to the image file
        model_path: path to saved model
        classes: list of class names (if None, will use defaults)
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    if classes is None:
        classes = load_classes(model, model_path)
    
    # Preprocess image
    print(f"Processing image: {image_path}")
    preprocessor = LeafPreprocessor(img_size=(128, 128))
    img = preprocessor._load_and_preprocess_image(image_path)
    
    if img is None:
        print("âŒ Failed to load image")
        return None
    
    # Extract features
    features = preprocessor.extract_features([img])
    
    # Make prediction
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Get class name
    predicted_class = classes[prediction]
    display_class = predicted_class.replace('___', ' / ').replace('_', ' ')
    
    # Display results
    print("\n" + "="*50)
    print("ðŸŒ½ PREDICTION RESULTS")
    print("="*50)
    print(f"ðŸ“Œ Predicted Disease: {display_class}")
    print("\nðŸ“Š Class Probabilities:")
    for i, prob in enumerate(probabilities):
        percentage = prob * 100
        bar = "â–ˆ" * int(percentage/5) + "â–‘" * (20 - int(percentage/5))
        display_name = str(classes[i]).replace('___', ' / ').replace('_', ' ')
        print(f"  {display_name:20s}: {bar} {percentage:.1f}%")
    print("="*50)
    
    return predicted_class, probabilities

if __name__ == "__main__":
    # Example usage
    image_path = input("Enter path to plant leaf image: ")
    model_path = "models/maize_disease_classifier.pkl"
    
    predict_single_image(image_path, model_path)