"""
Module for making predictions on new images
"""
import cv2
import numpy as np
import joblib
from data_preprocessing import MaizeLeafPreprocessor

def predict_single_image(image_path, model_path, classes=None):
    """
    Predict disease for a single maize leaf image
    
    Args:
        image_path: path to the image file
        model_path: path to saved model
        classes: list of class names (if None, will use defaults)
    """
    # Default classes from our project
    if classes is None:
        classes = ['Blight', 'Gray_Leaf_Spot', 'Healthy', 'Maize_Rust']
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Preprocess image
    print(f"Processing image: {image_path}")
    preprocessor = MaizeLeafPreprocessor(img_size=(128, 128))
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
    
    # Display results
    print("\n" + "="*50)
    print("ðŸŒ½ PREDICTION RESULTS")
    print("="*50)
    print(f"ðŸ“Œ Predicted Disease: {predicted_class}")
    print("\nðŸ“Š Class Probabilities:")
    for i, prob in enumerate(probabilities):
        percentage = prob * 100
        bar = "â–ˆ" * int(percentage/5) + "â–‘" * (20 - int(percentage/5))
        print(f"  {classes[i]:15s}: {bar} {percentage:.1f}%")
    print("="*50)
    
    return predicted_class, probabilities

if __name__ == "__main__":
    # Example usage
    image_path = input("Enter path to maize leaf image: ")
    model_path = "models/maize_disease_classifier.pkl"
    
    predict_single_image(image_path, model_path)