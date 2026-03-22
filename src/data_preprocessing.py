"""
Data preprocessing module for maize leaf images
Handles loading, resizing, and basic preprocessing of images
"""
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class MaizeLeafPreprocessor:
    """
    Handles loading and preprocessing of maize leaf images
    """
    
    def __init__(self, img_size=(128, 128)):
        """
        Initialize preprocessor with desired image size
        
        Args:
            img_size: tuple (height, width) to resize images to
        """
        self.img_size = img_size
        self.label_encoder = LabelEncoder()
        
        # Map folder names to our category names
        self.category_mapping = {
            'Cercospora_leaf_spot': 'Gray_Leaf_Spot',
            'Gray_leaf_spot': 'Gray_Leaf_Spot',
            'Common_rust': 'Maize_Rust',
            'Northern_Leaf_Blight': 'Blight',
            'healthy': 'Healthy',
            'Blight': 'Blight',
            'Gray_Leaf_Spot': 'Gray_Leaf_Spot',
            'Maize_Rust': 'Maize_Rust',
            'Healthy': 'Healthy'
        }
    
    def load_images_from_folder(self, data_path):
        """
        Load all images from the dataset folder
        
        Args:
            data_path: path to the PlantVillage dataset folder
            
        Returns:
            images: numpy array of images
            labels: numpy array of corresponding labels
        """
        images = []
        labels = []
        
        logger.info(f"Scanning folder: {data_path}")
        
        # Walk through all folders
        for root, dirs, files in os.walk(data_path):
            for file in tqdm(files, desc="Loading images"):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Get the folder name (contains disease info)
                    folder_name = os.path.basename(root)
                    
                    # Determine label from folder name
                    label = self._extract_label_from_folder(folder_name)
                    
                    # Skip if not a maize leaf image
                    if label is None:
                        continue
                    
                    # Load and preprocess image
                    img_path = os.path.join(root, file)
                    img = self._load_and_preprocess_image(img_path)
                    
                    if img is not None:
                        images.append(img)
                        labels.append(label)
        
        logger.info(f"Loaded {len(images)} images")
        logger.info(f"Labels found: {set(labels)}")
        
        return np.array(images), np.array(labels)
    
    def _extract_label_from_folder(self, folder_name):
        """
        Extract disease label from folder name
        """
        for key, value in self.category_mapping.items():
            if key.lower() in folder_name.lower():
                return value
        return None
    
    def _load_and_preprocess_image(self, img_path):
        """
        Load a single image and apply preprocessing
        
        Args:
            img_path: path to image file
            
        Returns:
            preprocessed image as numpy array, or None if failed
        """
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            # Convert BGR to RGB (OpenCV loads as BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size
            img = cv2.resize(img, self.img_size)
            
            # Apply Gaussian blur to reduce noise
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            return None
    
    def extract_features(self, images):
        """
        Extract numerical features from images for Random Forest
        
        Args:
            images: numpy array of images
            
        Returns:
            features: 2D array of features for each image
        """
        # Import here to avoid circular imports
        from src.feature_extraction import FeatureExtractor
        extractor = FeatureExtractor()
        return extractor.extract_features(images)
    
    def prepare_dataset(self, data_path, test_size=0.3):
        """
        Complete pipeline to prepare dataset for training
        
        Args:
            data_path: path to dataset
            test_size: fraction of data to use for testing
            
        Returns:
            dictionary containing training and testing data
        """
        # Load images
        logger.info("Step 1: Loading images...")
        images, labels = self.load_images_from_folder(data_path)
        
        # Extract features
        logger.info("\nStep 2: Extracting features...")
        X = self.extract_features(images)
        
        # Encode labels
        logger.info("\nStep 3: Encoding labels...")
        y = self.label_encoder.fit_transform(labels)
        
        # Split dataset
        logger.info("\nStep 4: Splitting into train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"\nâœ… Dataset prepared!")
        logger.info(f"Training set: {X_train.shape[0]} images")
        logger.info(f"Testing set: {X_test.shape[0]} images")
        logger.info(f"Classes: {self.label_encoder.classes_}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'classes': self.label_encoder.classes_,
            'feature_names': self._get_feature_names()
        }
    
    def _get_feature_names(self):
        """
        Generate feature names for reference
        """
        feature_names = []
        
        # Color features (RGB, HSV, LAB)
        color_spaces = ['RGB', 'HSV', 'LAB']
        stats = ['mean', 'std', 'q25', 'q75']
        for cs in color_spaces:
            for ch in range(3):
                for stat in stats:
                    feature_names.append(f"{cs}_ch{ch}_{stat}")
        
        # Texture features
        texture_stats = ['mean', 'std', 'var', 'q10', 'q90']
        for stat in texture_stats:
            feature_names.append(f"texture_{stat}")
        
        # Edge features
        feature_names.extend(['edge_mean', 'edge_density'])
        
        return feature_names