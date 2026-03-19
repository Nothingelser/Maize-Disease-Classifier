"""
Module for preprocessing maize leaf images
"""
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt


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
            'Blight': 'Blight',
            'Common_Rust': 'Maize_Rust',
            'Gray_Leaf_Spot': 'Gray_Leaf_Spot',
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
        
        print(f"Scanning folder: {data_path}")
        
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
        
        print(f"Loaded {len(images)} images")
        print(f"Labels found: {set(labels)}")
        
        return np.array(images), np.array(labels)
    
    def _extract_label_from_folder(self, folder_name):
        """
        Extract disease label from folder name
        """
        for key, value in self.category_mapping.items():
            if key in folder_name:
                return value
        return None
    
    def _load_and_preprocess_image(self, img_path):
        """
        Load a single image and apply preprocessing
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
            print(f"Error processing {img_path}: {e}")
            return None
    
    def extract_features(self, images):
        """
        Extract numerical features from images for Random Forest
        
        Args:
            images: numpy array of images
            
        Returns:
            features: 2D array of features for each image
        """
        features = []
        
        for img in tqdm(images, desc="Extracting features"):
            # Convert back to uint8 for OpenCV operations
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            
            # Color features
            color_features = []
            
            # For each color space (RGB, HSV, LAB)
            for color_space, name in [(img, 'RGB'), (hsv/255.0, 'HSV'), (lab/255.0, 'LAB')]:
                for channel in range(3):
                    channel_data = color_space[:, :, channel]
                    color_features.extend([
                        np.mean(channel_data),
                        np.std(channel_data),
                        np.percentile(channel_data, 25),
                        np.percentile(channel_data, 75)
                    ])
            
            # Texture features from grayscale
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            texture_features = [
                np.mean(gray),
                np.std(gray),
                np.var(gray),
                np.percentile(gray, 10),
                np.percentile(gray, 90)
            ]
            
            # Edge features (to detect spots/lesions)
            edges = cv2.Canny(gray, 50, 150)
            edge_features = [
                np.mean(edges),
                np.sum(edges > 0) / edges.size  # edge density
            ]
            
            # Combine all features
            all_features = color_features + texture_features + edge_features
            features.append(all_features)
        
        return np.array(features)
    
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
        print("Step 1: Loading images...")
        images, labels = self.load_images_from_folder(data_path)
        
        # Extract features
        print("\nStep 2: Extracting features...")
        X = self.extract_features(images)
        
        # Encode labels
        print("\nStep 3: Encoding labels...")
        y = self.label_encoder.fit_transform(labels)
        
        # Split dataset
        print("\nStep 4: Splitting into train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n✅ Dataset prepared!")
        print(f"Training set: {X_train.shape[0]} images")
        print(f"Testing set: {X_test.shape[0]} images")
        print(f"Classes: {self.label_encoder.classes_}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'classes': self.label_encoder.classes_
        }