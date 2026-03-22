"""
Feature extraction module for maize leaf images
"""
import cv2
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extracts numerical features from images"""
    
    def __init__(self):
        pass
    
    def extract_features(self, images):
        """Extract features from a list of images"""
        features = []
        
        for img in tqdm(images, desc="Extracting features"):
            img_uint8 = (img * 255).astype(np.uint8)
            img_features = self._extract_single_image_features(img_uint8)
            features.append(img_features)
        
        return np.array(features)
    
    def _extract_single_image_features(self, img):
        """Extract features from a single image"""
        features = []
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # RGB features
        for ch in range(3):
            features.extend([
                np.mean(img[:,:,ch]),
                np.std(img[:,:,ch]),
                np.percentile(img[:,:,ch], 25),
                np.percentile(img[:,:,ch], 75)
            ])
        
        # HSV features
        for ch in range(3):
            features.extend([
                np.mean(hsv[:,:,ch] / 255.0),
                np.std(hsv[:,:,ch] / 255.0),
                np.percentile(hsv[:,:,ch] / 255.0, 25),
                np.percentile(hsv[:,:,ch] / 255.0, 75)
            ])
        
        # LAB features
        for ch in range(3):
            features.extend([
                np.mean(lab[:,:,ch] / 255.0),
                np.std(lab[:,:,ch] / 255.0),
                np.percentile(lab[:,:,ch] / 255.0, 25),
                np.percentile(lab[:,:,ch] / 255.0, 75)
            ])
        
        # Texture features
        features.extend([
            np.mean(gray), np.std(gray), np.var(gray),
            np.percentile(gray, 10), np.percentile(gray, 90)
        ])
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features.extend([
            np.mean(edges),
            np.sum(edges > 0) / edges.size
        ])
        
        return features
