"""
Data preprocessing module for plant leaf images.
Handles loading, resizing, and preprocessing of image datasets.
"""
import os
import re
import cv2
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class LeafPreprocessor:
    """
    Handles loading and preprocessing of leaf images.
    """
    
    def __init__(self, img_size=(128, 128)):
        """
        Initialize preprocessor with desired image size
        
        Args:
            img_size: tuple (height, width) to resize images to
        """
        self.img_size = img_size
        self.label_encoder = LabelEncoder()
        
        # Legacy aliases to keep older maize-only folder names working.
        self.category_mapping = {
            'Cercospora_leaf_spot': 'Maize___Gray_Leaf_Spot',
            'Gray_leaf_spot': 'Maize___Gray_Leaf_Spot',
            'Common_rust': 'Maize___Rust',
            'Northern_Leaf_Blight': 'Maize___Blight',
            'healthy': 'Maize___Healthy',
            'Blight': 'Maize___Blight',
            'Gray_Leaf_Spot': 'Maize___Gray_Leaf_Spot',
            'Maize_Rust': 'Maize___Rust',
            'Healthy': 'Maize___Healthy'
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
        
        data_path_abs = os.path.abspath(data_path)

        # Walk through all folders
        for root, dirs, files in os.walk(data_path):
            if os.path.abspath(root) == data_path_abs:
                # Skip the dataset root itself; labels are inferred from child folders.
                continue

            for file in tqdm(files, desc="Loading images"):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Get the folder name (contains disease info)
                    folder_name = os.path.basename(root)
                    
                    # Determine label from folder name
                    label = self._extract_label_from_folder(folder_name)
                    
                    # Skip folders that cannot be interpreted as class labels
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
        Extract class label from folder name.

        Supported examples:
            - Tomato___Early_Blight
            - Potato___Late_Blight
            - Corn_(maize)___healthy
            - Blight (legacy maize-only folder)
        """
        if not folder_name:
            return None

        folder_name = folder_name.strip()
        if not folder_name:
            return None

        for key, value in self.category_mapping.items():
            if key.lower() == folder_name.lower():
                return value

        if '___' in folder_name:
            parts = [part for part in folder_name.split('___') if part]
            if len(parts) >= 2:
                crop = self._normalize_label_token(parts[0])
                disease = self._normalize_label_token('_'.join(parts[1:]))
                if crop and disease:
                    return f"{crop}___{disease}"

        return self._normalize_label_token(folder_name)

    @staticmethod
    def _normalize_label_token(token):
        """Normalize label tokens into a filesystem-safe consistent format."""
        normalized = token.replace(' ', '_').replace('-', '_')
        normalized = re.sub(r'[^A-Za-z0-9_]+', '_', normalized)
        normalized = re.sub(r'_+', '_', normalized).strip('_')
        return normalized or None
    
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

    def _random_augment_image(self, image, rng):
        """Apply one lightweight augmentation to an RGB float image in [0, 1]."""
        aug_type = int(rng.integers(0, 4))
        augmented = image.copy()

        if aug_type == 0:
            # Horizontal flip helps invariance to leaf orientation.
            augmented = np.fliplr(augmented)
        elif aug_type == 1:
            # Small rotation handles capture angle differences.
            angle = float(rng.uniform(-14.0, 14.0))
            height, width = augmented.shape[:2]
            center = (width / 2.0, height / 2.0)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(
                augmented,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        elif aug_type == 2:
            # Mild brightness jitter.
            factor = float(rng.uniform(0.82, 1.18))
            augmented = augmented * factor
        else:
            # Mild sensor-like gaussian noise.
            noise = rng.normal(0.0, 0.02, augmented.shape).astype(np.float32)
            augmented = augmented + noise

        return np.clip(augmented, 0.0, 1.0).astype(np.float32)

    def augment_minority_classes(
        self,
        images_train,
        y_train,
        min_samples_per_class=700,
        max_aug_per_source=6,
        random_state=42,
    ):
        """
        Add synthetic train samples for classes below a target count.

        Args:
            images_train: train split images
            y_train: encoded train labels
            min_samples_per_class: target train samples for each class
            max_aug_per_source: cap synthetic copies generated from one source image
            random_state: reproducibility seed

        Returns:
            Tuple of (augmented_images_train, augmented_y_train, augmentation_report)
        """
        y_train = np.asarray(y_train)
        rng = np.random.default_rng(random_state)

        class_ids, class_counts = np.unique(y_train, return_counts=True)
        extra_images = []
        extra_labels = []
        report: Dict[str, Dict[str, Any]] = {}

        for class_id, original_count in zip(class_ids, class_counts):
            class_name = self.label_encoder.inverse_transform([class_id])[0]
            original_count = int(original_count)

            if original_count >= int(min_samples_per_class):
                report[class_name] = {
                    'original': original_count,
                    'target': int(min_samples_per_class),
                    'generated': 0,
                    'final': original_count,
                    'status': 'sufficient',
                }
                continue

            class_positions = np.where(y_train == class_id)[0]
            if class_positions.size == 0:
                report[class_name] = {
                    'original': 0,
                    'target': int(min_samples_per_class),
                    'generated': 0,
                    'final': 0,
                    'status': 'empty',
                }
                continue

            required = int(min_samples_per_class) - original_count
            per_source_cap = max(1, int(max_aug_per_source))
            generated = 0

            while generated < required:
                source_index = int(rng.choice(class_positions))
                source_image = images_train[source_index]
                extra_images.append(self._random_augment_image(source_image, rng))
                extra_labels.append(class_id)
                generated += 1

                if generated >= class_positions.size * per_source_cap:
                    break

            report[class_name] = {
                'original': original_count,
                'target': int(min_samples_per_class),
                'generated': generated,
                'final': original_count + generated,
                'status': 'augmented' if generated > 0 else 'capped',
            }

        if extra_images:
            images_aug = np.concatenate([images_train, np.asarray(extra_images, dtype=np.float32)], axis=0)
            y_aug = np.concatenate([y_train, np.asarray(extra_labels, dtype=np.int64)], axis=0)
        else:
            images_aug = images_train
            y_aug = y_train

        shuffle_idx = rng.permutation(len(y_aug))
        return images_aug[shuffle_idx], y_aug[shuffle_idx], report
    
    def prepare_dataset(
        self,
        data_path,
        test_size=0.3,
        augment_train=False,
        min_samples_per_class=700,
        max_aug_per_source=6,
        random_state=42,
    ):
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
        
        # Encode labels
        logger.info("\nStep 2: Encoding labels...")
        y = self.label_encoder.fit_transform(labels)
        
        # Split dataset
        logger.info("\nStep 3: Splitting into train/test...")
        images_train, images_test, y_train, y_test = train_test_split(
            images, y, test_size=test_size, random_state=random_state, stratify=y
        )

        augmentation_report = {}
        if augment_train:
            logger.info("\nStep 4: Augmenting minority classes in training split...")
            images_train, y_train, augmentation_report = self.augment_minority_classes(
                images_train,
                y_train,
                min_samples_per_class=min_samples_per_class,
                max_aug_per_source=max_aug_per_source,
                random_state=random_state,
            )
            logger.info("Augmentation complete; %s classes evaluated", len(augmentation_report))

        # Extract features after split/augmentation so test data remains untouched.
        logger.info("\nStep 5: Extracting features...")
        X_train = self.extract_features(images_train)
        X_test = self.extract_features(images_test)
        
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
            'feature_names': self._get_feature_names(),
            'augmentation_report': augmentation_report,
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


# Backward compatibility alias used across the existing codebase.
MaizeLeafPreprocessor = LeafPreprocessor