"""
Train one Random Forest model per crop and save crop-specific model artifacts.
"""
import argparse
import os
import sys
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add project root to import src modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import LeafPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'raw'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'


def _choose_balance_strategy(crop_name, balance_strategy):
    if balance_strategy != 'auto':
        return balance_strategy
    return 'none'


def _rebalance_training_data(X_train, y_train, strategy, random_state=42):
    if strategy == 'none':
        return X_train, y_train

    rng = np.random.default_rng(int(random_state))
    classes, counts = np.unique(y_train, return_counts=True)

    if len(classes) < 2:
        return X_train, y_train

    if strategy == 'undersample':
        target_count = int(np.min(counts))
        selected_indices = []
        for class_id in classes:
            class_indices = np.where(y_train == class_id)[0]
            sampled = rng.choice(class_indices, size=target_count, replace=False)
            selected_indices.append(sampled)
        indices = np.concatenate(selected_indices)
    elif strategy == 'oversample':
        target_count = int(np.max(counts))
        selected_indices = []
        for class_id in classes:
            class_indices = np.where(y_train == class_id)[0]
            sampled = rng.choice(class_indices, size=target_count, replace=True)
            selected_indices.append(sampled)
        indices = np.concatenate(selected_indices)
    else:
        raise ValueError(f"Unsupported balance strategy: {strategy}")

    rng.shuffle(indices)
    return X_train[indices], y_train[indices]


def save_class_metadata(classes, output_path):
    payload = {'classes': [str(class_name) for class_name in classes]}
    with open(output_path, 'w', encoding='utf-8') as metadata_file:
        json.dump(payload, metadata_file, indent=2)


def load_crop_images(preprocessor, crop, max_images_per_class=None):
    """Load images for a single crop using folder-prefix filtering."""
    crop_key = crop.strip().lower().replace('-', '_').replace(' ', '_')
    images = []
    labels = []
    per_class_counts = {}

    for class_dir in sorted(DATA_PATH.iterdir()):
        if not class_dir.is_dir():
            continue

        folder_name = class_dir.name
        if '___' not in folder_name:
            continue

        prefix, _ = folder_name.split('___', 1)
        class_crop = prefix.strip().lower().replace('-', '_').replace(' ', '_')
        if class_crop != crop_key:
            continue

        class_label = preprocessor._extract_label_from_folder(folder_name)
        image_files = [
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        ]

        if max_images_per_class:
            image_files = image_files[:max_images_per_class]

        for image_path in tqdm(image_files, desc=f"Loading {folder_name}"):
            image = preprocessor._load_and_preprocess_image(str(image_path))
            if image is None:
                continue
            images.append(image)
            labels.append(class_label)
            per_class_counts[class_label] = per_class_counts.get(class_label, 0) + 1

    if not images:
        return None

    X_images = np.array(images)
    labels_encoded = preprocessor.label_encoder.fit_transform(np.array(labels))
    class_names = list(preprocessor.label_encoder.classes_)
    X_features = preprocessor.extract_features(X_images)

    return {
        'X': X_features,
        'y': labels_encoded,
        'classes': class_names,
        'per_class_counts': per_class_counts,
    }


def train_per_crop_models(crop=None, max_images_per_class=None, n_estimators=200, balance_strategy='auto'):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    target_crops = [crop] if crop else ['maize', 'cassava', 'rice', 'tomato', 'potato', 'pepper_bell']

    training_summary = []

    for crop_name in target_crops:
        preprocessor = LeafPreprocessor(img_size=(128, 128))
        crop_data = load_crop_images(
            preprocessor,
            crop=crop_name,
            max_images_per_class=max_images_per_class,
        )
        if not crop_data:
            continue

        X = crop_data['X']
        y = crop_data['y']
        classes = crop_data['classes']

        unique_labels = sorted(set(int(x) for x in y))
        if len(unique_labels) < 2 or len(y) < 20:
            continue

        X_crop_train, X_crop_test, y_crop_train, y_crop_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        effective_balance_strategy = _choose_balance_strategy(crop_name, balance_strategy)
        X_crop_train_balanced, y_crop_train_balanced = _rebalance_training_data(
            X_crop_train,
            y_crop_train,
            strategy=effective_balance_strategy,
            random_state=42,
        )

        train_class_counts = {
            classes[int(class_id)]: int(class_count)
            for class_id, class_count in zip(*np.unique(y_crop_train_balanced, return_counts=True))
        }

        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_crop_train_balanced, y_crop_train_balanced)

        predictions = model.predict(X_crop_test)
        accuracy = float(accuracy_score(y_crop_test, predictions))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_crop_test,
            predictions,
            labels=np.arange(len(classes)),
            zero_division=0,
        )
        macro_f1 = float(np.mean(f1))
        per_class_metrics = {
            classes[idx]: {
                'precision': float(precision[idx]),
                'recall': float(recall[idx]),
                'f1': float(f1[idx]),
            }
            for idx in range(len(classes))
        }

        model_file = MODELS_DIR / f'maize_disease_classifier_{crop_name}.pkl'
        labels_file = MODELS_DIR / f'class_labels_{crop_name}.json'

        joblib.dump(model, model_file)
        save_class_metadata(classes, labels_file)

        training_summary.append({
            'crop': crop_name,
            'train_samples': int(X_crop_train.shape[0]),
            'train_samples_effective': int(X_crop_train_balanced.shape[0]),
            'test_samples': int(X_crop_test.shape[0]),
            'classes': classes,
            'per_class_counts': crop_data['per_class_counts'],
            'train_class_counts_effective': train_class_counts,
            'balance_strategy': effective_balance_strategy,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class_metrics': per_class_metrics,
            'model_path': str(model_file),
            'labels_path': str(labels_file),
        })

        print(
            f"[{crop_name}] accuracy={accuracy:.3f} | macro_f1={macro_f1:.3f} "
            f"| balance={effective_balance_strategy} | model={model_file.name}"
        )

    summary_file = REPORTS_DIR / 'per_crop_model_training_summary.json'
    if crop:
        summary_file = REPORTS_DIR / f'per_crop_model_training_summary_{crop}.json'

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(training_summary, f, indent=2)

    print(f"Saved summary: {summary_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train per-crop models')
    parser.add_argument('--crop', type=str, default=None, help='Optional crop key: maize, cassava, rice, tomato, potato, pepper_bell')
    parser.add_argument('--max-images-per-class', type=int, default=None, help='Optional cap for faster training in constrained environments')
    parser.add_argument('--n-estimators', type=int, default=200, help='Number of trees for RandomForest')
    parser.add_argument('--balance-strategy', type=str, default='auto', choices=['auto', 'none', 'undersample', 'oversample'], help='Class balancing strategy for training data')
    args = parser.parse_args()

    train_per_crop_models(
        crop=args.crop,
        max_images_per_class=args.max_images_per_class,
        n_estimators=args.n_estimators,
        balance_strategy=args.balance_strategy,
    )
