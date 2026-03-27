"""Train and evaluate the plant disease classifier from a local dataset.

This script adds a practical CLI around the existing training pipeline:
- optional migration of legacy maize-only class folders
- dataset validation and class-level image counts
- model training/evaluation with metadata output
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import sys
import numpy as np


# Make project imports work when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from src.data_preprocessing import LeafPreprocessor
from src.evaluation import ModelEvaluator
from src.model_training import MaizeDiseaseClassifier


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LEGACY_FOLDER_MAP = {
	"Blight": "Maize___Blight",
	"Common_Rust": "Maize___Rust",
	"Gray_Leaf_Spot": "Maize___Gray_Leaf_Spot",
	"Healthy": "Maize___Healthy",
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train the plant disease classifier")
	parser.add_argument("--data-dir", default="data/raw", help="Dataset root directory")
	parser.add_argument("--model-path", default="models/maize_disease_classifier.pkl", help="Output model path")
	parser.add_argument("--labels-path", default="models/class_labels.json", help="Output labels metadata path")
	parser.add_argument("--test-size", type=float, default=0.3, help="Test split ratio")
	parser.add_argument("--img-size", type=int, default=128, help="Square image size in pixels")
	parser.add_argument("--n-estimators", type=int, default=100, help="RandomForest tree count")
	parser.add_argument(
		"--augment-weak-classes",
		action="store_true",
		help="Apply lightweight train-time augmentation for underrepresented classes"
	)
	parser.add_argument(
		"--min-train-samples-per-class",
		type=int,
		default=700,
		help="Minimum desired samples per class in training split when augmentation is enabled"
	)
	parser.add_argument(
		"--max-augment-per-image",
		type=int,
		default=6,
		help="Maximum synthetic copies generated from one source image"
	)
	parser.add_argument(
		"--class-weight",
		choices=["none", "balanced", "balanced_subsample"],
		default="balanced_subsample",
		help="Class weighting strategy for imbalanced classes"
	)
	parser.add_argument("--optimize", action="store_true", help="Run GridSearchCV optimization")
	parser.add_argument(
		"--balance-cassava",
		action="store_true",
		help="Resample cassava classes in the training split to reduce dominance of Mosaic Disease"
	)
	parser.add_argument(
		"--cassava-target",
		type=int,
		default=4000,
		help="Target sample count for non-mosaic cassava classes in training split"
	)
	parser.add_argument(
		"--cassava-mosaic-target",
		type=int,
		default=5000,
		help="Target sample count for Cassava___Mosaic_Disease in training split"
	)
	parser.add_argument("--migrate-legacy", action="store_true", help="Rename legacy maize folders to Crop___Disease format")
	parser.add_argument("--min-images-per-class", type=int, default=20, help="Minimum images expected per class")
	parser.add_argument("--inspect-only", action="store_true", help="Validate and summarize dataset without training")
	parser.add_argument("--report-path", default="reports/dataset_summary.json", help="Dataset summary JSON output")
	return parser.parse_args()


def rebalance_cassava_training_split(
	X_train,
	y_train,
	class_names,
	target_non_mosaic,
	target_mosaic,
	random_state=42,
):
	"""Resample cassava classes in the training split to desired targets."""
	rng = np.random.default_rng(random_state)
	y_train = np.asarray(y_train)

	selected_indices = []
	resample_report = {}

	for class_index, class_name in enumerate(class_names):
		class_positions = np.where(y_train == class_index)[0]
		class_count = int(class_positions.size)

		if class_name.startswith("Cassava___"):
			target = target_mosaic if class_name == "Cassava___Mosaic_Disease" else target_non_mosaic
			if class_count == 0:
				resample_report[class_name] = {
					"original": 0,
					"target": int(target),
					"final": 0,
					"strategy": "skip-empty",
				}
				continue

			replace = class_count < target
			chosen = rng.choice(class_positions, size=target, replace=replace)
			selected_indices.extend(chosen.tolist())
			resample_report[class_name] = {
				"original": class_count,
				"target": int(target),
				"final": int(target),
				"strategy": "upsample" if replace else "downsample",
			}
		else:
			selected_indices.extend(class_positions.tolist())

	selected_indices = np.array(selected_indices, dtype=np.int64)
	rng.shuffle(selected_indices)

	X_balanced = X_train[selected_indices]
	y_balanced = y_train[selected_indices]

	return X_balanced, y_balanced, resample_report


def count_images(folder: Path) -> int:
	return sum(1 for file in folder.rglob("*") if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS)


def migrate_legacy_folders(data_dir: Path) -> List[Tuple[str, str]]:
	renamed: List[Tuple[str, str]] = []

	for old_name, new_name in LEGACY_FOLDER_MAP.items():
		old_path = data_dir / old_name
		new_path = data_dir / new_name

		if not old_path.exists() or not old_path.is_dir():
			continue

		if new_path.exists():
			print(f"[WARN] Skip renaming '{old_name}' because '{new_name}' already exists.")
			continue

		old_path.rename(new_path)
		renamed.append((old_name, new_name))

	return renamed


def inspect_dataset(data_dir: Path, min_images_per_class: int) -> Tuple[Dict[str, int], List[str]]:
	class_counts: Dict[str, int] = {}
	warnings: List[str] = []

	if not data_dir.exists() or not data_dir.is_dir():
		raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

	class_dirs = [entry for entry in sorted(data_dir.iterdir()) if entry.is_dir()]
	if not class_dirs:
		raise ValueError(f"No class folders found in: {data_dir}")

	for class_dir in class_dirs:
		class_name = class_dir.name
		image_count = count_images(class_dir)
		class_counts[class_name] = image_count

		if "___" not in class_name:
			warnings.append(f"Class folder '{class_name}' is not in Crop___Disease format.")

		if image_count < min_images_per_class:
			warnings.append(
				f"Class '{class_name}' has only {image_count} images (minimum recommended: {min_images_per_class})."
			)

	if len(class_counts) < 2:
		raise ValueError("At least 2 class folders are required for training.")

	return class_counts, warnings


def save_class_metadata(classes, labels_path: Path) -> None:
	payload = {"classes": [str(name) for name in classes]}
	labels_path.parent.mkdir(parents=True, exist_ok=True)
	labels_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_dataset_summary(report_path: Path, class_counts: Dict[str, int], warnings: List[str]) -> None:
	summary = {
		"class_counts": class_counts,
		"num_classes": len(class_counts),
		"total_images": int(sum(class_counts.values())),
		"warnings": warnings,
	}
	report_path.parent.mkdir(parents=True, exist_ok=True)
	report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_class_summary(class_counts: Dict[str, int]) -> None:
	"""Print a clean, deterministic class summary once."""
	normalized_counts: Dict[str, int] = {}
	duplicates_suppressed = 0

	for class_name, image_count in class_counts.items():
		normalized_name = class_name.strip()
		if normalized_name in normalized_counts:
			normalized_counts[normalized_name] += image_count
			duplicates_suppressed += 1
		else:
			normalized_counts[normalized_name] = image_count

	total_images = int(sum(normalized_counts.values()))
	print(f"\nDetected {len(normalized_counts)} classes ({total_images} images total):")
	for class_name in sorted(normalized_counts):
		print(f"- {class_name}: {normalized_counts[class_name]} images")

	if duplicates_suppressed > 0:
		print(f"- [info] Suppressed {duplicates_suppressed} duplicate class name entries in console output.")


def main() -> int:
	args = parse_args()

	data_dir = Path(args.data_dir)
	model_path = Path(args.model_path)
	labels_path = Path(args.labels_path)
	report_path = Path(args.report_path)
	img_size = (args.img_size, args.img_size)
	class_weight = None if args.class_weight == "none" else args.class_weight

	print("=" * 60)
	print("Plant Disease Training Pipeline")
	print("=" * 60)
	print(f"Class weighting: {class_weight or 'none'}")
	if args.balance_cassava:
		print(f"Cassava rebalance: enabled (target={args.cassava_target}, mosaic_target={args.cassava_mosaic_target})")
	if args.augment_weak_classes:
		print(
			"Minority augmentation: enabled "
			f"(min_per_class={args.min_train_samples_per_class}, max_per_image={args.max_augment_per_image})"
		)

	if args.migrate_legacy:
		renamed = migrate_legacy_folders(data_dir)
		if renamed:
			print("\nRenamed legacy folders:")
			for old_name, new_name in renamed:
				print(f"- {old_name} -> {new_name}")
		else:
			print("\nNo legacy folders were renamed.")

	class_counts, warnings = inspect_dataset(data_dir, args.min_images_per_class)
	save_dataset_summary(report_path, class_counts, warnings)
	print_class_summary(class_counts)

	if warnings:
		print("\nDataset warnings:")
		for warning in warnings:
			print(f"- {warning}")

	if args.inspect_only:
		print("\nInspect-only mode: training skipped.")
		print(f"- Dataset summary: {report_path}")
		return 0

	preprocessor = LeafPreprocessor(img_size=img_size)
	data = preprocessor.prepare_dataset(
		str(data_dir),
		test_size=args.test_size,
		augment_train=args.augment_weak_classes,
		min_samples_per_class=args.min_train_samples_per_class,
		max_aug_per_source=args.max_augment_per_image,
	)

	if args.augment_weak_classes:
		augmentation_report = data.get("augmentation_report", {})
		if augmentation_report:
			aug_report_path = Path("reports/augmentation_report.json")
			aug_report_path.parent.mkdir(parents=True, exist_ok=True)
			aug_report_path.write_text(json.dumps(augmentation_report, indent=2), encoding="utf-8")
			print("\nAugmentation report:")
			for class_name, details in augmentation_report.items():
				if details.get("generated", 0) > 0:
					print(
						f"- {class_name}: {details['original']} -> {details['final']} "
						f"(+{details['generated']})"
					)
			print(f"- Saved report: {aug_report_path}")

	if args.balance_cassava:
		X_train_balanced, y_train_balanced, cassava_report = rebalance_cassava_training_split(
			data["X_train"],
			data["y_train"],
			data["classes"],
			target_non_mosaic=args.cassava_target,
			target_mosaic=args.cassava_mosaic_target,
		)
		print("\nCassava resampling report:")
		for class_name, details in cassava_report.items():
			print(
				f"- {class_name}: {details['original']} -> {details['final']} ({details['strategy']})"
			)

		balance_report_path = Path("reports/cassava_rebalance_report.json")
		balance_report_path.parent.mkdir(parents=True, exist_ok=True)
		balance_report_path.write_text(json.dumps(cassava_report, indent=2), encoding="utf-8")

		data["X_train"] = X_train_balanced
		data["y_train"] = y_train_balanced

	classifier = MaizeDiseaseClassifier(
		n_estimators=args.n_estimators,
		class_weight=class_weight
	)
	if args.optimize:
		classifier.optimize_hyperparameters(data["X_train"], data["y_train"])
	else:
		classifier.train(data["X_train"], data["y_train"])

	model_path.parent.mkdir(parents=True, exist_ok=True)
	classifier.save_model(str(model_path))
	save_class_metadata(data["classes"], labels_path)

	evaluator = ModelEvaluator(classifier, data["classes"])
	results = evaluator.evaluate(data["X_test"], data["y_test"])
	evaluator.print_metrics(results["metrics"])

	confusion_matrix_path = Path("reports/confusion_matrix.png")
	confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
	evaluator.plot_confusion_matrix(results["confusion_matrix"], save_path=str(confusion_matrix_path))

	print("\nTraining completed.")
	print(f"- Model: {model_path}")
	print(f"- Labels: {labels_path}")
	print(f"- Dataset summary: {report_path}")
	print(f"- Confusion matrix: {confusion_matrix_path}")
	if args.balance_cassava:
		print("- Cassava rebalance report: reports/cassava_rebalance_report.json")
	if args.augment_weak_classes:
		print("- Augmentation report: reports/augmentation_report.json")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())