#!/usr/bin/env python3
"""Train and evaluate baseline and augmented CNNs for the NIELIT AIDA project.

Expected dataset layout:
    dataset/
      train/<class_name>/*
      validation/<class_name>/*
      test/<class_name>/*

This is a consolidated, reproducible training script based on the project
architecture and recorded experiment settings. It is not an export of the
original interactive Colab notebook. Images are normalised in the input
pipeline to match the Streamlit app's 0-to-1 inference preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score


CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
IMAGE_SIZE = (224, 224)
NUM_CLASSES = len(CLASS_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset",
        help="Directory containing train, validation and test folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "training_outputs",
        help="Directory for models, histories and evaluation results.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["baseline", "augmented"],
        default=["baseline", "augmented"],
        help="Select one or both experiments.",
    )
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def check_dataset(dataset_dir: Path) -> None:
    required_splits = ["train", "validation", "test"]
    missing = [split for split in required_splits if not (dataset_dir / split).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset split folder(s): {missing}. Expected {dataset_dir}/"
            "train, validation and test, each containing the three class folders."
        )

    for split in required_splits:
        split_dir = dataset_dir / split
        missing_classes = [name for name in CLASS_NAMES if not (split_dir / name).is_dir()]
        if missing_classes:
            raise FileNotFoundError(
                f"{split_dir} is missing class folder(s): {missing_classes}."
            )


def load_datasets(dataset_dir: Path, batch_size: int, seed: int):
    """Load split folders with a fixed class order and 224x224 RGB resizing."""
    common: dict[str, Any] = {
        "labels": "inferred",
        "label_mode": "int",
        "class_names": CLASS_NAMES,
        "color_mode": "rgb",
        "batch_size": batch_size,
        "image_size": IMAGE_SIZE,
        "interpolation": "bilinear",
    }

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "train", shuffle=True, seed=seed, **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "validation", shuffle=False, **common
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "test", shuffle=False, **common
    )

    # Normalise dataset images here (rather than inside the CNN). This keeps
    # trained-model input in [0, 1], matching app.py preprocessing and avoids
    # dividing by 255 twice during Streamlit inference.
    def normalise_batch(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels

    train_ds = train_ds.map(normalise_batch, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(normalise_batch, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(normalise_batch, num_parallel_calls=tf.data.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_deterministic = True
    train_ds = train_ds.with_options(options).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.with_options(options).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.with_options(options).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds


def build_model(experiment: str) -> tf.keras.Model:
    """Build the documented four-block CNN; augmentation is training-only."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    # Dataset batches are already float32 and scaled to [0, 1].
    x = inputs

    if experiment == "augmented":
        x = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.10),
                tf.keras.layers.RandomZoom(0.10),
                tf.keras.layers.RandomContrast(0.10),
            ],
            name="data_augmentation",
        )(x)

    # Recorded experiments differed in padding: baseline used Keras default
    # ('valid'); augmented explicitly used 'same'. The report notes this.
    padding = "valid" if experiment == "baseline" else "same"
    for filters in (32, 64, 128, 256):
        x = tf.keras.layers.Conv2D(
            filters, kernel_size=(3, 3), activation="relu", padding=padding
        )(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return tf.keras.Model(
        inputs=inputs, outputs=outputs, name=f"{experiment}_bean_leaf_cnn"
    )


def save_history(history: tf.keras.callbacks.History, csv_path: Path) -> None:
    keys = list(history.history.keys())
    row_count = len(history.history[keys[0]]) if keys else 0
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch"] + keys)
        for index in range(row_count):
            writer.writerow(
                [index + 1] + [history.history[key][index] for key in keys]
            )


def evaluate_model(
    model: tf.keras.Model, test_ds, experiment: str, results_dir: Path
) -> dict[str, Any]:
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in test_ds:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).astype(int).tolist())

    test_loss, _ = model.evaluate(test_ds, verbose=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    with (
        results_dir / f"{experiment}_classification_report.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    np.savetxt(
        results_dir / f"{experiment}_confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d",
    )

    return {
        "experiment": experiment,
        "test_samples": len(y_true),
        "test_accuracy": accuracy,
        "test_loss": float(test_loss),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "class_order": CLASS_NAMES,
        "classification_report": report,
    }


def train_experiment(
    experiment: str,
    train_ds,
    val_ds,
    test_ds,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
) -> dict[str, Any]:
    models_dir = output_dir / "models"
    results_dir = output_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = models_dir / f"best_{experiment}_bean_leaf_cnn.keras"
    final_filename = (
        "final_bean_leaf_disease_cnn.keras"
        if experiment == "baseline"
        else "final_augmented_bean_leaf_cnn.keras"
    )
    final_model_path = models_dir / final_filename

    model = build_model(experiment)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n{'=' * 72}\nTraining: {experiment}\n{'=' * 72}")
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    save_history(history, results_dir / f"{experiment}_history.csv")

    # Evaluate the best validation-loss checkpoint, not the final epoch.
    best_model = tf.keras.models.load_model(best_model_path)
    best_model.save(final_model_path)
    metrics = evaluate_model(best_model, test_ds, experiment, results_dir)
    metrics["best_validation_loss"] = float(min(history.history["val_loss"]))
    metrics["epochs_completed"] = len(history.history["loss"])

    with (results_dir / f"{experiment}_test_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    print(
        f"\n{experiment} test accuracy: {metrics['test_accuracy']:.4f} | "
        f"macro F1: {metrics['macro_f1']:.4f} | "
        f"weighted F1: {metrics['weighted_f1']:.4f}"
    )
    print(f"Best model: {final_model_path}")
    return metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    set_reproducibility(args.seed)
    check_dataset(args.dataset_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = load_datasets(
        args.dataset_dir, args.batch_size, args.seed
    )
    all_metrics: dict[str, Any] = {}

    for experiment in args.experiments:
        tf.keras.backend.clear_session()
        all_metrics[experiment] = train_experiment(
            experiment,
            train_ds,
            val_ds,
            test_ds,
            args.output_dir,
            args.epochs,
            args.learning_rate,
        )

    summary = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "image_size": list(IMAGE_SIZE),
        "class_order": CLASS_NAMES,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "adam_learning_rate": args.learning_rate,
        "seed": args.seed,
        "experiments": all_metrics,
        "notes": [
            "This is a consolidated reproduction script, not an export of the original interactive Colab notebook.",
            "Dataset images are normalised to float32 [0, 1] before model input to match app.py preprocessing.",
            "Baseline Conv2D padding is 'valid' (Keras default); augmented Conv2D padding is 'same', matching the documented implementations.",
            "Augmentation layers are active during training and inactive during inference.",
            "Fresh runs can produce different metrics because model initialisation and accelerator execution may vary.",
        ],
    }
    with (args.output_dir / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"\nTraining complete. Outputs saved under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
