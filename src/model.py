import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from keras_core import layers, models
from keras_core.applications import MobileNetV3Small
from keras_core.callbacks import EarlyStopping


def save_training_history(history: Dict[str, List[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)

    epochs = range(1, len(history["loss"]) + 1)
    figure, (accuracy_axis, loss_axis) = plt.subplots(1, 2, figsize=(12, 5))

    accuracy_axis.plot(epochs, history["accuracy"], label="Entrenamiento")
    accuracy_axis.plot(epochs, history["val_accuracy"], label="Validacion")
    accuracy_axis.set_title("Accuracy - MobileNetV3Small")
    accuracy_axis.set_xlabel("Epocas")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.legend()
    accuracy_axis.grid(True)

    loss_axis.plot(epochs, history["loss"], label="Entrenamiento")
    loss_axis.plot(epochs, history["val_loss"], label="Validacion")
    loss_axis.set_title("Loss - MobileNetV3Small")
    loss_axis.set_xlabel("Epocas")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend()
    loss_axis.grid(True)

    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=150)
    plt.close(figure)

def transfer_learning(
        config_yaml:Dict[str, Dict[str,Any]],
        loaders:Tuple[Any, Any, Any, List[str]],
        root_folder:Path
        ) -> models:
    IMG_SIZE = config_yaml['data']['img_size']
    train_dataset = loaders[0]
    val_dataset = loaders[1]
    class_names = loaders[3]
    base_model = MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(class_names), activation='softmax')(x)

    transfered_model = models.Model(inputs, outputs)
    transfered_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = transfered_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        shuffle=False,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )

    output_dir = root_folder / config_yaml.get("outputs", {}).get("checkpoint_path", "data")
    output_dir.mkdir(parents=True, exist_ok=True)
    history_data = {
        metric: [float(value) for value in values]
        for metric, values in history.history.items()
    }
    save_training_history(history_data, output_dir)
    transfered_model.save(output_dir / "mobilenetv3small_transfer_learning.keras")
    transfered_model.save_weights(output_dir / "mobilenetv3small_transfer_learning.weights.h5")

    return transfered_model

def train_model():
    pass