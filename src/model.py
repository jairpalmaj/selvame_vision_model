from keras_core import layers, models
from keras_core.applications import MobileNetV2
from keras_core.callbacks import EarlyStopping
from typing import Any, Dict, List, Tuple

def transfer_learning(config_yaml:Dict[str, Dict[str,Any]], loaders:Tuple[Any, Any, Any, List[str]]) -> models:
    IMG_SIZE = config_yaml['data']['img_size']
    train_dataset = loaders[0]
    val_dataset = loaders[1]
    class_names = loaders[3]
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(len(class_names), activation='softmax')(x)

    transfered_model = models.Model(inputs, outputs)
    transfered_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    transfered_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        shuffle=False,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )

    return transfered_model

def train_model():
    pass