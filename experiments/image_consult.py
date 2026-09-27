import json
from pathlib import Path

import numpy as np
from keras_core.models import load_model
from keras_core.applications.mobilenet_v2 import preprocess_input
from keras_core.utils import img_to_array, load_img

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "notebooks" / "best_transfer_model.keras"
ROOT_IMAGES = BASE_DIR / "real_pics"
OUTPUT_JSON_PATH = BASE_DIR / "image_consult_results.json"
IMG_SIZE = 224  # tamaño de las imagenes usadas en el entrenamiento

# Diccionario índice → clase obtenido en el EDA
index_to_class = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash"
}

# Cargar y preprocesar la foto
def prepare_image(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_single_image(image_path):
    processed_img = prepare_image(image_path)
    predictions = model.predict(processed_img, verbose=0)
    predicted_class_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    predicted_class_name = index_to_class.get(predicted_class_index, str(predicted_class_index))
    return predicted_class_name, confidence


model = load_model(str(MODEL_PATH))

results_payload = {
    "root_images": "./experiments/real_pics",
    "test_cases": []
}

if ROOT_IMAGES.exists():
    image_files = sorted(ROOT_IMAGES.iterdir())
    for image_path in image_files:
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue

        class_predicted, accuracy = predict_single_image(str(image_path))
        results_payload["test_cases"].append(
            {
                image_path.name: {
                    "class_predicted": class_predicted,
                    "accuracy": accuracy,
                    "human_verdict": ""
                }
            }
        )

with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as json_file:
    json.dump(results_payload, json_file, indent=4, ensure_ascii=False)
    json_file.write("\n")

print(f"Predicciones guardadas en: {OUTPUT_JSON_PATH}")