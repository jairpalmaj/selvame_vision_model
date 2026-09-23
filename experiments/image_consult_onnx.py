import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


BASE_DIR = Path(__file__).resolve().parent

# Update these paths manually before running the script.
MODEL_PATH = BASE_DIR.parent / "data" / "MobilenetV3" / "best_model_v3_large-20260915.onnx"
ROOT_IMAGES = BASE_DIR / "real_pics"
OUTPUT_JSON_PATH = BASE_DIR / "image_consult_results_onnx_20260915.json"

IMG_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Diccionario índice -> clase obtenido en el EDA.
index_to_class = {
    0: "paper",
    1: "metal",
    2: "cardboard",
    3: "trash",
    4: "glass",
    5: "plastic",
}


def prepare_image(image_path: Path, input_shape) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = preprocess_input(image_array)

    # The exported MobileNetV3 model uses NCHW input; support NHWC as well.
    if len(input_shape) == 4 and input_shape[1] == 3:
        image_array = np.transpose(image_array, (2, 0, 1))

    return np.expand_dims(image_array, axis=0).astype(np.float32)


def predict_single_image(session, image_path: Path) -> tuple[str, float]:
    input_metadata = session.get_inputs()[0]
    processed_image = prepare_image(image_path, input_metadata.shape)
    output_name = session.get_outputs()[0].name
    predictions = session.run(
        [output_name],
        {input_metadata.name: processed_image},
    )[0]

    predicted_class_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    predicted_class_name = index_to_class.get(
        predicted_class_index,
        str(predicted_class_index),
    )
    return predicted_class_name, confidence


session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

results_payload = {
    "root_images": str(ROOT_IMAGES),
    "test_cases": [],
}

if ROOT_IMAGES.exists():
    image_files = sorted(
        image_path
        for image_path in ROOT_IMAGES.iterdir()
        if image_path.is_file()
        and image_path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in image_files:
        class_predicted, confidence = predict_single_image(session, image_path)
        results_payload["test_cases"].append(
            {
                image_path.name: {
                    "class_predicted": class_predicted,
                    "accuracy": confidence,
                    "human_verdict": "",
                }
            }
        )

with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as json_file:
    json.dump(results_payload, json_file, indent=4, ensure_ascii=False)
    json_file.write("\n")

print(f"Predicciones guardadas en: {OUTPUT_JSON_PATH}")