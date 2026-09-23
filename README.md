# Descripcion
Este repositorio contiene el desarrollo del modelo de vision, el analisis de los datos asi como data augmentation y metricas de desempeño.

# Diseño

El pipe de construccion del modelo esta pensado para que sea modular en scripts de Python, evitando utilizar NoteBooks
Facilitando asi la colaboracion.

La estructura del repositorio que se propone es la siguiente:

vision_model/
├── config.yaml                 # Hiperparámetros, rutas y configuraciones fijas
├── requirements.txt            # Dependencias del proyecto (torch, torchvision, scikit-learn, etc.)
├── main.py                     # Módulo ORQUESTADOR (script de entrada)
├── data/                       # Modelos adicionales (mobileNetV2) y resultados como matrices de confusión
|   └── MobilenetV3/            # Modelos output del entrenamiento y ajuste a ONNX
├── experiments\
|   ├── image_consult.py        # Stress testing enfocado al modelo usando un lote de imágenes locales
|   ├── compare_model_results.html  # Generador de un reporte en HMTL para visualizar las diferencias enrtre las predicciones 
                                    # de dos modelos y la lista del lote de imágenes
|   └── real_pics\              # Lote de imágenes locales para evaluar el desempeño del modelo
├── notebooks\                  # Compendio de Jupyter notebooks con los que se hacen pruebas o entrenamientos adicionales de modelos
└── src/                        # Código fuente modular
    ├── __init__.py
    ├── dataset.py              # Módulo 1: Creación de DataFrames, Dataset PyTorch y DataLoaders
    ├── model.py                # Módulo 2: Construcción del modelo MobileNetV3 y Fine-Tuning
    └── evaluate.py             # Módulo 3: Evaluación final, matriz de confusión y métricas

# Uso

Config.yaml debe ser modificado para otrogar la ruta donde se encuentre el dataset con las imágenes a utilizar.
El script main.py sera el orquestador que llame a los diferentes modulos segun vaya progresando el entrenamiento de cada modelo en particular. Este script ya se encarga de guardar el archivo en extensión .keras el modelo entrenado.

# Quality Assurance

Los scripts de Python image_consult.py e image_consult_onnx.py se encargan de obtener la predicción del modelo especificado como entrada de forma iterativa al proporcionarle un lote de imágenes locales.
image_consult.py carga el modelo en formato .keras, mientras que image_consult_onnx.py lo hace en formato .onnx

Las siguientes variables deben modificarse en caso de realizar alguna prueba mas especifica:
* MODEL_PATH = susituír el nombre del modelo guardado en la carpeta "data" o "data\notebooks"
* ROOT_IMAGES = en caso de utilizar imágenes diferentes especificar la locación
* OUTPUT_JSON_PATH = sustuír con el nombre deseado para el nuevo reporte de QA

La salida de este script será un archivo JSON, que se utilizará como entrada en el compare_model_results.html.
compare_model_results.html se debe abrir en algun web browser, y tras presionar los botones "Seleccionar archivo" se debe buscar el archivo JSON deseado para realizar la comparación.