# Descripcion
Este repositorio contiene el desarrollo del modelo de vision, el analisis de los datos asi como data augmentation y metricas de desempeño.

# Diseño

El pipe de construccion del modelo esta pensado para que sea modular en scripts de Python, evitando utilizar NoteBooks
Facilitando asi la colaboracion.

La estructura del repositorio que se propone es la siguiente:

vision_model/
│
├── requirements.txt            # Dependencias del proyecto (torch, torchvision, scikit-learn, etc.)
├── main.py                     # Módulo ORQUESTADOR (script de entrada)
├── data/
|   ├── config.yaml             # Hiperparámetros, rutas y configuraciones fijas
|   ├── crafter_x.yaml          # Secuencia inicial de capas superiores del modelo
|   └── fine_tuning_params.yaml # Parametros para congelar capas durante el fine tuning
└── src/                        # Código fuente modular
    ├── __init__.py
    ├── utils.py                # Funciones auxiliares (TBD)
    ├── dataset.py              # Módulo 1: Creación de DataFrames, Dataset PyTorch y DataLoaders
    ├── model.py                # Módulo 2: Construcción del modelo MobileNetV3 y Fine-Tuning
    └── evaluate.py             # Módulo 3: Evaluación final, matriz de confusión y métricas

# Uso

El script main.py sera el orquestador que llame a los diferentes modulos segun vaya progresando el entrenamiento de cada modelo en particular.

# Experimentos/Desempeño