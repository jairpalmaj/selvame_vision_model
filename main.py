import os
from pathlib import Path
from src.dataset import get_dataloaders
from src.model import transfer_learning   # Placeholder para la base del modelo
#from src.model import train_model   # Placeholder para bucle de entrenamiento
from src.evaluate import evaluate_model   # Placeholder para métricas
import yaml

class MainFrame:
    """
    Esta clase contiene el flujo central para entrenar el modelo.
    Cada etapa de dicho entrenamiento se intenta realizar en cada uno de
    los metodos por se requieren crear mas modelos sin tener que
    pasar por todo el flujo nuevamente.
    """
    def __init__(self, config_path='config.yaml') -> None:
        self.root = Path(config_path).parent
        with open(config_path, 'r', encoding='utf-8') as config_file:
            self.config = yaml.safe_load(config_file)
        self.loaders = get_dataloaders(self.config) # Metodo para obtener los datasets
        self.base_model()

    def base_model(self) -> None:
        """
        Construye la base del modelo a partir de MobileNetV2
        """
        self.transfer_model = transfer_learning(self.config, self.loaders, self.root)

    def fine_tuning(self, fine_tuning_params):
        """
        Re-entrena el modelo con fine tuning si se requiere.
        Se indican las capas a congelar en fine_tuning_params
        """
        

    def evaluate(self, model=None):
        """Evalúa el modelo en el conjunto de prueba (placeholder)."""
        if model is None:
            model_to_evaluate = self.transfer_model
        else:
            model_to_evaluate = model
        model_evaluation = evaluate_model(model_to_evaluate, self.loaders, self.root)
        return model_evaluation

def main():
    config_path = os.getcwd() + "\\vision_model\\config.yaml"
    model_alpha = MainFrame(config_path)
    model_alpha.evaluate()

if __name__ == "__main__":
    main()
