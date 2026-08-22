from src.dataset import get_dataloaders # Metodo para obtener los datasets
from src.model import build_model   # Placeholder para la base del modelo
# from src.model import compile_model # Placeholder para la compilación del modelo
from src.model import train_model   # Placeholder para bucle de entrenamiento
# from src.evaluate import evaluate_model   # Placeholder para métricas
import yaml

class MainFrame:
    """
    Esta clase contiene el flujo central para entrenar el modelo.
    Cada etapa de dicho entrenamiento se intenta realizar en cada uno de
    los metodos en caso que se requieran crear mas modelos sin tener que
    pasar por todo el flujo nuevamente.
    """
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file)
        for section in config.values():
        # atributos de la clase creados a partir del yaml de configuracion
            if isinstance(section, dict):
                for key, value in section.items():
                    setattr(self, key, value)

        # requested_device = self.device
        # if requested_device == "cuda" and not torch.cuda.is_available():
        #     requested_device = "cpu"
        # self.device = torch.device(requested_device)

        # Inicializar dataset
        self.train_loader, self.val_loader, self.test_loader, self.class_names = get_dataloaders(
            data_dir=self.raw_dataset,
            batch_size=self.batch_size,
            img_size=self.img_size,
            seed=self.seed
        )
        self.base_model(self)

    def base_model(self):
        """
        Construye la base del modelo a partir de MobileNetV2
        """
        return build_model(self.img_size)

    def model_top_layers(self, crafter_path=None):
        """
        Se genera el modelo a partir de un 'crafter' que esta definido
        en una archivo yaml separado
        """

        if not crafter_path:
            raise AttributeError
        
        with open(crafter_path, 'r', encoding='utf-8') as crafter_file:
            crafter = yaml.safe_load(crafter_file)
        # TODO: crafter debe poderse consumir como una serie de argumentos que
        # se introduzcan al models.Model
        return train_model(crafter)

    def fine_tuning(self, fine_tuning_params):
        """
        Re-entrena el modelo con fine tuning si se requiere.
        Se indican las capas a congelar en fine_tuning_params
        """
        

    def evaluate(self):
        """Evalúa el modelo en el conjunto de prueba (placeholder)."""

def main():
    overall_model = MainFrame()
    overall_model.base_model()
    model_alpha = overall_model.model_top_layers('./data/crafter_alpha.yaml')
    model_beta = overall_model.model_top_layers('./data/crafter_beta.yaml')
    model_beta.fine_tuning('fine_tuning_params')

    model_alpha.evaluate()
    model_beta.evaluate()

if __name__ == "__main__":
    main()
