from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

import torch 
from pathlib import Path
from importlib import import_module
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class DataWrapper():

    def __init__(self):
        """
        Initializes the DataWrapper.
        Data is not loaded until loadInferenceData() is called.
        """

        self.inference_loader = None
        self.current_data_config = None

    def _getModelTransforms(self, model_info):
        """
        Generates the data tranforms based on the provided image_size
        """
        module = import_module(model_info['module'])

        str_weights = model_info['weights_class']

        # Analyze the weights class, standard weights class aren't more than two parts
        parts = str_weights.split(".")
        weights_class = getattr(module, parts[0])

        if len(parts) == 1:
            weights = getattr(weights_class, "DEFAULT")
        else:
            weights = getattr(weights_class, parts[1])

        image_size = model_info['image_size']

        return transforms.Compose([
            weights.transforms()
        ])

    def loadInferenceData(self, dataset_info, model_info):
        """
        Loads or re-loads the inference data (from the 'test' folder)
        """ 

        logger.debug(f"-----> [DATAWRAPPER MODULE] LOAD INFERENCE DATA")
        logger.debug(f"Loading dataset from: {dataset_info['data_dir']}.")
        
        image_size = model_info['image_size']
        weights = model_info['weights_class']

        self.current_data_config = {"data_dir": dataset_info['data_dir'], "batch_size": dataset_info['batch_size'], "image_size": model_info['image_size']}

        data_path = PROJECT_ROOT / self.current_data_config['data_dir']
        data_transforms = self._getModelTransforms(model_info)

        try:
            inference_dataset = datasets.ImageFolder(
                str(data_path / "test"),
                data_transforms
            )

            self.inference_loader = DataLoader(
                inference_dataset, 
                batch_size=dataset_info['batch_size'],
                shuffle=False,
                drop_last=True
            )

            self.current_data_config['class_names'] = inference_dataset.classes
            logger.info(f"Data loaded and setted on {model_info['model_name']}. Classes found: {self.getDatasetInfo('class_names')}")

        except FileNotFoundError:
            logger.error(f"Data directory not found at {data_path / 'test'}")
            self.inference_loader = None
            self.current_data_config['class_names'] = None
        except Exception as e:
            logger.error(f"An error occurred during data loading: {e}")
            self.current_data_config['class_names'] = None

        logger.debug(f"<----- [DATAWRAPPER MODULE] LOAD INFERENCE DATA\n")
        
    def getLoader(self):
        """
        Returns the currently loaded inference dataloader
        """
        if self.inference_loader is None:
            logger.warning("Data has not been loaded... check the loading process")
        return self.inference_loader

    def getDatasetInfo(self, info: str):
        """
        Function that returns current config info based on info passed

        Input:
            -info: key name of current_dataset_config dictionary

        Output:
            -output: value of info wanted
        """

        return self.current_data_config[info]
    


if __name__ == "__main__":


    logger.debug("--- DATAWRAPPER TEST: This is a DEBUG log ---")

    model_path = str(PROJECT_ROOT / "ModelData" / "Weights" / "casting_efficientnet_b0.pth")
    info = {
            'module': 'torchvision.models',
            'model_name': "efficientnet_v2",
            'native': True,
            'distilled': False,
            'weights_path': model_path,
            'device': "cpu",
            'class_name': 'efficientnet_b0',
            'weights_class': 'EfficientNet_B0_Weights.DEFAULT',
            'image_size': 224,
            'num_classes': 2,
            'description': 'efficientnet_v2 from torchvision'
        }

    data_dir = str(PROJECT_ROOT / "ModelData" / "Dataset" / "casting_data")
    dataset_info = {
        'data_dir': data_dir,
        'batch_size': 32
    }

    dataset = DataWrapper()
    dataset.loadInferenceData(model_info=info, dataset_info = dataset_info)

    classes = dataset.getDatasetInfo('class_names')
    logger.debug(f"Dataset classes are {classes}")

    logger.debug("--- DATAWRAPPER TEST END ---")




