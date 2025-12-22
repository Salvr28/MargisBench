from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

import os
import sys
import subprocess
from pathlib import Path
from abc import ABC, abstractclassmethod

PROJECT_ROOT = Path(__file__).resolve().parent.parent

######

#####


class Initializers():

    @abstractclassmethod
    def initialize():
        pass

class CoralInizializer():

    def __init__(self, config, config_id):

        # THINK ABOUT TO PASS CONFIG_ID
        self.__config = config
        self.__config_id = config_id

    def createCalibrationData(self):

        calibration_script_path = str(PROJECT_ROOT / "Converters" / "CoralConverter" / "Calibration" / "calibration_data.py")

        args_sets = []
        for model_config in self.getConfig()["models"]:
            arg_set = []
            print(model_config)
            weights = model_config['weights_class']
            weights = weights.split(".")[0]

            arg_set.append(model_config['model_name'])
            arg_set.append(weights)
            args_sets.append(arg_set)

        for args in args_sets:
            logger.info(f"Creating calibration data with these args: {args}")
            subprocess.run([sys.executable, calibration_script_path] + args)

    def createCoralModels(self):
        
        onnx2tf_path = str(PROJECT_ROOT / "Converters" / "venv" / "bin" / "onnx2tf")




    def getConfig(self):
        return self.__config

if __name__ == "__main__":

    config = {
        "models": [
            {
                "module": "torchvision.models",
                "model_name": "mobilenet_v2",
                "native": False,
                "weights_path": "ModelData/Weights/mobilenet_v2.pth",
                "device": "cpu",
                "class_name": "mobilenet_v2",
                "weights_class": "MobileNet_V2_Weights.DEFAULT", 
                "image_size": 224, 
                "num_classes": 2,
                "task": "classification",
                "description": "Mobilenet V2 from torchvision"
            }, 
            {
                "model_name": "efficientnet",
                "native": True,
                "model_name": "efficientnet_b0",
                "weights_class": "EfficientNet_B0_Weights"
            },
            {
                'module': 'torchvision.models',
                'model_name': "mnasnet1_0",
                'native': False,
                'weights_path': "ModelData/Weights/mnasnet1_0.pth",
                'device': "cpu",
                'class_name': 'mnasnet1_0',
                'weights_class': 'MNASNet1_0_Weights.DEFAULT',
                'image_size': 224,
                'num_classes': 2,
                "task": "classification",
                'description': 'mnasnet_v2 from torchvision'
            }
        ],
        "optimizations": {
            "Quantization": {
                "method": "QInt8",
                "type": "static"
            }, 
            "Pruning": {
                "method": "LnStructured",
                'n': 1, 
                "amount": 0.1
            },
            "Distillation":{
                "method": True,
                "distilled_paths": {}
            }
        },
        "dataset": {
            "data_dir": "ModelData/Dataset/casting_data",
            "batch_size": 1
        },
        "repetitions": 2,
        "platform": "generic"
    }


    coral_init = CoralInizializer(config)
    coral_init.createCalibrationData()