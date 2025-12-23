from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

import os
import sys
import subprocess
from pathlib import Path
from abc import ABC, abstractclassmethod
from Utils.utilsFunctions import createPathDirectory

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#####

# TODO 
# Continue creation of directory during convertion
# Then compile every converted model for edge tpu
# Finally inject in coral data, scripts and models

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
            weights = model_config['weights_class']
            weights = weights.split(".")[0]

            arg_set.append(model_config['model_name'])
            arg_set.append(weights)
            args_sets.append(arg_set)

        for args in args_sets:
            logger.info(f"Creating calibration data with these args: {args}")
            subprocess.run([sys.executable, calibration_script_path] + args)

    def createCoralModels(self):
        # Setup venv paths
        venv_root = PROJECT_ROOT / "Converters" / "CoralConverter" / "venv"
        venv_bin = venv_root / "bin"
        onnx2tf_path = str(venv_bin / "onnx2tf")

        # Isolated environment for the subprocess
        sub_env = os.environ.copy()
        sub_env['PATH'] = str(venv_bin) + os.pathsep + sub_env.get("PATH", "")
        #sub_env["TF_CPP_MIN_LOG_LEVEL"] = "3" # Keep the logs clean

        config_id = self.getConfigID()
        onnx_path = PROJECT_ROOT / "ModelData" / "ONNXModels" / config_id

        for model_config in self.getConfig()["models"]:
            model_name = model_config['model_name']
            
            # Base Model Conversion
            base_onnx = str(onnx_path / f"{model_name}.onnx")
            base_out_dir = PROJECT_ROOT / "Converters" / "CoralConverter" / "TfModels" / model_name / f"{model_name}Q"
            base_tflite = base_out_dir / f"{model_name}_full_integer_quant.tflite"
            base_calib = str(PROJECT_ROOT / "Converters" / "CoralConverter" / "Calibration" / "CalibrationArrays" / f"{model_name}_calibration_data.npy")

            createPathDirectory(base_out_dir)

            if not base_tflite.exists():
                logger.info(f"Converting base model: {model_name}")
                base_cmd = [
                    onnx2tf_path, "-i", base_onnx, "-o", str(base_out_dir),
                    "-oiqt", "-iqd", "int8", "-oqd", "int8", "-qt", "per-tensor", 
                    "-b", "1", "-dgc", "-cind", "input", base_calib, "[[[[0.0]]]]", "[[[[1.0]]]]"
                ]
                subprocess.run(base_cmd, env=sub_env)
            else:
                logger.info(f"Base Tflite Model already exists at {base_tflite}")

            # Optimized Models Conversion 
            opts = self.getConfig().get("optimizations", {})
            for opti_key in opts.keys():
                opti_onnx = None
                
                if opti_key == "Pruning":
                    suffix = "pruned"
                elif opti_key == "Distillation":
                    suffix = "distilled"
                else:
                    continue # Skip Quantization 

                opti_onnx = str(onnx_path / f"{model_name}_{suffix}.onnx")
                opti_out_dir = PROJECT_ROOT / "Converters" / "CoralConverter" / "TfModels" / model_name / f"{model_name}Q_{suffix.capitalize()}"
                opti_tflite = opti_out_dir / f"{model_name}_{suffix}_full_integer_quant.tflite"

                if os.path.exists(opti_onnx):
                    createPathDirectory(opti_out_dir)
                    if not opti_tflite.exists():
                        logger.info(f"Converting {opti_key} model: {model_name}")
                        opti_cmd = [
                            onnx2tf_path, "-i", opti_onnx, "-o", str(opti_out_dir),
                            "-oiqt", "-iqd", "int8", "-oqd", "int8", "-qt", "per-tensor", 
                            "-b", "1", "-dgc", "-cind", "input", base_calib, "[[[[0.0]]]]", "[[[[1.0]]]]"
                        ]
                        subprocess.run(opti_cmd, env=sub_env)
                    else:
                        logger.info(f"Optimized tflite model already exists at: {opti_tflite}")
                else:
                    logger.warning(f"Expected optimized model {opti_onnx} not found. Skipping.")


    def compileCoralModelsForEdgeTPU(self):

        compiler_bin = str(PROJECT_ROOT / "Converters" / "CoralConverter" / "coral_compiler" / "x86_64" / "edgetpu_compiler")
        
        config_id = self.getConfigID()
        
        for model_config in self.getConfig()["models"]:
            model_name = model_config['model_name']
            
            tflite_dir = PROJECT_ROOT / "Converters" / "CoralConverter" / "TfModels"
            model_dir = tflite_dir / f"{model_name}"
            base_out_dir =  tflite_dir / f"{model_name}EdgeTPU"


            
            base_tflite_path = model_dir / f"{model_name}Q" /  f"{model_name}_full_integer_quant.tflite"
            base_out_path = base_out_dir / f"{model_name}_edgeTPU.tflite"

            if base_tflite_path.exists():
                createPathDirectory(base_out_dir)
                logger.info(f"Compiling Base EdgeTPU model for {model_name}...")
                # edgetpu_compiler <file> -o <output_dir>
                subprocess.run([compiler_bin, str(base_tflite_path), "-o", str(base_out_dir)])

            opts = self.getConfig().get("optimizations", {})
            for opti_key in opts.keys():
                suffix = None
                if opti_key == "Pruning":
                    suffix = "Pruned"
                elif opti_key == "Distillation":
                    suffix = "Distilled"
                else:
                    continue

                # Path to the .tflite we created in createCoralModels
                opti_tflite_path = PROJECT_ROOT / "Converters" / "CoralConverter" / "TfModels" / model_name / f"{model_name}Q_{suffix}" / f"{model_name}_{suffix.lower()}_full_integer_quant.tflite"
                
                # Destination directory for the compiled version
                opti_out_dir = base_out_dir

                if opti_tflite_path.exists():
                    createPathDirectory(opti_out_dir)
                    logger.info(f"Compiling {opti_key} EdgeTPU model for {model_name}...")
                    subprocess.run([compiler_bin, str(opti_tflite_path), "-o", str(opti_out_dir)])
                else:
                    logger.warning(f"TFLite source for {opti_key} not found at {opti_tflite_path}. Skipping compilation.")  





    def getConfig(self):
        return self.__config
    
    def getConfigID(self):
        return self.__config_id

if __name__ == "__main__":

    config_id = "6bae1867a5_generic"

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


    coral_init = CoralInizializer(config, config_id)
    coral_init.createCalibrationData()
    coral_init.createCoralModels()
    coral_init.compileCoralModelsForEdgeTPU()