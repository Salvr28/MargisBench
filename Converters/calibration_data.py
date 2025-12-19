from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

import torch
import numpy as np
import onnxruntime.quantization as quantization
from pathlib import Path
from BenchmarkingFactory.calibrationDataReader import CustomCalibrationDataReader
from BenchmarkingFactory.dataWrapper import DataWrapper
from BenchmarkingFactory.aiModel import AIModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent

model_weights_path = PROJECT_ROOT / "ModelData" / "Weights"

mobile_path = str(model_weights_path / "mobilenet_v2.pth")
mobile_info = {
    'module': 'torchvision.models',
    'model_name': "mobilenet_v2",
    'native': True,
    'weights_path': mobile_path,
    'device': 'tflite',
    'class_name': 'mobilenet_v2',
    'weights_class': 'MobileNet_V2_Weights.DEFAULT',
    'image_size': 224,
    'num_classes': 2,
    'description': 'mobilenet_v2 from torchvision'
}

data_dir = "ModelData/Dataset/casting_data"
calibration_path = str(PROJECT_ROOT / "Converters" / "CalibrationArray" / "calibration_data.npy")

dataset_info = {
    'data_dir': data_dir,
    'batch_size': 32
}

calibration_dataset = DataWrapper()
calibration_dataset.loadInferenceData(model_info = mobile_info, dataset_info = dataset_info)
dl = calibration_dataset.getLoader()

qdr = CustomCalibrationDataReader(dl, input_name="input")

all_data = []
while True:
    data = qdr.get_next()
    if data is None:
        break

    nchw_data = data['input']
    nhwc_data = nchw_data.transpose(0, 2, 3, 1)
    all_data.append(nhwc_data)

full_calibration_set = np.concatenate(all_data, axis=0) # (N, 224, 224, 3)
full_calibration_set = np.expand_dims(full_calibration_set, axis=1) # (N, 1, 224, 224, 3)

print(f"Saving {full_calibration_set.shape} to {calibration_path}")
np.save(calibration_path, full_calibration_set)
print("Done!!")
