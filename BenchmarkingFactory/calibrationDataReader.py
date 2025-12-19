import torch
import torchvision as tv
import onnx
import onnxruntime as ort
from onnxruntime import quantization

class CustomCalibrationDataReader(quantization.CalibrationDataReader):
    def __init__(self, torch_dl, input_name="input"):

        self.torch_dl = torch_dl        
        self.input_name = input_name
        self.enum_data = iter(self.torch_dl)

    def to_numpy(self, pt_tensor):
        return pt_tensor.detach().cpu.numpy if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

    def get_next(self):

        batch = next(self.enum_data, None)

        batch = next(self.enum_data, None)
        if batch is not None:

            image_tensor = batch[0]

            return {self.input_name: self.to_numpy(image_tensor)}
        else:
            return None
    
    def rewind(self):
        self.enum_data = iter(self.torch_dl)
