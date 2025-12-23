import os
import sys
import torch
import numpy as np
from importlib import import_module
from pathlib import Path
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, mnasnet1_0, MNASNet1_0_Weights,  MobileNet_V2_Weights, mobilenet_v2
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# --- CONFIGURATION ---
DATA_DIR = str(PROJECT_ROOT / "ModelData" / "Dataset" / "casting_data")
BATCH_SIZE = 1 
IMAGE_SIZE = 224
NUM_SAMPLES = 200 

def generate_calibration_data(model_name, model_weights):

    # Getting the Image Transformations (Normalization, Resize etc) from officials
    module = import_module("torchvision.models")
    weight_class = getattr(module, model_weights)
    weights = getattr(weight_class, 'DEFAULT')
    official_transforms = weights.transforms()

    # Load dataset
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=official_transforms)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Found {len(dataset)} images. Collecting {NUM_SAMPLES} samples...")

    all_images = []
    
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= NUM_SAMPLES:
                break
            
            # image shape: (1, 3, 224, 224) -> NCHW
            numpy_batch = images.cpu().numpy()
            
            # TRANSPOSE to NHWC: (1, 3, 224, 224) -> (1, 224, 224, 3)
            # This is critical for TFLite/onnx2tf
            nhwc_batch = numpy_batch.transpose(0, 2, 3, 1)
            
            all_images.append(nhwc_batch)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1} images...")

    # CONCATENATE into one 4D array
    # Resulting shape: (NUM_SAMPLES, 224, 224, 3)
    calibration_array = np.concatenate(all_images, axis=0)

    # Saving the array
    OUTPUT_PATH = str(PROJECT_ROOT / "Converters" / "CoralConverter" / "Calibration" / "CalibrationArrays" / f"{model_name}_calibration_data.npy")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.save(OUTPUT_PATH, calibration_array)
    
    print("-" * 30)
    print(f"SUCCESS!")
    print(f"Final .npy shape: {calibration_array.shape}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("-" * 30)

if __name__ == "__main__":

    # Get model name and weights from terminal
    args = sys.argv[1:]
    model_name = args[0]
    model_weights = args[1]

    print (f"I get the following args: {model_name} | {model_weights}")

    generate_calibration_data(model_name, model_weights)