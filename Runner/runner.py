from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

import torch
import onnxruntime as ort
from tqdm import tqdm
from abc import ABC, abstractmethod
from pathlib import Path
from numpy import float32
from BenchmarkingFactory.aiModel import AIModel
from Utils.calculateStats import CalculateStats


PROJECT_ROOT = Path(__file__).resolve().parent.parent

class RunnerModule(ABC):

    @abstractmethod
    def _runInference(self, aimodel: AIModel, input_data, config_id):
        pass


class RunnerModuleGeneric(RunnerModule):

    def __init__(self):
        pass

    def _runInference(self, aimodel: AIModel, input_data, config_id):
        """
        Function to run inference on a given dataset

        Inputs: 
            - input_data: Dataset passed from the dataloader

        Outputs:
            - stats_path: Path of 
        """
        
        logger.debug(f"-----> [RUNNER GENERIC MODULE] RUN INFERENCE")

        device_str = aimodel.getInfo('device')
        model_name = aimodel.getInfo('model_name')
        onnx_model_path = PROJECT_ROOT / "ModelData" / "ONNXModels"  / f"{config_id}" / f"{model_name}.onnx"

        provider_list = aimodel._getProviderList(aimodel.getInfo('device'))

        try:
            # Enable profiling
            sess_options = ort.SessionOptions()
            sess_options.enable_mem_pattern = True
            sess_options.enable_profiling = True
            
            sess_options.profile_file_prefix = model_name
            logger.debug(f"Session is enabled with profiling")

            ort_session = ort.InferenceSession(str(onnx_model_path), providers=provider_list, sess_options = sess_options)

            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            
            input_type = float32
            output_type = float32

            logger.debug(f"Input of ort session: {ort_session.get_inputs()[0]}")
            logger.debug(f"Output of ort session: {ort_session.get_outputs()[0]}")

        except Exception as e:
            logger.debug(f"Error loading ONNX model: {e}")

        io_binding = ort_session.io_binding()

        n_total_images = len(input_data.dataset)
        num_batches = len(input_data)
        logger.info(f"INFERENCING OVER {num_batches} BATCHES...\n")

        logger.debug(f"In this dataset there are {n_total_images} images across {num_batches} batches")
        
        total = 0
        correct = 0
        running_loss = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in tqdm(input_data):

                labels = labels.to(device_str)
                batch_size = inputs.shape[0]

                input_as_numpy = inputs.numpy()

                ort_input_value = ort.OrtValue.ortvalue_from_numpy(input_as_numpy)

                # Binding inputs and outputs on cpu
                io_binding.bind_input(
                    name=input_name,
                    device_type='cpu',
                    device_id=0,  
                    element_type=input_type,
                    shape=tuple(input_as_numpy.shape),
                    buffer_ptr=input_as_numpy.ctypes.data 
                )
                io_binding.bind_output(output_name, device_type = 'cpu',
                                        device_id=0)


                ort_session.run_with_iobinding(io_binding)


                # Get outputs and reconvert into torch tensors
                onnx_outputs_ort = io_binding.get_outputs()
                numpy_output = onnx_outputs_ort[0].numpy()
                onnx_outputs_tensor = torch.from_numpy(numpy_output)

                # Cleaning binding for next iteration
                io_binding.clear_binding_inputs()
                io_binding.clear_binding_outputs()

                loss = criterion(onnx_outputs_tensor, labels)
                running_loss += loss.item() * batch_size
                predicted_indices = torch.argmax(onnx_outputs_tensor, dim=1)
                total += labels.shape[0]
                correct += (predicted_indices == labels).sum().item()

        # Get profile path
        profile_file_path = ort_session.end_profiling()


        if not profile_file_path:
            logger.error(f"Profiling enabled but no file was generated.")
            return {}


        logger.debug(f"Profile file generated: {profile_file_path}")
        
        # Get kernel stats
        #logger.info(f"MEMORY ALLOCATED FOR THE SESSION: {getHumanReadableValue(memory_after_session-memory_before_session)}")
        #logger.info(f"TOTAL MEMORY ALLOCATED THROUGH RUN (WEIGHTS + ARENA): {getHumanReadableValue(max_memory_arena_allocated)}")
        stats = CalculateStats.calculateStats(profile_file_path, num_batches, n_total_images, correct, total, running_loss)

        
        CalculateStats.printStats(stats, f" {model_name.upper()} STATS ")            

        logger.debug(f"<----- [AIMODEL MODULE] RUN INFERENCE\n")

        return stats

