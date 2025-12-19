from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

from os import remove
from json import JSONDecodeError, load
from numpy import percentile
from Utils.utilsFunctions import getHumanReadableValue

class CalculateStats:

    def printStats(input: dict, topic: str) -> None:
        """
        Handler function to print Stas of the model on terminal.

        Input:
            - input: dict that contains couples key, value to print.
            - topic: the topic to print at the first line

        """
        print("\n" +"-"*10 + '\x1b[32m' + topic + '\033[0m' + "-"*10)
        for key, value in input.items():
            if key=="Accuracy":
                print("\n")
                
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
                continue
            
            print(f"{key}: {value}")
                

        print("-"*10 + "-"*len(topic)+"-"*10+"\n")



    def _calculateKernelStats(profile_file_path: str, num_batches: int, total_images: int, correct: int, total: int, running_loss: int) -> dict:
        """
        Parses an ONNX Runtime profile JSON file to get pure kernel statistics.
        
        Input:
            -profile_file_path: The path to the profile.json file.
            -num_batches: The total number of batches in the inference run
                        (e.g., len(inference_loader)).
            -total_images: The total number of images in the dataset
                        (e.g., len(inference_loader.dataset)).
            -correct: total correct prediction counter
            -total: total prediction counter
            -running_loss: calculated loss

        Output:
            -stats: A dictionary with total time, per-batch avg, and per-image avg.
        """
        total_kernel_time_us = 0
        total_model_run_time_us = 0
        total_sequential_execution_time_us = 0
        max_layer_memory_arena_consumption= 0
        
        memory_weights_consumption = 0
        layer_finished  = False


        node_events = []
        

        try:
            with open(profile_file_path, 'r') as f:
                trace_data = load(f)

            # Iterate over all events in the trace
            for event in trace_data:
                event_cat = event.get("cat")
                event_name = event.get("name")
                event_dur = event.get("dur", 0)

                if event_cat  == "Node":
                    duration_us = event_dur
                    total_kernel_time_us += duration_us
                    node_events.append(duration_us)

                    if not layer_finished:
                        memory_arena_consumption = int(event.get("args").get("output", 0)) + int(event.get("args").get("activation_size", 0))
                        if memory_arena_consumption > max_layer_memory_arena_consumption:
                            max_layer_memory_arena_consumption = memory_arena_consumption

                    if not layer_finished:
                        memory_weights_consumption += int(event.get("args").get("parameter_size", 0))

                elif event_cat == "Session" and event_name == "model_run":
                    total_model_run_time_us += event_dur

                elif event_cat == "Session" and event_name == "SequentialExecutor::Execute":
                    total_sequential_execution_time_us += event_dur
                    layer_finished=True
            
            if num_batches == 0 or total_images == 0:
                logger.error(f"Number of batches or images cannot be zero. im: {total_images}; batch: {num_batches}")
                return {}

            if not node_events:
                logger.warning(f"No Node events found in {profile_file_path}.")
                return {}

            # Calculate the stats to return
            total_kernel_time_ms = total_kernel_time_us / 1000.0
            total_model_run_time_ms = total_model_run_time_us / 1000.0
            total_sequential_execution_time_ms = total_sequential_execution_time_us / 1000.0

            avg_kernel_time_per_batch_ms = total_kernel_time_ms / num_batches
            avg_kernel_time_per_image_ms = total_kernel_time_ms / total_images
            avg_sequential_executor_time_per_batch_ms = total_sequential_execution_time_ms / num_batches
            avg_sequential_executor_time_per_image_ms = total_sequential_execution_time_ms / total_images
            avg_model_run_time_per_batch_ms = total_model_run_time_ms / num_batches
            avg_model_run_time_per_image_ms = total_model_run_time_ms / total_images
            
            total_onnx_runtime_overhead = total_model_run_time_ms - total_sequential_execution_time_ms
            avg_onnx_runtime_overhead_per_batch_ms = avg_model_run_time_per_batch_ms - avg_sequential_executor_time_per_batch_ms
            avg_onnx_runtime_overhead_per_image_ms = avg_model_run_time_per_image_ms - avg_sequential_executor_time_per_image_ms
            

            node_latencies_ms = [n / 1000.0 for n in node_events]
            p95_node_latency_ms = percentile(node_latencies_ms, 95)

            accuracy = 100 * correct / total
            average_loss = running_loss / total

            logger.debug(f"Inference Event Path: {profile_file_path}")

            stats = {
                "Total 'kernel' inference time": total_kernel_time_ms,
                "Avg. 'kernel' inference time per batch": avg_kernel_time_per_batch_ms,
                "Avg. 'kernel' inference time per image": avg_kernel_time_per_image_ms,
                "Total sequential executor time": total_sequential_execution_time_ms,
                "Avg. sequential executor time per batch": avg_sequential_executor_time_per_batch_ms,
                "Avg. sequential executor time per image": avg_sequential_executor_time_per_image_ms,
                "Total model run time": total_model_run_time_ms,
                "Avg. model run time per batch": avg_model_run_time_per_batch_ms,
                "Avg. model run time per image": avg_model_run_time_per_image_ms,
                "Total ONNX runtime overhead": total_onnx_runtime_overhead,
                "Avg. ONNX runtime overhead per batch": avg_onnx_runtime_overhead_per_batch_ms,
                "Avg. ONNX runtime overhead per image": avg_onnx_runtime_overhead_per_batch_ms,
                "total_nodes_executed": len(node_events),
                "p95_node_latency_ms": p95_node_latency_ms,
                "Max Memory Arena Consumption": getHumanReadableValue(max_layer_memory_arena_consumption), 
                "Weights Memory Consumptions": getHumanReadableValue(memory_weights_consumption),
                "Accuracy": accuracy,
                "Avg. Loss": average_loss
            }

        
            # Clean up the file
            try:
                remove(profile_file_path)
                logger.debug(f"Cleaned up profile file: {profile_file_path}")
            except OSError as e:
                logger.warning(f"Could not delete profile file {profile_file_path}: {e}")


            return stats

        except FileNotFoundError:
            logger.error(f"Profile file not found: {profile_file_path}")
            return {}
        except JSONDecodeError:
            logger.error(f"Error decoding JSON from {profile_file_path}")
            return {}
        except Exception as e:
            logger.error(f"An error occured during profiling: {e}")
            return {}


    # def getRssTracing(process) -> int:
    #     return process.memory_info().rss

    def printStats(input: dict, topic: str) -> None:
        """
        Handler function to print Stas of the model on terminal.

        Input:
            - input: dict that contains couples key, value to print.
            - topic: the topic to print at the first line
        Output:
            - None 
        """
        print("\n" +"-"*10 + '\x1b[32m' + topic + '\033[0m' + "-"*10+"\n")
        for key, value in input.items():
            if key=="Accuracy":
                print("\n")
                
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
                continue
            
            print(f"{key}: {value}")
                

        print("\n"+"-"*10 + "-"*len(topic)+"-"*10+"\n")



    def calculateStats(profile_file_path: str, num_batches: int, total_images: int, correct: int, total: int, running_loss: int) -> dict:
        """
        Calculates the Kernel Stats, Ram Usage for each model inferencing. 


        Input:
            -profile_file_path: The path to the profile.json file.
            -num_batches: The total number of batches in the inference run
                        (e.g., len(inference_loader)).
            -total_images: The total number of images in the dataset
                        (e.g., len(inference_loader.dataset)).
            -correct: total correct prediction counter
            -total: total prediction counter
            -running_loss: calculated loss

        Output:
            A dictionary with total time, per-batch avg, and per-image avg.
        """

        state_dict = {}


        try:
            state_dict = CalculateStats._calculateKernelStats(profile_file_path, num_batches, total_images, correct, total, running_loss)
            # TODO: RAM TRACING
        except Exception as e:
            logger.error(f"Encountered a generic error calculating kernel stats.\nThe specific error is: {e}")


        return state_dict