from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG) #logger config
logger = getLogger(__name__) #logger

import gc
import difflib
import traceback # TRYING
import questionary
from json import dump
from difflib import SequenceMatcher
from pathlib import Path
from subprocess import run, DEVNULL
from json import dump, load, JSONDecodeError
import torch.nn as nn
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent

clean_caches_script_bash = "./" / PROJECT_ROOT / "Utils/scripts/cleancache.sh"
supported_devices_lib_path = PROJECT_ROOT / "ConfigurationModule/ConfigFiles/supported_devices_library.json"


def compareModelArchitecture(model1: object, model2: object) -> None:
    """ 
    Utility function to compare two model architectures.

    Input:
        - model1: the first model object (torch model)
        - model2: the second model object (torch model)
    Output:
        - None
    """


    model1_str = str(model1).splitlines()
    model2_str = str(model2).splitlines()

    # Create a differ object
    diff = difflib.ndiff(model1_str, model2_str)

    print(f"Comparing Model 1 vs Model 2:")
    print("-" * 30)
    
    # Print only the differences
    has_diff = False
    for line in diff:
        if line.startswith('+') or line.startswith('-'):
            print(line)
            has_diff = True
    
    if not has_diff:
        print("Architectures are identical.")


def getHumanReadableValue(value: bytes, suffix: str="B") -> str:
        """
        Scale bytes to its proper format
        e.g:
            1253656 => '1.20MB'
            1253656678 => '1.17GB'

        Input:
            - value: the value in bytes
            - suffix: the string suffix
        Output: 
            - string: the value in string format

        """
        factor = 1024
        for unit in ["", "K", "M", "G", "T", "P"]:
            if value < factor:
                return f"{value:.2f}{unit}{suffix}"
            value /= factor

def getLongestSubString(string1: str, string2: str) -> str:
    """
    Return the longest substring between two strings. 

    Input:
        - string1
        - string2
    Output: 
        - str
    """

    string_match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
    longest_substring = string1[string_match.a : string_match.a + string_match.size]

    return longest_substring

def getFilenameList(directory_path: str) -> list:
    """
    Return a list of filename in a given directory_path. 

    Input: 
        - directory_path: the specified path
    Output:
        - file_name_list: list of filenames
    """

    file_name_list = []
    dir_path = Path(directory_path)

    for file_path in dir_path.iterdir():

        # Check if it is a file
        if file_path.is_file():
            file_name_list.append(file_path.name)

    if file_name_list:
        return file_name_list
    else:
        raise FileNotFoundError(f"No files were found in this directory: {directory_path}")
        exit(0)

def cleanCaches() -> None:
    """
    This function writes the '3' in /proc/sys/vm/drop_caches file in order to drop the unused pages, inodes and dentries.
    Visit man proc_sys_vm manpages for more.

    """
    try:

        result = run([str(clean_caches_script_bash)], check=True, stdout=DEVNULL)

        if result.returncode==0:
            logger.info("CACHE CLEANED FOR INDEPENDENT EXPERIMENTS")
    except ChildProcessError as e:
        logger.error(f"Cache not cleaned correctly. The next measurements could be not independent.\nThe error is: {e}")
    except Exception as e:
        logger.error(f"Encountered a generic problem cleaning the caches. The next measurements could be not independent.\nThe error is: {e}")


def subRunQueue(context, aimodel, inference_loader, config_id, queue):
    """
    Worker function to run in the subprocess.
    """
    try:  
        stats = context.run(aimodel=aimodel, input_data=inference_loader, config_id=config_id)

        queue.put({"status": "success", "data": stats})

    except Exception as e:
        logger.error(f"SubProcess CRASHED: {e}")
        logger.error(traceback.format_exc())
        queue.put({"status": "error", "message": str(e)})



def initialPrint(topic: str) -> None:
    """
    Utility function in order to print the main section of the execution in violet. 

    Input:
        - topic: main section title

    Output:
        - None
    """

    print("\n\t\t"+ '\x1b[35m' + topic + '\033[0m')

def trainEpoch(model: object, loader: object, criterion: object, optimizer: object, device: object):
    """
    Function useful for re-training after pruning optimization. 

    Input:
        - model

    """
    model.train()

    #Freezing parameters, unfreezing classifier.
    for param in model.parameters():
            param.requires_grad = False

    # This finds the last linear layer we just added and unfreezes it.
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for param in module.parameters():
                param.requires_grad = True

    running_loss = 0.0
    for inputs, labels in tqdm(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        #nn.functional.dropout(inputs, p=0.5, training=True)
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)



def checkModelExistence(aimodel: object, config_id: str)-> bool:

    model_name = aimodel.getAllInfo()['model_name']
    onnx_directory_path = PROJECT_ROOT / "ModelData" / "ONNXModels" / f"{config_id}" 
    onnx_model_path = onnx_directory_path /f"{model_name}.onnx"

    if onnx_model_path.exists():
        logger.info(f"ONNX file of {model_name} already exists at {onnx_model_path}")
        return True #TO PASS THE CREATION IF IT ALREADY EXISTS 

    return False


def pickAPlatform() -> (str, int):
    title = "Choose the target device: "

    try:
        with open(supported_devices_lib_path, "r") as supported_device_lib_file:
            supported_device = load(supported_device_lib_file)


        option = questionary.select(title, choices=supported_device["devices"], pointer='>>',  use_indicator=True).ask()

        if option is None:
            logger.critical(f"None option encountered, exiting...")
            exit(1)

        return option

    except JSONDecodeError as e:
        logger.critical(f"Encountered a problem loading the supported devices library file.\nThe specific problem is {e}")

    except Exception as e:
        logger.critical(f"Encountered a generic proble loading the supported devices library file.\nThe specific problem is: {e}")


    exit(1)

def acceleratorWarning() -> None:
    """
    Shows a warning for platforms fournished with an accelerator. 

    """

    logger.warning(f"\nThe target platform is fournished with accelerator. All the models will be quantized. 'Quantization' optimization field is not allowed.\n")
    input("\nPress a key to continue...")


