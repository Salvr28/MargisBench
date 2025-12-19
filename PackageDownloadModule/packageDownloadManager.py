from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG) #logger config
logger = getLogger(__name__) #logger


from rich.pretty import pprint
from pathlib import Path
from json import load, decoder, dump
from subprocess import check_call
from sys import executable
from importlib.metadata import distributions
from time import sleep
from Utils.utilsFunctions import initialPrint
from abc import ABC, abstractmethod

PROJECT_ROOT = Path(__file__).resolve().parent.parent
requirements_file_generic_path = str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory" / "generic.txt" )
requirements_file_coral_path = str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory" / "coral.txt")
requirements_file_fusion_path = str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory" / "fusion.txt")

#add other paths here?..

#requirements_file_gpu_path = str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory" / "gpu.txt" )
requirements_installed_path= str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory" / ".installed.json" )



class PackageDownloadManager(ABC):

    
    @abstractmethod
    def _checkAlreadyInstalled(self) -> (bool, bool):
        pass
    
    @abstractmethod
    def _downloadDependencies(self, device: str, installed_requirements_dict: dict):
        pass


    def checkDownloadedDependencies(self) -> None:
        """
        Checks if dependecies are already installed calling _checkingAlreadyInstalled. After, it installs the required dependencies.


        Input:
            - there_is_gpu: bool 
        Output:
            - None
        
        """

        initialPrint("DEPENDENCIES DOWNLOAD\n")
        installed_requirements_dict={}

        try:
            with open(requirements_installed_path, "r") as installed_requirements_file:
                installed_requirements_dict = load(installed_requirements_file)


            installed, _ = self._checkAlreadyInstalled()
            
            if not installed:
                self._downloadDependencies(self._platform, installed_requirements_dict)
            else:
                logger.info(f"NEEDED DEPENDENCIES ALREADY PRESENT...")

            if installed_requirements_dict and not installed:
                with open(requirements_installed_path, "w") as installed_requirements_file:
                    dump(installed_requirements_dict, installed_requirements_file, indent=4)
                
            logger.info("ALL DEPENDENCIES INSTALLED! IF THERE ARE PROBLEMS, MAKE A FORCE-REINSTALL OF THE DEPENDENCIES WITHOUT PIP CACHING.")


        except decoder.JSONDecodeError as e:
            logger.critical(f"Encountered an error Decoding the JSON file at path {requirements_installed_path}. It shouldn't be empty!\nThe specific error is: {e}")
            exit(0)
        except (FileNotFoundError,Exception) as e:
            logger.critical(f"Encountered a generic error checking the already installed dependencies.\nThe specific error is: {e}")
            exit(0)

        


class PackageDownloadManagerGeneric(PackageDownloadManager):

        def __init__(self):
            self._platform = "generic"

        def _checkAlreadyInstalled(self) -> (bool, bool):
            """
                Checks if dependecies are already installed, inspectionating the .installed.json file. 

                Input:
                    - there_is_gpu: bool
                
                Output:
                    - install_needed: bool
                    - install_gpu: bool

            """

            requirementInstalled = {}
            try:
                with open(requirements_installed_path, "r") as installed_requirements:
                    requirementInstalled = load(installed_requirements)

                installed = requirementInstalled.get(self._platform, False)

                return installed, False

            except decoder.JSONDecodeError as e:
                logger.error(f"Encountered an error decoding the JSON file at path {requirements_installed_path}. It shouldn't be empty!\nThe specific error is: {e}")
                logger.info(f"Installing only basic dependencies...")
                return True, False

            except (FileNotFoundError,Exception) as e:
                logger.critical(f"Encountered a generic error checking the already installed dependencies.\nThe specific error is: {e}")
            
            exit(0)



        def _downloadDependencies(self, device: str, installed_requirements_dict: dict):

            logger.info(f"INSTALLING {device.upper()} DEPENDENCIES...")
            sleep(1)
            check_call([executable, '-m', 'pip', 'install', '-r', requirements_file_generic_path])
            installed_requirements_dict[device] = True


class PackageDownloadManagerCoral(PackageDownloadManager):


        def __init__(self):
            self._platform = "coral"

        def _checkAlreadyInstalled(self) -> (bool, bool):
            """
                Checks if dependecies are already installed, inspectionating the .installed.json file. 

                Input:
                    - there_is_gpu: bool
                
                Output:
                    - install_needed: bool
                    - install_gpu: bool

            """

            requirementInstalled = {}
            try:
                with open(requirements_installed_path, "r") as installed_requirements:
                    requirementInstalled = load(installed_requirements)

                installed = requirementInstalled.get(self._platform, False)

                return installed, False

            except decoder.JSONDecodeError as e:
                logger.error(f"Encountered an error decoding the JSON file at path {requirements_installed_path}. It shouldn't be empty!\nThe specific error is: {e}")
                logger.info(f"Installing only basic dependencies...")
                return True, False

            except (FileNotFoundError,Exception) as e:
                logger.critical(f"Encountered a generic error checking the already installed dependencies.\nThe specific error is: {e}")
            
            exit(0)


        def _downloadDependencies(self, device: str, installed_requirements_dict: dict):

            logger.info(f"INSTALLING {device.upper()} DEPENDENCIES...")
            sleep(1)
            check_call([executable, '-m', 'pip', 'install', '-r', requirements_file_coral_path])
            installed_requirements_dict[device] = True


class PackageDownloadManagerFusion(PackageDownloadManager):


        def __init__(self):
            self._platform = "fusion"

        def _checkAlreadyInstalled(self) -> (bool, bool):
            """
                Checks if dependecies are already installed, inspectionating the .installed.json file. 

                Input:
                    - there_is_gpu: bool
                
                Output:
                    - install_needed: bool
                    - install_gpu: bool

            """

            requirementInstalled = {}
            try:
                with open(requirements_installed_path, "r") as installed_requirements:
                    requirementInstalled = load(installed_requirements)

                installed = requirementInstalled.get(self._platform, False)

                return installed, False

            except decoder.JSONDecodeError as e:
                logger.error(f"Encountered an error decoding the JSON file at path {requirements_installed_path}. It shouldn't be empty!\nThe specific error is: {e}")
                logger.info(f"Installing only basic dependencies...")
                return True, False

            except (FileNotFoundError,Exception) as e:
                logger.critical(f"Encountered a generic error checking the already installed dependencies.\nThe specific error is: {e}")
            
            exit(0)


        def _downloadDependencies(self, device: str, installed_requirements_dict: dict):

            logger.info(f"INSTALLING {device.upper()} DEPENDENCIES...")
            sleep(1)
            check_call([executable, '-m', 'pip', 'install', '-r', requirements_file_fusion_path])
            installed_requirements_dict[device] = True





if __name__ == "__main__":
    there_is_gpu = False

    pdm = PackageDownloadManagerFusion()

    pdm.checkDownloadedDependencies(there_is_gpu)
    