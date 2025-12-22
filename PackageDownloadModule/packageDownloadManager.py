from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG) #logger config
logger = getLogger(__name__) #logger


import os
import venv
from rich.pretty import pprint
from pathlib import Path
from json import load, decoder, dump
from subprocess import check_call, CalledProcessError
from sys import executable
from importlib.metadata import distributions
from time import sleep
from Utils.utilsFunctions import initialPrint
from abc import ABC, abstractmethod

PROJECT_ROOT = Path(__file__).resolve().parent.parent
requirements_file_directory = str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory")
converters_file_directory = str(PROJECT_ROOT / "Converters")

#add other paths here?..

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
            self._deps_dir = Path(requirements_file_directory) / "Generic"


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



        def _downloadDependencies(self, platform: str, installed_requirements_dict: dict):

            requirements_file_generic_path = str(self._deps_dir / "generic.txt")

            try:
                logger.info(f"INSTALLING {platform.upper()} DEPENDENCIES...")
                sleep(1)
                return_value = check_call([executable, '-m', 'pip', 'install', '-r', requirements_file_generic_path])
                
                if return_value == 0:
                    installed_requirements_dict[platform] = True

            except CalledProcessError as e:
                logger.critical(f"Encountered error installing dependencies.\nThe specific error is {e}.")
                exit(1)


class PackageDownloadManagerCoral(PackageDownloadManager):


        def __init__(self):
            self._platform = "coral"
            self._deps_dir = Path(requirements_file_directory) / "Coral"
            self._converter_dir = Path(converters_file_directory) / "CoralConverter"
            self._converter_venv_dir = Path(self._converter_dir) / "venv"



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


        def _downloadDependencies(self, platform: str, installed_requirements_dict: dict):

            requirements_file_coral_path = str(self._deps_dir / "coral.txt")
            requirements_file_coral_converter_dependencies_path = str(self._deps_dir / "coral_converter.txt")

            try:
                logger.info(f"INSTALLING {platform.upper()} BASIC DEPENDENCIES...")
                sleep(1)
                return_value_basic = check_call([executable, '-m', 'pip', 'install', '-r', requirements_file_coral_path])

                logger.info(f"INSTALLED BASIC DEPENDENCIES! PASSING TO {platform.upper()} CONVERTER ONES...")

                builder = venv.EnvBuilder(with_pip=True) #for venv
                builder.create(self._converter_venv_dir)

                logger.info(f"VIRTUAL ENV FOR {self._platform} CONVERTER CREATED. INSTALLING DEPENDENCIES...\n")

                return_value_converter = check_call([os.path.join(self._converter_venv_dir, "bin", "python3.10"),"-m", "pip", "install", "-r", requirements_file_coral_converter_dependencies_path])

                logger.info("INSTALLED DEPENDENCIES IN VENV!")

                if return_value_basic == 0 and return_value_converter == 0:
                    installed_requirements_dict[platform] = True


            except CalledProcessError as e:
                logger.critical(f"Encountered error installing dependencies.\nThe specific error is {e}.")
                exit(1)
                


class PackageDownloadManagerFusion(PackageDownloadManager):


        def __init__(self):
            self._platform = "fusion"
            self._deps_dir = Path(requirements_file_directory) / "Fusion"
            self._converter_dir = Path(converters_file_directory) / "FusionConverter"
            self._converter_venv_dir = Path(self._converter_dir) / "venv"


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


        def _downloadDependencies(self, platform: str, installed_requirements_dict: dict):

            requirements_file_fusion_path = str(self._deps_dir / "fusion.txt")
            requirements_file_fusion_converter_dependencies_path = str(self._deps_dir / "fusion_converter.txt")

            try:
                logger.info(f"INSTALLING {platform.upper()} BASIC DEPENDENCIES...")
                sleep(1)
                return_value_basic = check_call([executable, '-m', 'pip', 'install', '-r', requirements_file_fusion_path])

                logger.info(f"INSTALLED BASIC DEPENDENCIES! PASSING TO {platform.upper()} CONVERTER ONES...")

                builder = venv.EnvBuilder(with_pip=True) #for venv
                builder.create(self._converter_venv_dir)

                logger.info(f"VIRTUAL ENV FOR {self._platform} CONVERTER CREATED. INSTALLING DEPENDENCIES...\n")

                return_value_converter = check_call([os.path.join(self._converter_venv_dir, "bin", "python3.10"),"-m", "pip", "install", "-r", requirements_file_fusion_converter_dependencies_path])

                logger.info("INSTALLED DEPENDENCIES IN VENV!")

                if return_value_basic == 0 and return_value_converter == 0:
                    installed_requirements_dict[platform] = True


            except CalledProcessError as e:
                logger.critical(f"Encountered error installing dependencies.\nThe specific error is {e}.")
                exit(1)





if __name__ == "__main__":
    there_is_gpu = False

    pdm = PackageDownloadManagerFusion()

    pdm.checkDownloadedDependencies(there_is_gpu)
    