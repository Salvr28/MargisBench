from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)


from Utils.utilsFunctions import pickAPlatform, acceleratorWarning

class PlatformContext():


    def __init__(self):
        """
        Initialize the Device Context of the system, creates the concrete strategies
        """

        self.__packageDownloadManager = None
        self.__runnerModule = None
        self.__configurationManager = None
        self.__statsModule = None
        self.__platform = pickAPlatform()

        match self.__platform:

            case "generic":
                
                # --- Generic (ONNX) Imports ---
                from PackageDownloadModule.packageDownloadManager import PackageDownloadManagerGeneric 
                from ConfigurationModule.configurationManager import ConfigManagerGeneric 
                from Runner.runner import RunnerModuleGeneric

                self.__configurationManager = ConfigManagerGeneric(self.__platform)
                self.__packageDownloadManager = PackageDownloadManagerGeneric()
                self.__runnerModule = RunnerModuleGeneric()
                #self.__statsModule = StatsModuleGeneric()

                logger.debug(f"CONTEXT INITIALIZED:")
                logger.debug(f"RUNNER MODULE: GENERIC RUNNER with {self.__runnerModule}")

            case "coral":

                # --- Coral Imports ---
                from PackageDownloadModule.packageDownloadManager import PackageDownloadManagerCoral
                from ConfigurationModule.configurationManager import ConfigManagerCoral

                acceleratorWarning()

                self.__packageDownloadManager = PackageDownloadManagerCoral()
                self.__configurationManager = ConfigurationManagerCoral()
                #self.__runnerModule = RunnerModuleCoral()
                #self.__statsModule = StatsModuleCoral() 

            case "fusion":

                # --- Fusion Imports ---
                from PackageDownloadModule.packageDownloadManager import PackageDownloadManagerFusion
                from ConfigurationModule.configurationManager import ConfigManagerFusion

                acceleratorWarning()

                self.__configurationManager = ConfigurationManagerFusion()
                self.__packageDownloadManager = PackageDownloadManagerFusion()
                #self.__runnerModule = RunnerModuleFusion()
                #self.__statsModule = StatsModuleFusion()

            
            case _:
                logger.error(f"No Match for platform")
                exit(0)



    def run(self, aimodel, input_data, config_id):
        return self.__runnerModule._runInference(aimodel=aimodel, input_data=input_data, config_id=config_id)

    def createConfigFile(self, config) -> str:
        return self.__configurationManager.createConfigFile(config)

    def loadConfigFile(self)-> (dict, str):
        return self.__configurationManager.loadConfigFile()
    
    def checkDownloadedDependencies(self):
        self.__packageDownloadManager.checkDownloadedDependencies()
            
                




