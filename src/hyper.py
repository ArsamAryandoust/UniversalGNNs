import os
from datasets import ClimARTDataset, UberMovementDataset, BuildingElectricityDataset

class HyperParameter:

    """
    Bundles a bunch of hyper parameters.
    """
    
    # choose which results to save
    SAVE_BASELINE_RESULTS = True
    SAVE_MAIN_RESULTS = False         # ARSAM-CAUTION: NEEDS TO BE IMPLEMENTED 
    
    # Choose which experiments to run
    RUN_MAIN_EXPERIMENTS = False
    RUN_BASELINE_EXPERIMENTS = True
    
    # Only apply if RUN_BASELINE_EXPERIMENTS == True
    if RUN_BASELINE_EXPERIMENTS: 
        RUN_BASELINE_RF = False
        RUN_BASELINE_GB = True
        RUN_BASELINE_MLP = False
    
    # Choose which dataset to consider
    UBERMOVEMENT = True
    CLIMART = True
    BUILDINGELECTRICITY = True
    
    # model parameters
    LATENT_DIM = 512
    
    # trainng parameters
    MAX_EPOCHS = 30
    BATCH_SIZE = 128
    NUM_BATCHES_PER_EPOCH = 1000
    
    # baseline experiment parameters
    BATCH_SIZE_BASELINE = 2048
    EPOCHS_BASELINE = 20
    NUM_ESTIMATORS_RF = 128
    
    # random seed
    SEED = 3
    
    # data paths
    PATH_TO_RESULTS = 'results/'
    
        
    def __init__(self):
    
        """ """
        
        # add datasets we cant to consider to list
        self.DATASET_CLASS_LIST = []
        if self.UBERMOVEMENT:
            self.DATASET_CLASS_LIST.append(UberMovementDataset)
        if self.CLIMART:
            self.DATASET_CLASS_LIST.append(ClimARTDataset)
        if self.BUILDINGELECTRICITY:
            self.DATASET_CLASS_LIST.append(BuildingElectricityDataset)
            
        # create result directories here
        self.check_create_dir(self.PATH_TO_RESULTS)
        
        
    def check_create_dir(self, path):
    
        """ """
        
        if not os.path.isdir(path):
            os.mkdir(path)
