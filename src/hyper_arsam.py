import os
import random
import pandas as pd

class HyperParameter:

    """
    Boundles a bunch of hyper parameters.
    """
    
    # Random seed
    SEED = 3
    
    
    # Some paths
    PATH_TO_DATA = '../data/'
    
    PATH_TO_UBERMOVEMENT = PATH_TO_DATA + 'UberMovement/'
    PATH_TO_UBERMOVEMENT_ADD = PATH_TO_UBERMOVEMENT + 'additional/'
    PATH_TO_UBERMOVEMENT_TRAIN = PATH_TO_UBERMOVEMENT + 'training/'
    PATH_TO_UBERMOVEMENT_VAL = PATH_TO_UBERMOVEMENT + 'validation/'
    PATH_TO_UBERMOVEMENT_TEST = PATH_TO_UBERMOVEMENT + 'testing/'
    
    
    ### Methods ###
    
    def __init__(self):
    
        """ """
        
        ### Uber Movement ###
        
        
        

