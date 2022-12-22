import os
import random
import pandas as pd

from IPython.display import display

class Datasets:

    """
    Bundles data
    """
    
    
    

    
    ### Methods ###
    
    def __init__(self):
    
        """ """

        
        
        
    def import_ubermovement_sample(self, HYPER, display_data=False):
    
        """ """
        
        ###
        # Import training validation and testing data ###
        ###

        # list filenames in each folder
        filename_list_train = os.listdir(HYPER.PATH_TO_UBERMOVEMENT_TRAIN)
        filename_list_val = os.listdir(HYPER.PATH_TO_UBERMOVEMENT_VAL)
        filename_list_test = os.listdir(HYPER.PATH_TO_UBERMOVEMENT_TEST)
        
        # sample exemplar filenames
        random.seed(HYPER.SEED)
        filename_train = random.choice(filename_list_train)
        random.seed(HYPER.SEED)
        filename_val = random.choice(filename_list_val)
        random.seed(HYPER.SEED)
        filename_test = random.choice(filename_list_test)
        
        # create path to chosen exemplar files
        path_to_train = HYPER.PATH_TO_UBERMOVEMENT_TRAIN + filename_train
        path_to_val = HYPER.PATH_TO_UBERMOVEMENT_VAL + filename_val
        path_to_test = HYPER.PATH_TO_UBERMOVEMENT_TEST + filename_test

        # import data
        df_uber_train = pd.read_csv(path_to_train)
        df_uber_val = pd.read_csv(path_to_val)
        df_uber_test = pd.read_csv(path_to_test)
        
        
        ###
        # Import city name to ID mapping ###
        ###

        # set filename of city name to id mapping
        filename = '0_city_to_id_mapping.csv'

        # set path to file
        path_to_file = HYPER.PATH_TO_UBERMOVEMENT_ADD + filename

        # read the file as csv
        df_uber_citymapping = pd.read_csv(path_to_file, index_col=0)
        
        
        ###
        # Import geographic city zone data ###
        ###

        # list of cities
        list_of_cities = df_uber_citymapping.index.to_list()

        # declare empty dictionary to save data in
        dict_df_uber_cityzones = {}

        # iterate over all city names
        for city_name in list_of_cities:
            
            # set path to data
            path_to_data = HYPER.PATH_TO_UBERMOVEMENT_ADD + city_name + '.csv'
            
            # import data as csv files
            df_uber_cityzones = pd.read_csv(path_to_data, index_col=0)
            
            # save geographic data of iterated city in dictionary
            dict_df_uber_cityzones[city_name] = df_uber_cityzones
            
                
        # Save data as attributes to class instance
        self.df_uber_train = df_uber_train
        self.df_uber_val = df_uber_val
        self.df_uber_test = df_uber_test
        self.df_uber_citymapping = df_uber_citymapping
        self.dict_df_uber_cityzones = dict_df_uber_cityzones
        
        
        # display data samples
        if display_data:
            display(self.df_uber_train)
            print('Exemplar geographic city zone data for city of', city_name)
            display(self.dict_df_uber_cityzones[city_name])
            display(self.df_uber_citymapping)
        
        
        
    def import_climart_sample(self, HYPER, display_data=False):
    
        """ """
        
        ###
        # Import training validation and testing data ###
        ###

        # list filenames in each folder
        filename_list_train = os.listdir(HYPER.PATH_TO_CLIMART_TRAIN)
        filename_list_val = os.listdir(HYPER.PATH_TO_CLIMART_VAL)
        filename_list_test = os.listdir(HYPER.PATH_TO_CLIMART_TEST)
        
        # sample exemplar filenames
        random.seed(HYPER.SEED)
        filename_train = random.choice(filename_list_train)
        random.seed(HYPER.SEED)
        filename_val = random.choice(filename_list_val)
        random.seed(HYPER.SEED)
        filename_test = random.choice(filename_list_test)
        
        # create path to chosen exemplar files
        path_to_train = HYPER.PATH_TO_CLIMART_TRAIN + filename_train
        path_to_val = HYPER.PATH_TO_CLIMART_VAL + filename_val
        path_to_test = HYPER.PATH_TO_CLIMART_TEST + filename_test

        # import data
        df_climart_train = pd.read_csv(path_to_train)
        df_climart_val = pd.read_csv(path_to_val)
        df_climart_test = pd.read_csv(path_to_test)
        
        # Save data as attributes to class instance
        self.df_climart_train = df_climart_train
        self.df_climart_val = df_climart_val
        self.df_climart_test = df_climart_test
        
        
        # display data samples
        if display_data:
            display(self.df_climart_train)
        
        
        
