import os
import random


class HyperParameter:

    """
    Boundles a bunch of hyper parameters.
    """
    
    # Random seed
    SEED = 3
    
    
    ### Data paths ###
    
    # general
    PATH_TO_DATA = '../data/'
    PATH_TO_DATA_RAW = PATH_TO_DATA + 'raw/'
    
    # Uber Movement
    PATH_TO_DATA_RAW_UBERMOVEMENT = PATH_TO_DATA_RAW + 'UberMovement/'
    PATH_TO_DATA_UBERMOVEMENT = PATH_TO_DATA + 'UberMovement/'
    PATH_TO_DATA_UBERMOVEMENT_ADDITIONAL = PATH_TO_DATA_UBERMOVEMENT + 'additional/'
    PATH_TO_DATA_UBERMOVEMENT_TRAIN = PATH_TO_DATA_UBERMOVEMENT + 'training/'
    PATH_TO_DATA_UBERMOVEMENT_VAL = PATH_TO_DATA_UBERMOVEMENT + 'validation/'
    PATH_TO_DATA_UBERMOVEMENT_TEST = PATH_TO_DATA_UBERMOVEMENT + 'testing/'
    
    # ClimART
    PATH_TO_DATA_RAW_CLIMART = PATH_TO_DATA_RAW + 'ClimART/'
    PATH_TO_DATA_RAW_CLIMART_INPUTS = PATH_TO_DATA_RAW_CLIMART + 'inputs/'
    PATH_TO_DATA_RAW_CLIMART_OUTPUTS_CLEAR_SKY = PATH_TO_DATA_RAW_CLIMART + 'outputs_clear_sky/'
    PATH_TO_DATA_RAW_CLIMART_OUTPUTS_PRISTINE = PATH_TO_DATA_RAW_CLIMART + 'outputs_pristine/'
    PATH_TO_DATA_CLIMART = PATH_TO_DATA + 'ClimART/'
    PATH_TO_DATA_CLIMART_ADDITIONAL = PATH_TO_DATA_CLIMART + 'additional/'
    PATH_TO_DATA_CLIMART_TRAIN = PATH_TO_DATA_CLIMART + 'training/'
    PATH_TO_DATA_CLIMART_VAL = PATH_TO_DATA_CLIMART + 'validation/'
    PATH_TO_DATA_CLIMART_TEST = PATH_TO_DATA_CLIMART + 'testing/'
    
    
    # Open Catalyst
    PATH_TO_DATA_RAW_OPENCATALYST = PATH_TO_DATA_RAW + 'OpenCatalyst/'
    PATH_TO_DATA_RAW_OPENCATALYST_TRAJECTORIES = PATH_TO_DATA_RAW_OPENCATALYST + 'oc22_trajectories/trajectories/oc22/'
    
    
    
    ### Training, validation, testing splits ###
    
    # Chunk size of data points per .csv file
    CHUNK_SIZE_UBERMOVEMENT = 20_000_000
    
    # share to split training and validation data
    TRAIN_VAL_SPLIT_UBERMOVEMENT = 0.5
    
    # out of distribution test splitting rules in time and space
    random.seed(SEED)
    quarter_of_year = random.sample(range(1,5), 1)
    random.seed(SEED)
    hours_of_day = random.sample(range(24), 4)
    
    TEST_SPLIT_DICT_UBERMOVEMENT = {
        'temporal_dict': {
            'year': 2017,
            'quarter_of_year': quarter_of_year,
            'hours_of_day': hours_of_day
        },
        'spatial_dict': {
            'city_share': 0.1,
            'city_zone_share': 0.1
        }
    }
    
    
    ### Methods ###
    
    def __init__(self):
    
        """ Set some paths by reading folders """
        
        ### Uber Movement ###
        year_list = list(range(2015, 2021))
        quarter_list = ['-1-', '-2-', '-3-', '-4-']
        self.UBERMOVEMENT_LIST_OF_CITIES = os.listdir(self.PATH_TO_DATA_RAW_UBERMOVEMENT)[:10]
        self.UBERMOVEMENT_CITY_FILES_MAPPING = {}
        self.UBERMOVEMENT_CITY_ID_MAPPING = {}
        for city_id, city in enumerate(self.UBERMOVEMENT_LIST_OF_CITIES):
            path_to_city = self.PATH_TO_DATA_RAW_UBERMOVEMENT + city + '/'
            file_list = os.listdir(path_to_city)
            csv_file_dict_list = []
            for filename in file_list:
                if filename.endswith('.json'):
                    json = filename
                    break
                    
                else:
                    # declare new empty directory to be filled with desired values
                    csv_file_dict = {}
                    
                    # determine if weekday
                    if 'OnlyWeekdays' in filename:
                        daytype = 1
                    elif 'OnlyWeekends' in filename:
                        daytype = 0
                    
                    # determine year
                    for year in year_list:
                        if str(year) in filename:
                            break
                            
                    # determine quarter of year
                    for quarter_of_year in quarter_list:
                        if quarter_of_year in filename:
                            quarter_of_year = int(quarter_of_year[1])
                            break
                    
                    # fill dictionary with desired values
                    csv_file_dict['daytype'] = daytype
                    csv_file_dict['year'] = year
                    csv_file_dict['quarter_of_year'] = quarter_of_year
                    csv_file_dict['filename'] = filename
                    
                    # append csv file dictionary to list
                    csv_file_dict_list.append(csv_file_dict)
                    
            # create file name dictionary
            file_dict = {
                'json' : json,
                'csv_file_dict_list': csv_file_dict_list
            }
            
            # save 
            self.UBERMOVEMENT_CITY_FILES_MAPPING[city] = file_dict
            self.UBERMOVEMENT_CITY_ID_MAPPING[city] = city_id
            
       
        ### Create directories for Uber Movement ###
        self.check_create_dir(self.PATH_TO_DATA_UBERMOVEMENT)
        self.check_create_dir(self.PATH_TO_DATA_UBERMOVEMENT_ADDITIONAL)
        self.check_create_dir(self.PATH_TO_DATA_UBERMOVEMENT_TEST)
        self.check_create_dir(self.PATH_TO_DATA_UBERMOVEMENT_VAL)
        self.check_create_dir(self.PATH_TO_DATA_UBERMOVEMENT_TRAIN)
        
        
        ### ClimART ###
        
        
           
        ### Create directories for CLimART ###
        self.check_create_dir(self.PATH_TO_DATA_CLIMART)
        self.check_create_dir(self.PATH_TO_DATA_CLIMART_ADDITIONAL) 
        self.check_create_dir(self.PATH_TO_DATA_CLIMART_TRAIN)
        self.check_create_dir(self.PATH_TO_DATA_CLIMART_VAL)
        self.check_create_dir(self.PATH_TO_DATA_CLIMART_TEST)
           
           
    def check_create_dir(self, path):
    
        """ """
        
        if not os.path.isdir(path):
            os.mkdir(path)
