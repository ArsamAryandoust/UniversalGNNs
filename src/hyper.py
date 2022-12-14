import os

class HyperParameter:

    """
    Boundles a bunch of hyper parameters.
    """
    
    ### Data paths ###
    
    # general
    PATH_TO_DATA = '../data/'
    PATH_TO_DATA_RAW = PATH_TO_DATA + 'raw/'
    
    # Uber Movement
    PATH_TO_DATA_RAW_UBERMOVEMENT = PATH_TO_DATA_RAW + 'UberMovement/'
    PATH_TO_DATA_UBERMOVEMENT = PATH_TO_DATA + 'UberMovement/'
    PATH_TO_DATA_UBERMOVEMENT_POLYGONES = PATH_TO_DATA_UBERMOVEMENT + 'city zone polygons/'
    PATH_TO_DATA_UBERMOVEMENT_TRAIN = PATH_TO_DATA_UBERMOVEMENT + 'training/'
    PATH_TO_DATA_UBERMOVEMENT_VAL = PATH_TO_DATA_UBERMOVEMENT + 'validation/'
    PATH_TO_DATA_UBERMOVEMENT_TEST = PATH_TO_DATA_UBERMOVEMENT + 'testing/'
    
    # ClimART
    PATH_TO_DATA_RAW_CLIMART = PATH_TO_DATA_RAW + 'ClimART/'
    PATH_TO_DATA_RAW_CLIMART_INPUTS = PATH_TO_DATA_RAW_CLIMART + 'inputs/'
    PATH_TO_DATA_RAW_CLIMART_OUTPUTS_CLEAR_SKY = PATH_TO_DATA_RAW_CLIMART + 'outputs_clear_sky/'
    PATH_TO_DATA_RAW_CLIMART_OUTPUTS_PRISTINE = PATH_TO_DATA_RAW_CLIMART + 'outputs_pristine/'
    
    
    # Open Catalyst
    PATH_TO_DATA_RAW_OPENCATALYST = PATH_TO_DATA_RAW + 'OpenCatalyst/'
    PATH_TO_DATA_RAW_OPENCATALYST_TRAJECTORIES = PATH_TO_DATA_RAW_OPENCATALYST + 'oc22_trajectories/trajectories/oc22/'
    
    
    
    ### Training, validation, testing splits ###
    
    # Chunk size of data points per .csv file
    CHUNK_SIZE_UBERMOVEMENT = 10_000_000
    
    # share to split training and validation data
    TRAIN_VAL_SPLIT_UBERMOVEMENT = 0.5
    
    # out of distribution test splitting rules in time and space
    TEST_SPLIT_DICT_UBERMOVEMENT = {
        'temporal_dict': {
            'year': 2017,
            'quarter': 3,
            'hours_of_day': [2, 4, 10, 12]
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
        self.UBERMOVEMENT_LIST_OF_CITIES = os.listdir(self.PATH_TO_DATA_RAW_UBERMOVEMENT)[:5]
        self.UBERMOVEMENT_CITY_FILES_MAPPING = {}
        for city in self.UBERMOVEMENT_LIST_OF_CITIES:
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
                    
                    # determine daytype
                    if 'OnlyWeekdays' in filename:
                        daytype = 'weekdays'
                    elif 'OnlyWeekends' in filename:
                        daytype = 'weekends'
                    
                    # determine year
                    for year in year_list:
                        if str(year) in filename:
                            break
                            
                    # determine quarter of year
                    for quarter in quarter_list:
                        if quarter in filename:
                            quarter = int(quarter[1])
                            break
                    
                    # fill dictionary with desired values
                    csv_file_dict['daytype'] = daytype
                    csv_file_dict['year'] = year
                    csv_file_dict['quarter'] = quarter
                    csv_file_dict['filename'] = filename
                    
                    # append csv file dictionary to list
                    csv_file_dict_list.append(csv_file_dict)
                    
            file_dict = {
                'json' : json,
                'csv_file_dict_list': csv_file_dict_list
            }
            self.UBERMOVEMENT_CITY_FILES_MAPPING[city] = file_dict
            
       
        ### Create directories ###
        self.check_create_dir(self.PATH_TO_DATA_UBERMOVEMENT_TEST)
        self.check_create_dir(self.PATH_TO_DATA_UBERMOVEMENT_VAL)
        self.check_create_dir(self.PATH_TO_DATA_UBERMOVEMENT_TRAIN)
        self.check_create_dir(self.PATH_TO_DATA_UBERMOVEMENT)
        self.check_create_dir(self.PATH_TO_DATA_UBERMOVEMENT_POLYGONES)
           
    def check_create_dir(self, path):
    
        """ """
        
        if not os.path.isdir(path):
            os.mkdir(path)
