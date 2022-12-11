from os import listdir

class HyperParameter:

    """
    Boundles a bunch of hyper parameters.
    """
    
    ### Data paths ###
    
    PATH_TO_DATA = '../data/public/'
    PATH_TO_DATA_RAW = PATH_TO_DATA + 'raw/'
    PATH_TO_DATA_RAW_UBERMOVEMENT = PATH_TO_DATA_RAW + 'UberMovement/'
    PATH_TO_DATA_RAW_CLIMART = PATH_TO_DATA_RAW + 'ClimART/'
    PATH_TO_DATA_RAW_CLIMART_INPUTS = PATH_TO_DATA_RAW_CLIMART + 'inputs/'
    PATH_TO_DATA_RAW_CLIMART_OUTPUTS_CLEAR_SKY = PATH_TO_DATA_RAW_CLIMART + 'outputs_clear_sky/'
    PATH_TO_DATA_RAW_CLIMART_OUTPUTS_PRISTINE = PATH_TO_DATA_RAW_CLIMART + 'outputs_pristine/'
    
    
    ### Methods ###
    
    def __init__(self):
    
        """ Set some paths by reading folders """
        
        self.UBERMOVEMENT_LIST_OF_CITIES = listdir(self.PATH_TO_DATA_RAW_UBERMOVEMENT)
        self.UBERMOVEMENT_CITY_FILES_MAPPING = {}
        for city in self.UBERMOVEMENT_LIST_OF_CITIES:
            path_to_city = self.PATH_TO_DATA_RAW_UBERMOVEMENT + city + '/'
            file_list = listdir(path_to_city)
            for filename in file_list:
                if filename.endswith('.json'):
                    json = filename
                elif 'OnlyWeekdays' in filename:
                    weekdays = filename
                elif 'OnlyWeekends' in filename:
                    weekends = filename
                    
                    
            filedictionary = {
                'json' : json,
                'weekdays': weekdays,
                'weekends': weekends
            }
            self.UBERMOVEMENT_CITY_FILES_MAPPING[city] = filedictionary
           
    
