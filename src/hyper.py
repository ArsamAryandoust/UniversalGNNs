from os import listdir

class HyperParameter:

    """
    Boundles a bunch of hyper parameters.
    """
    
    ### Data paths ###
    
    # general
    PATH_TO_DATA = '../data/public/'
    PATH_TO_DATA_RAW = PATH_TO_DATA + 'raw/'
    PATH_TO_DATA_PROCESSED = 'processed/'
    
    # Uber Movement
    PATH_TO_DATA_RAW_UBERMOVEMENT = PATH_TO_DATA_RAW + 'UberMovement/'
    PATH_TO_DATA_PROCESSED_UBERMOVEMENT = PATH_TO_DATA_PROCESSED + 'UberMovement/'
    PATH_TO_DATA_PROCESSED_UBERMOVEMENT_POLYGONES = PATH_TO_DATA_PROCESSED_UBERMOVEMENT + 'city zone polygons/'
    
    # ClimART
    PATH_TO_DATA_RAW_CLIMART = PATH_TO_DATA_RAW + 'ClimART/'
    PATH_TO_DATA_RAW_CLIMART_INPUTS = PATH_TO_DATA_RAW_CLIMART + 'inputs/'
    PATH_TO_DATA_RAW_CLIMART_OUTPUTS_CLEAR_SKY = PATH_TO_DATA_RAW_CLIMART + 'outputs_clear_sky/'
    PATH_TO_DATA_RAW_CLIMART_OUTPUTS_PRISTINE = PATH_TO_DATA_RAW_CLIMART + 'outputs_pristine/'
    
    # Open Catalyst
    
    
    ### Methods ###
    
    def __init__(self):
    
        """ Set some paths by reading folders """
        
        ### Uber Movement ###
        year_list = list(range(2010, 2023))
        quarter_list = ['-1-', '-2-', '-3-', '-4']
        self.UBERMOVEMENT_LIST_OF_CITIES = listdir(self.PATH_TO_DATA_RAW_UBERMOVEMENT)
        self.UBERMOVEMENT_CITY_FILES_MAPPING = {}
        for city in self.UBERMOVEMENT_LIST_OF_CITIES:
            path_to_city = self.PATH_TO_DATA_RAW_UBERMOVEMENT + city + '/'
            file_list = listdir(path_to_city)
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
           
       
    
