import os
import numpy as np
import json


def import_data_stats(HYPER):

    """ """
    
    # get list of all stats files
    stats_filename_list = os.listdir(HYPER.PATH_TO_DATA_RAW_CLIMART_STATISTICS)
    
    # declare empty stats dictionary to store all files
    stats_dict = {}
    
    # iterate over all file names
    for filename in stats_filename_list:
        
        # set the full path to file
        path_to_file = HYPER.PATH_TO_DATA_RAW_CLIMART_STATISTICS + filename
        
        # variable name
        variable_name = filename[:-4]
        
        # import data
        stats_dict[variable_name] = np.load(path_to_file)
    
    return stats_dict
    
    
def import_meta_json(HYPER):

    """ """
    
    # create path to json file
    path_to_meta_json = HYPER.PATH_TO_DATA_RAW_CLIMART + 'META_INFO.json'
    
    # load json file
    with open(path_to_meta_json, 'r') as f:
        meta_dict = json.load(f)
    
    
    input_dims = meta_dict['input_dims']
    variables = meta_dict['variables']
    feature_by_var = meta_dict['feature_by_var']
    
    return input_dims, variables, feature_by_var
