import os
import numpy as np
import json
import h5py
import pandas as pd
import gc


def create_input_col_name_list(
    var_dict
):
    
    """ """

    # declare empty column names list
    col_names_list = []
    
    # iterate over variable name dict
    for var_name in var_dict:
        
        # get dict with start and end range of current variable
        var_range_dict = var_dict[var_name]
        
        # get range size
        range_size = var_range_dict['end'] - var_range_dict['start']
        
        # append column name to list if range size is only 1
        if range_size == 1:
            col_names_list.append(var_name)
            
        elif range_size >1:
            
            for feature_iter in range(range_size):
                col_name = var_name + '_{}'.format(feature_iter)
                col_names_list.append(col_name)
                
        else:
        
            print('Caution. Something went wrong with creating column names')
    
    return col_names_list



def create_output_col_name_list(
    var_name,
    var_dict
):
    
    """ """
    
    # get variable range dictionary
    var_range_dict = var_dict[var_name]
    
    # declare empty column names list
    col_names_list = []
    
    # get range size
    range_size = var_range_dict['end'] - var_range_dict['start']
    
    # append column name to list if range size is only 1
    for feature_iter in range(range_size):
        col_name = var_name + '_{}'.format(feature_iter)
        col_names_list.append(col_name)

    return col_names_list


def process_raw_data(
    feature_by_var,
    inputs, 
    outputs_clear_sky, 
    outputs_pristine
):
    
    """ """
    
    ###
    # Process input data ###
    ###
    
    # define empty dataframes
    df_inputs_clear_sky = pd.DataFrame()
    df_inputs_pristine = pd.DataFrame()
    
    ### Do for globals ###
    
    # retrieve data and tranform into numpy arrays
    data = np.array(inputs['globals'])
    
    # create column names
    var_dict = feature_by_var['globals']
    col_names_list = create_input_col_name_list(var_dict)
    
    # transform into dataframe
    data = pd.DataFrame(data, columns=col_names_list)
    
    # append to input dataframes
    df_inputs_clear_sky = pd.concat([df_inputs_clear_sky, data], axis=1)
    df_inputs_pristine = pd.concat([df_inputs_pristine, data], axis=1)
    
    
    ### Do for layers ###
    
    # retrieve data and tranform into numpy arrays
    data = np.array(inputs['layers'])
    data_pristine = np.array(inputs['layers'][:, :, :14])
    
    # create column names
    var_dict = feature_by_var['layers']
    col_names_list = create_input_col_name_list(var_dict)
    col_names_list_pristine = col_names_list[:14].copy()
    
    # reshape data
    data, col_names_list = reshape(data, col_names_list)
    data_pristine, col_names_list_pristine = reshape(data_pristine, col_names_list_pristine)
    
    # transform into dataframe
    data = pd.DataFrame(data, columns=col_names_list)
    data_pristine = pd.DataFrame(data_pristine, columns=col_names_list_pristine)
    
    # append to input dataframes
    df_inputs_clear_sky = pd.concat([df_inputs_clear_sky, data], axis=1) 
    df_inputs_pristine = pd.concat([df_inputs_pristine, data_pristine], axis=1) 
    
    # free up memory
    del data_pristine
    gc.collect()
    
    
    ### Do for levels ###
    
    # retrieve data and tranform into numpy arrays
    data = np.array(inputs['levels'])
    
    # create column names
    var_dict = feature_by_var['levels']
    col_names_list = create_input_col_name_list(var_dict)
    
    # reshape data
    data, col_names_list = reshape(data, col_names_list)
    
    # transform into dataframe
    data = pd.DataFrame(data, columns=col_names_list)
    
    # append to input dataframes
    df_inputs_clear_sky = pd.concat([df_inputs_clear_sky, data], axis=1)
    df_inputs_pristine = pd.concat([df_inputs_pristine, data], axis=1)
    
    # free up memory
    del inputs
    gc.collect()
    
    
    ###
    # Process output data ###
    ###
    
    # define empty dataframes
    df_outputs_clear_sky = pd.DataFrame()
    df_outputs_pristine = pd.DataFrame()
    
    # iterate over both outputs simultaneously
    for key_clear_sky, key_pristine in zip(outputs_clear_sky, outputs_pristine):
        
        # retrieve data and tranform into numpy arrays
        data_clear_sky = np.array(outputs_clear_sky[key_clear_sky])
        data_pristine = np.array(outputs_pristine[key_pristine])
        
        # create column names
        var_dict_clear_sky = feature_by_var['outputs_clear_sky']
        col_names_list_outputs_clear_sky = create_output_col_name_list(key_clear_sky, var_dict_clear_sky)
        
        var_dict_pristine = feature_by_var['outputs_pristine']
        col_names_list_output_pristine = create_output_col_name_list(key_pristine, var_dict_pristine)
        
        # transform into dataframe
        data_clear_sky = pd.DataFrame(data_clear_sky, columns=col_names_list_outputs_clear_sky)
        data_pristine = pd.DataFrame(data_pristine, columns=col_names_list_output_pristine)
        
        # append to input dataframes
        df_outputs_clear_sky = pd.concat([df_outputs_clear_sky, data_clear_sky], axis=1)
        df_outputs_pristine = pd.concat([df_outputs_pristine, data_pristine], axis=1)
    
    
    return df_inputs_clear_sky, df_inputs_pristine, df_outputs_clear_sky, df_outputs_pristine



def reshape(data, col_names_list):

    """ """
    
    # get number of data points
    n_data = len(data)
    n_steps = data.shape[1]
    n_features = data.shape[2]
    
    # get number of columns for desired reshaping
    n_columns = n_steps * n_features
    
    # reshape with C order
    data = np.reshape(data, (n_data, n_columns), order='C')
    
    # declare new empty column names list
    expanded_col_names_list = []
    
    # expand col_names_list according to reshape with C order
    for steps in range(n_steps):
        
        # iterate over all column names
        for col_name in col_names_list:
            
            # create entry
            entry= 'l_{}_{}'.format(steps, col_name)
            
            # append entry
            expanded_col_names_list.append(entry)
            
        
    return data, expanded_col_names_list



def import_h5_data(HYPER, year):

    """ """
    
    # create paths to files
    path_to_inputs = (
        HYPER.PATH_TO_DATA_RAW_CLIMART_INPUTS
        + year
        + '.h5'
    )
    path_to_outputs_clear_sky = (
        HYPER.PATH_TO_DATA_RAW_CLIMART_OUTPUTS_CLEAR_SKY
        + year
        + '.h5'
    )
    path_to_outputs_pristine = (
        HYPER.PATH_TO_DATA_RAW_CLIMART_OUTPUTS_PRISTINE
        + year
        + '.h5'
    )
    
    # load data
    inputs = h5py.File(path_to_inputs, 'r')
    outputs_clear_sky = h5py.File(path_to_outputs_clear_sky, 'r')
    outputs_pristine = h5py.File(path_to_outputs_pristine, 'r')
    
    return inputs, outputs_clear_sky, outputs_pristine
    


def visualize_raw_keys_and_shapes(dataset, name):
    
    """ """
    dataset_key_list = list(dataset.keys())
    print('\nKeys of {} data:\n{}'.format(name, dataset_key_list))
    
    for key in dataset_key_list:
        value = dataset[key]
        print('\n{}: \n{}'.format(key, value.shape))



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
    
    
def augment_and_merge(
    year,
    df_inputs_clear_sky, 
    df_inputs_pristine, 
    df_outputs_clear_sky, 
    df_outputs_pristine
):

    """ """
    
    # concatenate dataframes
    df_clear_sky = pd.concat([df_inputs_clear_sky, df_outputs_clear_sky], axis=1)
    df_pristine = pd.concat([df_inputs_pristine, df_outputs_pristine], axis=1)
    
    # calculate for each data point the hour of year
    n_lat, n_lon, n_hours_per_step = 64, 128, 205
    n_points_space = n_lat * n_lon
    hour_of_year = (
        np.floor(df_clear_sky.index.values / n_points_space) * n_hours_per_step
    ).astype(int)
    
    # calculate lat and lon coordinates
    lat = np.arcsin(df_clear_sky['z_cord'])
    lon = np.arccos(df_clear_sky['x_cord'] / np.cos(lat))
    
    # augment data with latitudes
    df_clear_sky.insert(0, 'lat', lat)
    df_pristine.insert(0, 'lat', lat)
    
    # augment data with longitudes
    df_clear_sky.insert(1, 'lon', lon)
    df_pristine.insert(1, 'lon', lon)
    
    # drop geographic columns we do not need anymore
    df_clear_sky.drop(columns=['x_cord', 'y_cord', 'z_cord'])
    df_pristine.drop(columns=['x_cord', 'y_cord', 'z_cord'])
    
    # augment data with year
    df_clear_sky.insert(2, 'year', year)
    df_pristine.insert(2, 'year', year)
    
    # augment data with hour of year
    df_clear_sky.insert(3, 'hour_of_year', hour_of_year)
    df_pristine.insert(3, 'hour_of_year', hour_of_year)
    
    
    return df_clear_sky, df_pristine
    
    
    
