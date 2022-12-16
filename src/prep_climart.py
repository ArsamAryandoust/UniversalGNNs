import os
import numpy as np
import json
import h5py
import pandas as pd
import gc



def process_raw_data(
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
    
    # transform into dataframe
    data = pd.DataFrame(data)
    
    # append to input dataframes
    df_inputs_clear_sky = pd.concat([df_inputs_clear_sky, data], axis=1)
    df_inputs_pristine = pd.concat([df_inputs_pristine, data], axis=1)
    
    
    ### Do for layers ###
    
    # retrieve data and tranform into numpy arrays
    data = np.array(inputs['layers'])
    data_pristine = np.array(inputs['layers'][:, :, :14])
    
    # reshape data
    data = reshape(data)
    data_pristine = reshape(data_pristine)
    
    # transform into dataframe
    data = pd.DataFrame(data)
    data_pristine = pd.DataFrame(data_pristine)
    
    # append to input dataframes
    df_inputs_clear_sky = pd.concat([df_inputs_clear_sky, data], axis=1) 
    df_inputs_pristine = pd.concat([df_inputs_pristine, data_pristine], axis=1) 
    
    # free up memory
    del data_pristine
    gc.collect()
    
    
    ### Do for levels ###
    
    # retrieve data and tranform into numpy arrays
    data = np.array(inputs['levels'])
    
    # reshape data
    data = reshape(data)
    
    # transform into dataframe
    data = pd.DataFrame(data)
    
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
        
        # transform into dataframe
        data_clear_sky = pd.DataFrame(data_clear_sky)
        data_pristine = pd.DataFrame(data_pristine)
        
        # append to input dataframes
        df_outputs_clear_sky = pd.concat([df_outputs_clear_sky, data_clear_sky], axis=1)
        df_outputs_pristine = pd.concat([df_outputs_pristine, data_pristine], axis=1)
    
    
    return df_inputs_clear_sky, df_inputs_pristine, df_outputs_clear_sky, df_outputs_pristine


def reshape(data):

    # get number of data points
    n_data = len(data)
    
    # get number of columns for desired reshaping
    n_columns = data.shape[1] * data.shape[2]
    
    # reshape
    data = np.reshape(data, (n_data, n_columns), order='C')
    
    return data

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
