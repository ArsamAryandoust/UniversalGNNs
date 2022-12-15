import pandas as pd
import random
import gc


def import_csvdata(HYPER, city):

    """ Imports the Uber Movement data for a passed city """
    
    # import csv data
    files_dict = HYPER.UBERMOVEMENT_CITY_FILES_MAPPING[city]
    df_csv_dict_list = []
    for csv_file_dict in files_dict['csv_file_dict_list']:
        path_to_csv = HYPER.PATH_TO_DATA_RAW_UBERMOVEMENT + city + '/' + csv_file_dict['filename']
        df_csv = pd.read_csv(path_to_csv)
        csv_df_dict = csv_file_dict.copy()
        csv_df_dict['df'] = df_csv
        df_csv_dict_list.append(csv_df_dict)
        
    
    return df_csv_dict_list
    
 
    
def process_csvdata(HYPER, df_csv_dict, city):
    
    """ """
    
    # copy raw dataframe
    df_augmented_csvdata = df_csv_dict['df']
    
    # augment raw dataframe
    df_augmented_csvdata.insert(0, 'city_id', HYPER.UBERMOVEMENT_CITY_ID_MAPPING[city])
    df_augmented_csvdata.insert(3, 'year', df_csv_dict['year'])
    df_augmented_csvdata.insert(4, 'quarter_of_year', df_csv_dict['quarter_of_year'])
    df_augmented_csvdata.insert(5, 'daytype', df_csv_dict['daytype'])
    
    # rename some columns with more clear names
    df_augmented_csvdata.rename(
        columns={'hod':'hour_of_day', 'sourceid':'source_id', 'dstid':'destination_id'}, 
        inplace=True
    )
    
    return df_augmented_csvdata
    
  
    
def train_val_test_split(HYPER):

    """ """
    
    # split apart a number of cities for testing
    n_cities_test = round(
        HYPER.TEST_SPLIT_DICT_UBERMOVEMENT['spatial_dict']['city_share']
        * len(HYPER.UBERMOVEMENT_LIST_OF_CITIES)
    )
    random.seed(HYPER.SEED)
    list_of_cities_test = random.sample(
        HYPER.UBERMOVEMENT_LIST_OF_CITIES, 
        n_cities_test
    )
    list_of_cities_train_val = list(
        set(HYPER.UBERMOVEMENT_LIST_OF_CITIES) 
        - set(list_of_cities_test)
    )
    
    # decleare empty dataframes for trainining validation and testing
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()
    
    # declare data point counters
    train_chunk_counter, val_chunk_counter, test_chunk_counter = 0, 0, 0
    
    # iterate over all available cities
    for city in HYPER.UBERMOVEMENT_LIST_OF_CITIES:
        
        # check if city is in testing city list
        if city in list_of_cities_test:
            testing_city = True
        else:
            testing_city = False
        
        # import all csv files for currently iterated city
        df_csv_dict_list = import_csvdata(HYPER, city)
        
        
        # iterate over all imported csv files for this city
        for iter_csv, df_csv_dict in enumerate(df_csv_dict_list):
        
        
            # check if testing year
            if df_csv_dict['year'] == HYPER.TEST_SPLIT_DICT_UBERMOVEMENT['temporal_dict']['year']:
                testing_year = True
            else:
                testing_year = False
                
            # check if testing quarter of year
            if df_csv_dict['quarter_of_year'] == HYPER.TEST_SPLIT_DICT_UBERMOVEMENT['temporal_dict']['quarter_of_year']:
                testing_quarter = True
            else:
                testing_quarter = False
            
            # augment csv
            df_augmented_csvdata = process_csvdata(HYPER, df_csv_dict, city)
            
            # free up memory     
            del df_csv_dict['df']
            gc.collect()
            
            # get the subset of city zones for test splits once per city
            if iter_csv == 0:
                n_city_zones = max(
                    df_augmented_csvdata['source_id'].max(),
                    df_augmented_csvdata['destination_id'].max()
                )
                
                # get number of test city zones you want to split
                n_test_city_zones = round(
                    n_city_zones * HYPER.TEST_SPLIT_DICT_UBERMOVEMENT['spatial_dict']['city_zone_share']
                )
                
                # randomly sample test city zones
                random.seed(HYPER.SEED)
                test_city_zone_list = random.sample(range(n_city_zones), n_test_city_zones)
            
            if testing_city or testing_year or testing_quarter:
                
                # append all data to test dataframe
                df_test = pd.concat([df_test, df_augmented_csvdata])
                
            else:
                
                    
                # extract the rows from dataframe with matching city zones in origin and destination
                df_test_city_zones = df_augmented_csvdata.loc[
                    (df_augmented_csvdata['destination_id'].isin(test_city_zone_list)) 
                    | (df_augmented_csvdata['source_id'].isin(test_city_zone_list))
                ]
                
                # set the remaining rows for training and validation
                df_augmented_csvdata = df_augmented_csvdata.drop(df_test_city_zones.index)
                
                # append to test dataframe
                df_test = pd.concat([df_test, df_test_city_zones])
                
                # free up memory
                del df_test_city_zones
                gc.collect()
                
                # extract the rows from dataframe with matching hours of data for test
                df_test_hours_of_day = df_augmented_csvdata.loc[
                    df_augmented_csvdata['hour_of_day'].isin(
                        HYPER.TEST_SPLIT_DICT_UBERMOVEMENT['temporal_dict']['hours_of_day']
                    )
                ]
                
                # set the remaining rows for training and validation
                df_augmented_csvdata = df_augmented_csvdata.drop(df_test_hours_of_day.index)
                
                # append to test dataframe
                df_test = pd.concat([df_test, df_test_hours_of_day])
                
                # free up memory
                del df_test_hours_of_day
                gc.collect()
                
                # create training and validation datasets
                df_train_append = df_augmented_csvdata.sample(
                    frac=HYPER.TRAIN_VAL_SPLIT_UBERMOVEMENT
                )
                df_val_append = df_augmented_csvdata.drop(df_train_append.index)
                
                # append training dataset
                df_train = pd.concat([df_train, df_train_append])
                
                # free up memory     
                del df_train_append   
                gc.collect()
            
                # append validation dataset
                df_val = pd.concat([df_val, df_val_append])
                
                # free up memory     
                del df_val_append   
                gc.collect()
            
            # free up memory     
            del df_augmented_csvdata   
            gc.collect()
            
            
            ### Save resulting data in chunks
            df_train, train_chunk_counter = save_chunk(
                HYPER,
                df_train,
                train_chunk_counter,
                HYPER.PATH_TO_DATA_UBERMOVEMENT_TRAIN,
                'training_data'    
            )
            df_val, val_chunk_counter = save_chunk(
                HYPER,
                df_val,
                val_chunk_counter,
                HYPER.PATH_TO_DATA_UBERMOVEMENT_VAL,
                'validation_data'
            )
            df_test, test_chunk_counter = save_chunk(
                HYPER,
                df_test,
                test_chunk_counter,
                HYPER.PATH_TO_DATA_UBERMOVEMENT_TEST,
                'testing_data'
            )

    ### Tell us the rations that result from our splitting rules
    n_train = (train_chunk_counter * HYPER.CHUNK_SIZE_UBERMOVEMENT) + len(df_train.index)
    n_val = (val_chunk_counter * HYPER.CHUNK_SIZE_UBERMOVEMENT) + len(df_val.index)
    n_test = (test_chunk_counter * HYPER.CHUNK_SIZE_UBERMOVEMENT) + len(df_test.index)
    n_total = n_train + n_val + n_test
    
    print(
        "Training data   :    {:.0%} \n".format(n_train/n_total),
        "Validation data :    {:.0%} \n".format(n_val/n_total),
        "Testing data    :    {:.0%} \n".format(n_test/n_total)
    )
    
    ### Save results of last iteration
    df_train, train_chunk_counter = save_chunk(
        HYPER,
        df_train,
        train_chunk_counter,
        HYPER.PATH_TO_DATA_UBERMOVEMENT_TRAIN,
        'training_data',
        last_iteration=True  
    )
    df_val, val_chunk_counter = save_chunk(
        HYPER,
        df_val,
        val_chunk_counter,
        HYPER.PATH_TO_DATA_UBERMOVEMENT_VAL,
        'validation_data',
        last_iteration=True  
    )
    df_test, test_chunk_counter = save_chunk(
        HYPER,
        df_test,
        test_chunk_counter,
        HYPER.PATH_TO_DATA_UBERMOVEMENT_TEST,
        'testing_data',
        last_iteration=True  
    )
    
    return df_train, df_val, df_test



def save_chunk(
    HYPER,
    df,
    chunk_counter,
    saving_path,
    filename,
    last_iteration=False 
):

    """ """
    
    ### Save resulting data in chunks
    while len(df.index) > HYPER.CHUNK_SIZE_UBERMOVEMENT or last_iteration:
        
        # increment chunk counter 
        chunk_counter += 1
        
        # create path
        saving_path = (
            saving_path
            + filename
            + '_{}.csv'.format(chunk_counter)
        )
        
        # shuffle
        df = df.sample(frac=1)
        
        # save chunk
        df.iloc[:HYPER.CHUNK_SIZE_UBERMOVEMENT].to_csv(saving_path, index=False)
        
        # delete saved chunk 
        df = df[:HYPER.CHUNK_SIZE_UBERMOVEMENT]
        
        # set false for safety. Should not make a difference though.
        last_iteration = False
        
    return df, chunk_counter
    
    
    
def process_geojson(df_geojson):

    """ Maps Uber Movement city zone IDs to a flattened list of latitude and 
    longitude coordinates in the format of two dictionaries. Uses the recursive 
    function called foster_coordinates_recursive to flatten the differently nested 
    data.
    """
    
    df_geojson.pop('type')
    df_geojson = df_geojson['features']
    
    map_json_entry_to_movement_id = dict()

    for json_id, json_entry in enumerate(df_geojson):
        
        map_json_entry_to_movement_id[json_id] = int(
          json_entry['properties']['MOVEMENT_ID']
        )
    
    map_movement_id_to_latitude_coordinates = dict()
    map_movement_id_to_longitude_coordinates = dict()

    for k, v in map_json_entry_to_movement_id.items():
        map_movement_id_to_latitude_coordinates[v] = []
        map_movement_id_to_longitude_coordinates[v] = []


    for json_id, movement_id in map_json_entry_to_movement_id.items():
        coordinates = df_geojson[json_id]['geometry']['coordinates']
        
        (
            map_movement_id_to_latitude_coordinates, 
            map_movement_id_to_longitude_coordinates
        ) = foster_coordinates_recursive(
            movement_id,
            map_movement_id_to_latitude_coordinates,
            map_movement_id_to_longitude_coordinates,
            coordinates
        )
        
    
    df_latitudes = pd.DataFrame.from_dict(
        map_movement_id_to_latitude_coordinates, 
        orient='index'
    ).transpose()
    
    df_longitudes = pd.DataFrame.from_dict(
        map_movement_id_to_longitude_coordinates, 
        orient='index'
    ).transpose()
    
    return df_latitudes, df_longitudes



def foster_coordinates_recursive(
    movement_id,
    map_movement_id_to_latitude_coordinates,
    map_movement_id_to_longitude_coordinates,
    coordinates
):

    """ Flattens the coordinates of a passed city zone id (movement_id)
    and coordiates list recursively and saves their numeric values
    in the dictionaries that map movement ids to a list of latitude and 
    longitude coordinates.
    """

    dummy = 0

    for j in coordinates:

        if type(j) != list and dummy == 0:

            map_movement_id_to_longitude_coordinates[movement_id].append(j)
            dummy = 1
            continue

        elif type(j) != list and dummy == 1:

            map_movement_id_to_latitude_coordinates[movement_id].append(j)
            break

        else:

            dummy = 0
            coordinates = j
            (
                map_movement_id_to_latitude_coordinates,
                map_movement_id_to_longitude_coordinates
            ) = foster_coordinates_recursive(
                movement_id,
                map_movement_id_to_latitude_coordinates,
                map_movement_id_to_longitude_coordinates,
                coordinates
            )

    map_movement_id_to_coordinates = (
        map_movement_id_to_latitude_coordinates,
        map_movement_id_to_longitude_coordinates
    )

    return map_movement_id_to_coordinates

    
    
def import_geojson(HYPER, city):

    """ """
    
    # get dict of 'json', 'weekdays' and 'weekends' data file names
    files_dict = HYPER.UBERMOVEMENT_CITY_FILES_MAPPING[city]

    # import geojson data
    path_to_json = HYPER.PATH_TO_DATA_RAW_UBERMOVEMENT + city + '/' + files_dict['json']
    df_geojson = pd.read_json(path_to_json)
        
    return df_geojson
    
 
def save_city_id_mapping(HYPER):
    
    """ """
    
    # create dataframe from dictionary
    df = pd.DataFrame.from_dict(
        HYPER.UBERMOVEMENT_CITY_ID_MAPPING, 
        orient='index', 
        columns=['city_id']
    )
    
    # set filename
    filename = '0_city_to_id_mapping.csv'
    
    # create saving path
    saving_path = (
        HYPER.PATH_TO_DATA_UBERMOVEMENT_POLYGONS 
        + filename
    )
    
    # save 
    df.to_csv(saving_path)
    
def process_all_raw_geojson_data(HYPER):
    
    """ """
    
    save_city_id_mapping(HYPER)
    
    for city in HYPER.UBERMOVEMENT_LIST_OF_CITIES:
        df_geojson = import_geojson(HYPER, city)
        df_latitudes, df_longitudes = process_geojson(df_geojson)
        
        filename_lat = city + ' lat.csv'
        filename_lon = city + ' lon.csv' 
        saving_path_lat = (
            HYPER.PATH_TO_DATA_UBERMOVEMENT_POLYGONS 
            + filename_lat
        )
        saving_path_lon = (
            HYPER.PATH_TO_DATA_UBERMOVEMENT_POLYGONS 
            + filename_lon
        )
        df_latitudes.to_csv(saving_path_lat)
        df_longitudes.to_csv(saving_path_lon)
        
        
