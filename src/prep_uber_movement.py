import pandas as pd
import random


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
    
    
def process_csvdata(df_csv_dict, city):
    
    """
    """
    
    df_augmented_csvdata = df_csv_dict['df']
    daytype = df_csv_dict['daytype']
    quarter = df_csv_dict['quarter']
    year = df_csv_dict['year']
    
    df_augmented_csvdata['year'] = year
    df_augmented_csvdata['quarter'] = quarter
    df_augmented_csvdata['daytype'] = daytype
    df_augmented_csvdata['city'] = city 
    
    return df_augmented_csvdata
    
    
def train_val_test_split(HYPER):

    """
    """
    
    # split apart a number of cities for testing
    n_cities_test = HYPER.TEST_SPLIT_DICT_UBERMOVEMENT['spatial_dict']['n_cities']
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
        for df_csv_dict in df_csv_dict_list:
        
            # check if testing year
            if df_csv_dict['year'] == HYPER.TEST_SPLIT_DICT_UBERMOVEMENT['temporal_dict']['year']:
                testing_year = True
            else:
                testing_year = False
                
            # check if testing quarter
            if df_csv_dict['quarter'] == HYPER.TEST_SPLIT_DICT_UBERMOVEMENT['temporal_dict']['quarter']:
                testing_quarter = True
            else:
                testing_quarter = False
            
            # augment csv
            df_augmented_csvdata = process_csvdata(df_csv_dict, city)
            
            if testing_city or testing_year or testing_quarter:
                df_test = pd.concat([df_test, df_augmented_csvdata])
            else:
                pass
    
    return df_train, df_val, df_test
    
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
    
    
def process_all_raw_geojson_data(HYPER):
    
    """ """
    
    for city in HYPER.UBERMOVEMENT_LIST_OF_CITIES:
        df_geojson = import_geojson(HYPER, city)
        df_latitudes, df_longitudes = process_geojson(df_geojson)
        
        filename_lat = city + ' lat.csv'
        filename_lon = city + ' lon.csv' 
        saving_path_lat = (
            HYPER.PATH_TO_DATA_PROCESSED_UBERMOVEMENT_POLYGONES 
            + filename_lat
        )
        saving_path_lon = (
            HYPER.PATH_TO_DATA_PROCESSED_UBERMOVEMENT_POLYGONES 
            + filename_lon
        )
        df_latitudes.to_csv(saving_path_lat)
        df_longitudes.to_csv(saving_path_lon)
        
        
