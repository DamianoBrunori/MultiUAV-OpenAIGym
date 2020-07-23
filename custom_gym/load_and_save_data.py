# MAIN CLASSES AND METHODS TO SAVE AND LOAD DATA.

import pickle
from os import mkdir
from os.path import join, isdir
import numpy as np
from my_utils import *
#from scenario_objects import Point, Cell, User
import copyreg
import scenario_objects


class Directories:
    '''
    |-------------------------------------------------|
    |Create (if not exists) directories to store data:|
    |-------------------------------------------------|
    '''

    def __init__(self):
        pass

    def create_map_data_dir(self):

        if not isdir(MAP_DATA_DIR): mkdir(MAP_DATA_DIR)
    
    def create_map_status_dir(self):

        if not isdir(MAP_STATUS_DIR): mkdir(MAP_STATUS_DIR)

    def create_users_clusters_dir(self):

        if not isdir(INITIAL_USERS_DIR): mkdir(INITIAL_USERS_DIR)


class Loader(object):
    '''
    |----------|
    |Load data:|
    |----------|
    '''

    def maps_data(self):
        # Data map made by Points and Cells:

        with open(join(MAP_DATA_DIR, OBS_POINTS_LIST), "rb") as f:
            self._obs_points = pickle.load(f)
        with open(join(MAP_DATA_DIR, CS_POINTS), "rb") as f:
            self._cs_points = pickle.load(f)
        with open(join(MAP_DATA_DIR, POINTS_MATRIX), "rb") as f:
            self._points_matrix = pickle.load(f)
        with open(join(MAP_DATA_DIR, ENODEB_POINT), "rb") as f:
            self._eNB_point = pickle.load(f)
        
        with open(join(MAP_DATA_DIR, CELLS_MATRIX), "rb") as f:
            self._cells_matrix = pickle.load(f)
        with open(join(MAP_DATA_DIR, OBS_CELLS), "rb") as f:
            self._obs_cells = pickle.load(f)
        with open(join(MAP_DATA_DIR, CS_CELLS), "rb") as f:
            self._cs_cells = pickle.load(f)
        with open(join(MAP_DATA_DIR, ENB_CELLS), "rb") as f:
            self._enb_cells = pickle.load(f)

    def maps_status(self):
        # Data map made by Points and Cells states (used for plotting):  

        with open(join(MAP_STATUS_DIR, POINTS_STATUS_MATRIX), "rb") as f:
            self._points_status_matrix = pickle.load(f)
        with open(join(MAP_STATUS_DIR, CELLS_STATUS_MATRIX), "rb") as f:
            self._cells_status_matrix = pickle.load(f)
        with open(join(MAP_STATUS_DIR, PERCEIVED_STATUS_MATRIX), "rb") as f:
            self._perceived_status_matrix = pickle.load(f)

    def users_clusters(self):

        with open(join(INITIAL_USERS_DIR, INITIAL_USERS), "rb") as f:
            self._initial_users = pickle.load(f)
        with open(join(INITIAL_USERS_DIR, INITIAL_CENTROIDS), "rb") as f:
            self._initial_centroids = pickle.load(f)
        with open(join(INITIAL_USERS_DIR, INITIAL_CLUSTERS_RADIUSES), "rb") as f:
            self._initial_clusters_radiuses = pickle.load(f)
        with open(join(INITIAL_USERS_DIR, INITIAL_CLUSTERER), "rb") as f:
            self._initial_clusterer = pickle.load(f)

    # Points:
    @property 
    def obs_points(self):
        return self._obs_points

    @property
    def points_matrix(self):
        return self._points_matrix

    @property
    def cs_points(self):
        return self._cs_points

    @property
    def enb_point(self):
        return self._eNB_point

    # Cells:
    @property
    def cells_matrix(self):
        return self._cells_matrix

    @property
    def obs_cells(self):
        return self._obs_cells

    @property
    def cs_cells(self):
        return self._cs_cells    

    @property
    def enb_cells(self):
        return self._enb_cells

    # Status:
    @property
    def points_status_matrix(self):
        return self._points_status_matrix

    @property
    def cells_status_matrix(self):
        return self._cells_status_matrix

    @property
    def perceived_status_matrix(self):
        return self._perceived_status_matrix

    # Users:
    @property
    def initial_users(self):
        return self._initial_users

    @property
    def initial_centroids(self):
        return self._initial_centroids

    @property
    def initial_clusters_radiuses(self):
        return self._initial_clusters_radiuses

    @property
    def initial_clusterer(self):
        return self._initial_clusterer  

class Saver(object):
    '''
    |----------|
    |Save data:|
    |----------|
    '''

    def __init__(self):
        pass

    def maps_data(self, obs_points, points_matrix, cells_matrix, obs_cells, cs_cells, enb_cells, cs_points, eNB_point):
        # Data map made by Points and Cells:

        copyreg.pickle(scenario_objects.Point, scenario_objects.Point.pickle_MyClass)
        
        with open(join(MAP_DATA_DIR, OBS_POINTS_LIST), 'wb') as f:
            pickle.dump(obs_points, f)
        with open(join(MAP_DATA_DIR, CS_POINTS), 'wb') as f:
            pickle.dump(cs_points, f) 
        with open(join(MAP_DATA_DIR, POINTS_MATRIX), 'wb') as f:
            pickle.dump(points_matrix, f)
        with open(join(MAP_DATA_DIR, ENODEB_POINT), 'wb') as f:
            pickle.dump(eNB_point, f)
        
        copyreg.pickle(scenario_objects.Cell, scenario_objects.Cell.pickle_MyClass)

        with open(join(MAP_DATA_DIR, CELLS_MATRIX), 'wb') as f:
            pickle.dump(cells_matrix, f)
        with open(join(MAP_DATA_DIR, OBS_CELLS), 'wb') as f:
            pickle.dump(obs_cells, f)
        with open(join(MAP_DATA_DIR, CS_CELLS), 'wb') as f:
            pickle.dump(cs_cells, f)
        with open(join(MAP_DATA_DIR, ENB_CELLS), 'wb') as f:
            pickle.dump(enb_cells, f)

    def maps_status(self, points_status_matrix, cells_status_matrix, perceived_status_matrix):
        # Data map made by Points and Cells states (used for plotting):

        with open(join(MAP_STATUS_DIR, POINTS_STATUS_MATRIX), 'wb') as f:
            pickle.dump(points_status_matrix, f)
        with open(join(MAP_STATUS_DIR, CELLS_STATUS_MATRIX), 'wb') as f:
            pickle.dump(cells_status_matrix, f)
        with open(join(MAP_STATUS_DIR, PERCEIVED_STATUS_MATRIX), 'wb') as f:
            pickle.dump(perceived_status_matrix, f)

    def users_clusters(self, initial_users_positions, initial_centroids, initial_clusters_radiuses, initial_clusterer):
        # Initial users and centroids:

        copyreg.pickle(scenario_objects.User, scenario_objects.User.pickle_MyClass)

        with open(join(INITIAL_USERS_DIR, INITIAL_USERS), 'wb') as f:
            pickle.dump(initial_users_positions, f)
        with open(join(INITIAL_USERS_DIR, INITIAL_CENTROIDS), 'wb') as f:
            pickle.dump(initial_centroids, f)
        with open(join(INITIAL_USERS_DIR, INITIAL_CLUSTERS_RADIUSES), 'wb') as f:
            pickle.dump(initial_clusters_radiuses, f)
        with open(join(INITIAL_USERS_DIR, INITIAL_CLUSTERER), 'wb') as f:
            pickle.dump(initial_clusterer, f)