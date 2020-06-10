# ENVIRONMENT AND USERS MAIN CLASSES AND METHODS DEFINITION RELATED TO IT.

from random import random, randint
from scipy.stats import truncnorm
import numpy as np
from my_utils import *
from math import floor, pi, cos, sin, log10
from decimal import Decimal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from numpy import linalg as LA
from pylayers.antprop.loss import *
from load_and_save_data import *

class Cell:
    '''
    |------------------------------------------------------------------------|
    |Define the Cell by its state, points included in it and its coordinates:|
    |------------------------------------------------------------------------|
    '''

    def __init__(self, status, points, x_coord, y_coord, z_coord, users):
        self._status = status
        self._points = points
        self._x_coord = x_coord
        self._y_coord = y_coord
        self._z_coord = z_coord
        self._users = users
        
    @property
    def _vector(self):
        return np.array([self._x_coord, self._y_coord, self._z_coord])

    def pickle_MyClass(obj):
        assert type(obj) is Cell
        return scenario_objects.Cell, (obj._status, obj._points, obj._x_coord, obj._y_coord, obj._z_coord, obj._users)

class Point:
    '''
    |---------------------------------------------------------------------|
    |Define the Point (i.e. minimum available map resolution) by its      |
    |states and coordinates:                                              |
    |---------------------------------------------------------------------|
    '''

    def __init__(self, status, x_coord, y_coord, z_coord, users):
        self._status = status
        self._x_coord = x_coord
        self._y_coord = y_coord
        self._z_coord = z_coord
        self._users = users
    
    @property
    def _vector(self):
        self._vector = np.array([x_coord, y_coord, z_coord])

    def pickle_MyClass(obj):
        assert type(obj) is Point
        return scenario_objects.Point, (obj._status, obj._x_coord, obj._y_coord, obj._z_coord, obj._users)

class User:
    '''
    |------------------------------------------------------------------------------------------------------------------|
    |Define the User by his coordinate, maximum heights of the building in which it is (0 if user is not on a building)|
    |and his activity:                                                                                                 |
    |------------------------------------------------------------------------------------------------------------------|
    '''

    def __init__(self, x_coord, y_coord, z_coord, max_in_building, user_account, info):
        self._x_coord = x_coord
        self._y_coord = y_coord
        self._z_coord = z_coord
        self._max_in_building = max_in_building # --> Maximum reachable height for the user which is inside a building (if the user is not inside a building, then this attribute will be obviously equal to 0). 
        self._user_account = user_account
        self._info = info # --> is a list made up by (Bool 'served_or_not', Int 'type_of_service', needed_service_time, elapsed_time_between_request_and_service, served_time).

    def user_info_update(self, users_served_time, user_request_service_elapsed_time):

        if ( (self._info[1] != NO_SERVICE) and (self._info[0] == False) ):
            
            # Increase the elapsed time between the request and the provision of the service. 
            self._info[3] += 1
            # Set to zero the time for which the service is provided.            
            self._info[4] = 0 # --> DA RIVEDERE MEGLIO --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        elif ( (self._info[1] != NO_SERVICE) and (self._info[0] == True) ):

            # Increase the time for which the service is provided. 
            self._info[4] += 1
            # Set to zero the elapsed time between the (next) request and the (next) provision of the service.
            self._info[3] = 0 # --> DA RIVEDERE MEGLIO --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        elif (self._info[2] == self._info[4]):

            # if the requested service time is equal to the provision time of the service, then the user will stop to ask for a service. 
            self._info[1] = NO_SERVICE

        return users_served_time, user_request_service_elapsed_time

    def user_info_update_inf_request(self):
        # For a negligible time instant could happen that a user is not served because the service provision is switching from a UAV to another one, and in this case
        # the info related to the considered user is reset. In any case, the 'QoE' method allows you to track the actual info related to each user by negleting these 'switching service time').

        #global USERS_SERVED_TIME
        #global USERS_REQUEST_SERVICE_ELAPSED_TIME

        #print("ITERATION:", current_iteration)
        #if (current_iteration == (ITERATIONS_PER_EPISODE-1) ):
            #print("FROCCHIOOOOOOOOOOOOOOOOOOOOOOO")
            #USERS_SERVED_TIME += self._info[4]
            #USERS_REQUEST_SERVICE_ELAPSED_TIME += self._info[3]
            #users_served_time += self._info[4]
            #user_request_service_elapsed_time += self._info[3]

            #return users_served_time, user_request_service_elapsed_time

        if (self._info[0] == False):
            
            # Increase the elapsed time between the request and the provision of the service. 
            self._info[3] += 1
            
            # Case in which the user where served at the previous instant:
            #if (self._info[4] != 0):
                #print("PEGOLOOOOOOOOO")
                #USERS_SERVED_TIME += self._info[4]
                #users_served_time += self._info[4]
                #print("ALLLLAAAAAAAAAAAAAAAAAAAAAAaaaa", users_served_time)
            
            # Set to zero the time for which the service is provided.
            self._info[4] = 0
        
        elif (self._info[0] == True):

            # Increase the time for which the service is provided (without interruptions). 
            self._info[4] += 1
            
            # Case in which the user where not served at the previous instant: 
            #if (self._info[3] != 0):
                #print("PEGOLO2222222222222")
                #USERS_REQUEST_SERVICE_ELAPSED_TIME += self._info[3]
                #user_request_service_elapsed_time += self._info[3]
            
            # Set to zero the elapsed time between the (next) request and the (next) provision of the service.
            self._info[3] = 0

        #return users_served_time, user_request_service_elapsed_time

    @staticmethod
    def QoE(users):

        QoE1 = 0
        QoE2 = 0
        for user in users:
            if user._info[3] > 0:
                QoE2 += 1
            if user._info[4] > 0:
                QoE1 += 1

        return QoE1, QoE2

    @staticmethod
    def avg_QoE(current_epoch, users_served_per_epoch, service_request_per_epoch, users_request_service_elapsed_time, avg_QoE1_per_epoch_list, avg_QoE2_per_epoch_list):

        avg_QoE1_per_epoch_list[current_epoch-1] = users_served_per_epoch/service_request_per_epoch # --> Start from the first epoch, without considering the obviously 0 values for the instant immediately before the first epoch.
        avg_QoE2_per_epoch_list[current_epoch-1] = users_request_service_elapsed_time # --> Start from the first epoch, without considering the obviously 0 values for the instant immediately before the first epoch.

        #return avg_QoE1_per_epoch, avg_QoE2_per_epoch

    @staticmethod
    def get_truncated_normal(mean, std, lower_bound, upper_bound, n_users):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Truncated normal distribution with bounds 'lower_bound' and 'upper_bounds'. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        upper_bound -= 1
        # 'rvs' return random variates of a given type
        return truncnorm((lower_bound - mean) / std, (upper_bound - mean) / std, loc=mean, scale=std).rvs(size=(n_users))

    @staticmethod
    def centroids_user_cluster_generation(centroid_min_x, centroid_max_x, centroid_min_y, centroid_max_y, clusters_num):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Compute initial users clusters by drawing samples from a uniform distribution.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        centroids = np.random.uniform(low=[centroid_min_x, centroid_min_y], high=[centroid_max_x, centroid_max_y], size=(clusters_num, 2))
        '''
        variance = pow(std, 2)
        covariance_matrix = [[variance, 0], [0, variance]]
        mean = [mean, mean]

        centroids = np.random.multivariate_normal(mean, covariance_matrix, clusters_num)
        '''
        return centroids

    @staticmethod
    def spread_users_around_clusters(centroids, mean_x, mean_y, std_x, std_y, min_users_per_cluster, max_users_per_cluster):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Returns the users coordinates (and its clusters) computed along the generated centroids clusters by using a truncated (bounded) #
        # the normal distribution; in such a way it is possible to avoid to place the users outside of the area of interest.              #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        #variance = pow(std, 2)
        #covariance_matrix = [[variance, 0], [0, variance]]
        users_clusters = []
        users_xy = []

        for centroid in centroids:

            mean = centroid
            users_for_current_cluster = randint(min_users_per_cluster, max_users_per_cluster)
            users_x_coords = User.get_truncated_normal(mean[0], std_x, 0, AREA_WIDTH, users_for_current_cluster)
            users_y_coords = User.get_truncated_normal(mean[1], std_y, 0, AREA_HEIGHT, users_for_current_cluster)
            # Round users coordinates to the first decimal digit:
            current_cluster = [(round(Decimal(users_x_coords[idx]), 1), round(Decimal(users_y_coords[idx]), 1)) for idx in range(users_for_current_cluster)]
            for i in current_cluster:
                print((type(i[0]), type(i[1])))
            #current_cluster = np.random.multivariate_normal(mean, covariance_matrix, users_for_current_cluster)
            users_clusters.append(current_cluster)

        for cluster in users_clusters:
            for user in cluster:
                users_xy.append(user)
        #print(len(users_xy))
        #print(users_xy)

        return users_clusters, users_xy

    @staticmethod
    def users_heights(point_or_cell_matrix, users_xy):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Users 'z' coordinates are computed by the “discrete uniform” distribution in a closed interval; #
        # if 'point_or_cell' in which the current users is placed has no building inside, then            #
        # che closed interval will be (0, 0].                                                             #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        users_heights = []
        for user in users_xy:

            current_point_or_cell_matrix = point_or_cell_matrix[floor(user[1])][floor(user[0])]

            if (current_point_or_cell_matrix._status == OBS_IN):
                max_achievable_height_per_user = min(current_point_or_cell_matrix._z_coord, MAX_HEIGHT_PER_USER) # The Maximum height which can be reached by the users is the minimum between the 'MAX_HEIGHT_PER_USER' and the height of the building inside which they could be.
                us_height = np.random.random_integers(0, max_achievable_height_per_user)
            else:
                us_height = 0

            users_heights.append(us_height)

        #us_heights = np.random.random_integers(0, point_or_cell._z_coord)
        return users_heights

    @staticmethod
    def max_reachable_height_per_user(point_or_cell_matrix, user_xy):

        point_or_cell = point_or_cell_matrix[floor(user_xy[1])][floor(user_xy[0])]

        if (point_or_cell._status == OBS_IN):
            max_achievable_height_per_user = min(point_or_cell._z_coord, MAX_HEIGHT_PER_USER) # The Maximum height which can be reached by the users is the minimum between the 'MAX_HEIGHT_PER_USER' and the height of the building inside which they could be.
        else:
            max_achievable_height_per_user = 0

        return max_achievable_height_per_user

    @staticmethod
    def create_users(points_matrix, users_xy, users_z, n_users):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Initialize the users, set their 'z' coordinates and activity  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        users_xyz = []
        for user_idx in range(n_users):
            type_of_service = User.which_service()
            requested_service_life = User.needed_service_life(type_of_service)
            current_user_xy = users_xy[user_idx]
            current_user_z = users_z[user_idx]
            max_height_per_current_user = User.max_reachable_height_per_user(points_matrix, current_user_xy)
            print("MAX HEIGHT PER CURRENT USER:", max_height_per_current_user)
            if (INF_REQUEST == True):
                users_xyz.append( User(current_user_xy[0], current_user_xy[1], current_user_z, max_height_per_current_user, User.generate_user_account(), [False, None, 1, 0, 0]) )
            else:
                users_xyz.append( User(current_user_xy[0], current_user_xy[1], current_user_z, max_height_per_current_user, User.generate_user_account(), [False, type_of_service, requested_service_life, 0, 0]) )            

        #users_xyz = [User(users_xy[idx][0], users_xy[idx][1], users_z[idx], users_z[idx], User.generate_user_account(), (None, User.which_service(), np.random.choice(NEEDED_SERVICE_TIMES_PER_USER), None)) for idx in range(n_users)]

        return users_xyz

    @staticmethod
    def k_random_walk(origins, n_steps):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Generate 'k' random walks for 'n_steps' by starting from 'origins'.                             #
        # It returns the steps to be taken from each of the 'origins'; the                                #
        # generated walk is bounded in such a way to not allow users to go outside the area of interest.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Possible actions for walking at each step: 
        go_back = -1
        stop = 0
        go_ahead = +1

        # Upper bounds index of our area of interest:
        upper_x = AREA_WIDTH-1
        upper_y = AREA_HEIGHT-1

        users_steps = []
        for user in origins:
            current_user_steps = []

            for stps in range(n_steps):
                
                current_user_x = user._x_coord
                current_user_y = user._y_coord
                current_user_z = user._z_coord
                user_max_in_building = user._max_in_building

                x_val = np.random.random_integers(go_back, go_ahead)
                y_val = np.random.random_integers(go_back, go_ahead)
                   
                # Case in which the user is on a floor of a building which is between the first and the top floor (the latter excluded): 
                if ( (current_user_z > 0) and (current_user_z < user_max_in_building) ):
                    z_val = np.random.random_integers(go_back, go_ahead)
                    x_val = stop
                    y_val = stop

                # Case in which the user is on the top floor of a building:
                elif (current_user_z == user_max_in_building):
                    z_val = np.random.random_integers(go_back, stop)
                    x_val = stop
                    y_val = stop

                # Case in which the user is on the ground (or ground floor):
                else:

                    if (user_max_in_building > 0):
                        
                        x_val = np.random.random_integers(go_back, go_ahead)
                        y_val = np.random.random_integers(go_back, go_ahead)
                        if ( (x_val == stop) and (y_val == stop) ):
                            z_val = np.random.random_integers(stop, go_ahead)
                        else:
                            z_val = stop 
                    else:

                        if (current_user_x == 0):
                            x_val = np.random.random_integers(stop, go_ahead)
                        if (current_user_y == 0):
                            y_val = np.random.random_integers(stop, go_ahead)
                        if (current_user_x == upper_x):
                            x_val = np.random.random_integers(go_back, stop)
                        if (current_user_x == upper_y):
                            y_val = np.random.random_integers(go_back, stop)

                        z_val = stop

                current_user_steps.append((current_user_x + x_val, current_user_x + y_val, current_user_z + z_val))

            users_steps.append(current_user_steps)

        return users_steps

    @staticmethod
    def users_per_cluster_per_timeslot(min_users_per_day, max_users_per_day, num_clusters):

        max_n_peak_users_per_day = np.random.random_integers(min_users_per_day, max_users_per_day)

        daily_users_per_cluster_per_timeslot = [[] for time_slot in range(HOURS_PER_CONSIDERED_TIME)]
        current_hour = 0

        for macro_time_slot in range(MACRO_TIME_SLOTS):
            
            for micro_time_slot in range(MICRO_SLOTS_PER_MACRO_TIMESLOT):

                for cluster in range(num_clusters):

                    daily_users_per_cluster_per_timeslot[current_hour].append( int(round(np.random.uniform(MIN_MAX_USERS_PERCENTAGES[macro_time_slot][0], MIN_MAX_USERS_PERCENTAGES[macro_time_slot][1]) * max_n_peak_users_per_day)) )
                    current_hour += 1
        '''
        n_users_timeslot1 = round(np.random.uniform(MIN_USERS_PERCENTAGE1, MAX_USERS_PERCENTAGE1) * n_users_per_day)
        n_users_timeslot2 = round(np.random.uniform(MIN_USERS_PERCENTAGE2, MAX_USERS_PERCENTAGE2) * n_users_per_day)
        n_users_timeslot3 = round(np.random.uniform(MIN_USERS_PERCENTAGE3, MAX_USERS_PERCENTAGE3) * n_users_per_day)
        n_users_timeslot4 = round(np.random.uniform(MIN_USERS_PERCENTAGE4, MAX_USERS_PERCENTAGE4) * n_users_per_day)
        n_users_timeslot5 = round(np.random.uniform(MIN_USERS_PERCENTAGE5, MAX_USERS_PERCENTAGE5) * n_users_per_day)
        n_users_timeslot6 = round(np.random.uniform(MIN_USERS_PERCENTAGE6, MAX_USERS_PERCENTAGE6) * n_users_per_day)

        daily_users_per_timeslot = [n_users_timeslot1, n_users_timeslot2, n_users_timeslot3, n_users_timeslot4, n_users_timeslot5, n_users_timeslot6]
        '''

        print(daily_users_per_cluster_per_timeslot)
        return daily_users_per_cluster_per_timeslot

    @staticmethod
    def user_service_time(user):

        user._info[2] = np.random.choice(NEEDED_SERVICE_TIMES_PER_USER)

    @staticmethod
    def compute_clusterer(users, fixed_clusters=True):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Compute the clusterer which will be used then to detect users centroids once users have started to walk.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Use 'np.array' for User objects:
        users_array = np.array([[user._x_coord, user._y_coord, user._z_coord] for user in users])

        # Case in which we have set a priori the number of cluster that we want to use to group the users:
        if fixed_clusters == True:
            clusterer = KMeans(FIXED_CLUSTERS_NUM)
            clusterer.fit(users_array)
            users_clusters = [users_array[np.where(clusterer.labels_ == i)] for i in range(clusterer.n_clusters)]
        
        # Case in which we want to find the optimal number of clusters among a list of chosen numbers of clusters:
        else:
            optimal_clusters_num, optimal_clusterer, current_best_silhoutte_score = None, None, 1.0            
            
            for current_cluster_num in CLUSTERS_NUM_TO_TEST:
                current_clusterer = Kmeans(current_cluster_num)
                cluster_labels = current_clusterer.fit_predict(users_array)
                # Use the silhouette score evaluates the 'goodness' of the current number of clusters considered: 
                silhouette_avg = silhouette_score(users_array, cluster_labels)
                
                if (silhouette_avg < current_best_silhoutte_score):
                    current_best_silhoutte_score = silhouette_avg
                    optimal_clusters_num = current_cluster_num
                    optimal_clusterer = current_clusterer

            users_clusters = [users_array[np.where(clusterer.labels_ == i)] for i in range(clusterer.n_clusters)]
            optimal_clusters_num.fit(users_array)

            return optimal_clusterer, users_clusters, optimal_clusters_num, current_best_silhoutte_score

        return clusterer

    @staticmethod
    def actual_users_clusters_centroids(clusterer):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Compute the centroids clusters once the users have started to walk. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        #centroids = [[sum(cluster[cluster_idx][0])/users_per_cluster[cluster_idx], sum(cluster[cluster_idx][1])/users_per_cluster[cluster_idx]] for cluster_idx in num_clusters_range]
        #centroids_users_distances = [LA.norm(np.array([cluster[cluster_idx][0], cluster[cluster_idx][1]]) - np.array(centroids[cluster_idx])) for cluster_idx in num_clusters_range]
        #clusters_radiuses = [max(centroids_user_dist) for centroids_user_dist in centroids_users_distances]
        centroids = clusterer.cluster_centers_
        return centroids

    @staticmethod
    def actual_clusters_radiuses(centroids, users_clusters, num_clusters):

        users_per_cluster = [len(cluster) for cluster in users_clusters]
        clusters_radiuses = [None for cluster in users_clusters]
        #centroids_users_distances = []
        #num_clusters = len(clusters) 
        #num_clusters_range = range(num_clusters)
        print("oooooooooooo")
        print(users_clusters[0])
        print(users_per_cluster)
        decimal_centroids = [[Decimal(centroid[0]), Decimal(centroid[1]), Decimal(centroid[2])] for centroid in centroids]
        #print(centroids[0])
        #print(np.array([users_clusters[0]]))
        for cluster_idx in range(num_clusters):
            a = 0
            #u = LA.norm(np.array([decimal_centroids[cluster_idx][0], decimal_centroids[cluster_idx][1]]) - np.array([users_clusters[cluster_idx][user_idx][0], users_clusters[cluster_idx][user_idx][0]]))
            current_centroid_users_distances = [LA.norm(np.array([decimal_centroids[cluster_idx][0], decimal_centroids[cluster_idx][1]]) - np.array([users_clusters[cluster_idx][user_idx][0], users_clusters[cluster_idx][user_idx][1]])) for user_idx in range(0, users_per_cluster[cluster_idx])]
            #print(len(current_centroid_users_distances))
            clusters_radiuses[cluster_idx] = round(max(current_centroid_users_distances), 1)
            #centroids_users_distances.append(current_centroid_users_distances)
            #print(len(centroids_users_distances[cluster_idx]))
        '''
        for centroid in decimal_centroids:
            for users_per_cluster in users_clusters:
                current_centroid_users_distances = [LA.norm(np.array([centroid[0], centroid[1]]) - np.array([user[0], user[1]])) for user in users_per_cluster]  
            centroids_users_distances.append(current_centroid_users_distances)
            #centroids_usr_distances = [[LA.norm( np.array([centroid[0], centroid[1]]) - np.array(cluster[cluster_idx])     
        '''
        #centroids_usr_distances = [[LA.norm( np.array([centroid[0], centroid[1]]) - np.array([usr_per_cluster[0], usr_per_cluster[1]])) for usr_per_cluster in users_per_cluster] for centroid in decimal_centroids]
        #print("DIST")
        #print(len(centroids_users_distances[0]),len(centroids_users_distances[1]),len(centroids_users_distances[2]))
        #print("CLUSTERS LENS")
        #print(users_per_cluster[0], users_per_cluster[1], users_per_cluster[2])
        print("Radiuses")
        print(clusters_radiuses)

        return clusters_radiuses

    @staticmethod
    def which_service():

        service = np.random.choice(UAVS_SERVICES)
        return service

    @staticmethod
    def needed_service_life(service_to_provide):

        if (service_to_provide == THROUGHPUT_REQUEST):
            requested_service_life = np.random.choice(TR_SERVICE_TIMES)
        elif (service_to_provide == EDGE_COMPUTING):
            requested_service_life = EC_SERVICE_TIME
        elif (service_to_provide == DATA_GATHERING):
            requested_service_life = DG_SERVICE_TIME

        requested_service_life = np.random.choice(TR_SERVICE_TIMES)
        return requested_service_life

    @staticmethod
    def generate_user_account():
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Generate a specific type of user account among the 4 available by choosing each of them #
        # with a specific probability distribution indicated by USERS_ACCOUNTS_DITRIBUTIONS.      #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        user_account = np.random.choice([FREE_USER, BASE_USER, FULL_USER, PREMIUM_USER], p=USERS_ACCOUNTS_DITRIBUTIONS)
        return user_account

    @staticmethod
    def bitrate_request():
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Generate a random throughput request among the 'available' ones in 'TRHOUGHPUT_REQUESTS'. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        bt_request = np.random.choice(TRHOUGHPUT_REQUESTS)
        return bt_request

    @staticmethod
    def edge_computing_request():

        # TO DO . . .
        ec_request = np.random.choice(EDGE_COMPUTING_REQUESTS)
        return ec_request

    @staticmethod
    def data_gathering():

        # TO DO . . .
        dg_request = np.random.choice(DATA_GATHERING_REQUESTS)
        return dg_request

    def pickle_MyClass(obj):
        assert type(obj) is User
        return scenario_objects.User, (obj._x_coord, obj._y_coord, obj._z_coord, obj._max_in_building, obj._user_account, obj._info)

class Environment:
    # |-----------------------------------------------------------------------------|
    # |Define the Environment by specifing the width and the height of the selected |
    # |shape for the area and the desired resolution cell for the width and height  |
    # |which have been chosen (resolution in this case means the number of cell):   |
    # |-----------------------------------------------------------------------------|
    
    def __init__(self, area_width, area_height, area_z, cell_res_row, cell_res_col):
        self._area_width = area_width
        self._area_height = area_height
        self._area_z = MAXIMUM_AREA_HEIGHT
        self._N_points = area_width*area_height
        self._cell_res_row = cell_res_row
        self._cell_res_col = cell_res_col
        self._cells_rows = CELLS_ROWS
        self._cells_cols = CELLS_COLS
        self._cells_num = N_CELLS
        self._cs_height = CS_HEIGHT
        self._cs_num = N_CS
        #self._cs_distance = RADIAL_DISTANCE
        self._radial_distance_x = RADIAL_DISTANCE_X
        self._radial_distance_y = RADIAL_DISTANCE_Y        
        self._x_eNB = ENODEB_X
        self._y_eNB = ENODEB_Y
        self._z_eNB = ENODEB_Z
        self._free = FREE
        self._obs_in = OBS_IN
        self._cs_in = CS_IN
        self._uav_in = UAV_IN
        self._enb_in = ENB_IN
   
    def neighbors_elems_pos(self, elem, matrix_row_upper_bound, matrix_col_upper_bound, obs_points):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Check if the 8 elements (i.e. Points) close to 'elem' are occupied by an obstacle,        #
        # namely if they are in 'obs_points'; if it is not so it returns the considered element,    #
        # otherwise it continues by selecting 8 elements at a time around the one considered until  #
        # reaching the bounds of the matrix. NEIGHBOR_ELEMS_POS performs also SIDE-EFFECT on        #
        # 'obs_points' by expanding it with the selected element.                                   #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        matrix_row_upper_bound -= 1
        matrix_col_upper_bound -= 1

        current_row = elem[0]
        current_col = elem[1]
        z_coord = elem[2]
        
        prev_row = current_row - 1
        prev_col = current_col - 1
        next_row = current_row + 1
        next_col = current_col + 1

        candidate_found = False

        selected_position = None

        while (not candidate_found):

            if (prev_row < LOWER_BOUNDS):
                prev_row = LOWER_BOUNDS
            if (next_row > matrix_row_upper_bound):
                next_row = matrix_row_upper_bound
            if (prev_col < LOWER_BOUNDS):
                prev_col = LOWER_BOUNDS
            if (next_col > matrix_col_upper_bound):
                next_col = matrix_col_upper_bound

            for row in range(prev_row, next_row+1):
                if (candidate_found == True):
                    break

                for col in (prev_col, next_col+1):
                    if (candidate_found == True):
                        break

                    candidate_position = (row, col, z_coord)
                    
                    if (candidate_position not in obs_points):
                        selected_position = candidate_position
                        # SIDE-EFFECT on 'obs_points'
                        obs_points.append(selected_position)
                        candidate_found = True
                        break

            prev_row -= 1
            prev_col -= 1
            next_row += 1
            next_col += 1

            return selected_position

    # QUESTO METODO POTREBBE ESSERE ELIMINATO (E' RIDONDANTE) --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def set_users_on_map(self, points_matrix, users_xyz, users):
        # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Set 'users' attributes of 'points_matrix' points. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # #

        for user_xy in users_xy:
            points_matrix[floor(user_xy[1])][floor(user_xy[0])]._users.append()  

        return None


    def obstacles_generation(self, min_obs_per_area=MIN_OBS_PER_AREA, max_obs_per_area=MAX_OBS_PER_AREA, min_obs_height=MIN_OBS_HEIGHT, max_obs_height=MAX_OBS_HEIGHT):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Generate obstacles by computing random integers from the “discrete uniform” distribution in the closed              #
        # interval specified by the width and the height of the selected shape for the considered area. It is possible to     #
        # specify the minimum and maximum percentages of the area which is expected to be covered by obstacles as well as it  #
        # is possible to select the minimum and maximum height for each generated obstacle. OBSTACLES_GENERATION returns      #
        # a list consisting of tuples (x,y,z) on which obstacles have been generated.                                         #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Percentage of area to be covered by obstacles:
        obs_per_area = randint(min_obs_per_area*100, max_obs_per_area*100)/100

        # Number of Points to be covered by obstacles:
        N_obs_points = int(round(self._area_width*self._area_height*obs_per_area))
        reachable_width_idx = self._area_width - 1
        reachable_height_idx = self._area_height - 1

        x_random_matrix = []
        x_random = np.random.random_integers(0, reachable_width_idx, size=N_obs_points)
        #x_random_matrix = [round(point) for point in x_random]
        
        y_random = np.random.random_integers(0, reachable_height_idx, size=N_obs_points)
        #y_random_matrix = [round(point) for point in y_random]

        z_random = np.random.random_integers(min_obs_height, max_obs_height, size=N_obs_points)
        #z_random_matrix = [round(point) for point in z_random]        

        xyz_obs_points = [(x_random[idx], y_random[idx], z_random[idx]) for idx in range(N_obs_points)]
        print(len(xyz_obs_points), len(list(set(xyz_obs_points))))
 
        xyz_obs_points_occurences = [xyz_obs_points.count(xyz_obs) for xyz_obs in xyz_obs_points]

        # 'xyz_obs_points' could contains duplicates and for they have to be removed and
        # replaced by another Point which is not among the ones in 'xyz_obs_points' and which is found
        # by using NEIGHBOR_ELEMS_POS: 
        if (len(set(xyz_obs_points_occurences)) > 1):

            #xyz_obs_duplicates = [point for point in xyz_obs_points if xyz_obs_points.count(point)>1]

            xyz_obs_duplicates_temp = [(point if xyz_obs_points.count(point)>1 else None) for point in xyz_obs_points]
            xyz_obs_duplicates = list(filter(lambda a: a != None, xyz_obs_duplicates_temp))
            print("DUPLICATES IN OBSATCLES POINTS: ", len(xyz_obs_duplicates))
            
            # Obstacles Points duplicates wihtouth repetitions:
            xyz_obs_points_duplicates_no_repetitions = list(set(xyz_obs_duplicates))
            
            # Obstacles Points without duplicates:
            xyz_obs_points_no_duplicates = list(set(xyz_obs_points))
            print(len(xyz_obs_points))
            print("OBSTACLES POINTS WITHOUT DUPLICATES: ", len(xyz_obs_points_no_duplicates))
            
            # Removing from 'xyz_obs_duplicates' all the obstacles Points for which there is at least a duplicate: 
            [xyz_obs_duplicates.remove(elem) for elem in xyz_obs_points_duplicates_no_repetitions]
            
            # Computing neighbors points w.r.t the duplicates obstacles Points:
            neighbors_xyz_coords = [self.neighbors_elems_pos(point, self._area_height, self._area_width, xyz_obs_points_duplicates_no_repetitions) for point in xyz_obs_duplicates]
            print("NEIGHBORS: ", len(neighbors_xyz_coords))
            
            xyz_obs_points = list(np.concatenate((xyz_obs_points_no_duplicates, neighbors_xyz_coords)))
            print("OBSTACLES POINTS: ", len(xyz_obs_points))

            obs_points = [Point(self._obs_in, xyz_obs_points[idx][0], xyz_obs_points[idx][1], xyz_obs_points[idx][2], []) for idx in range(N_obs_points)]
        
        else:

            obs_points = [Point(self._obs_in, x_random[idx], y_random[idx], z_random[idx], []) for idx in range(N_obs_points)]

        return obs_points

    def set_obstacles_on_map(self, obs_points):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Initializing the map (i.e. a matrix) by setting every element equal to a Point        #
        # either containings an obstacle (if the matrix elem [i][j] is equal to a Point         #
        # whith coordinates [x][y] in 'obs_points') or containings nothing, i.e. a FREE Point.  #
        # SET_OBSTACLES_ON_MAP return a matrix made by Point elements.                          #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        points_matrix = [[Point(0, j, i, 0, []) for j in range(self._area_width)] for i in range(self._area_height)]
        
        for obs_point in obs_points:
            x_current_obs = obs_point._x_coord
            y_current_obs = obs_point._y_coord
            points_matrix[y_current_obs][x_current_obs] = obs_point

        return points_matrix

    def set_CS_on_map(self, points_matrix, obs_points):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Set charging stations on map (represented by 'points_matrix') #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        rad_between_points = 2*pi/self._cs_num
        CS_points = []
        #print(self._cs_distance)

        [CS_points.append( Point(self._cs_in, round(self._x_eNB + self._radial_distance_x*sin(CS_idx*rad_between_points)), round(self._y_eNB + self._radial_distance_y*cos(CS_idx*rad_between_points)), self._cs_height, []) ) for CS_idx in range(self._cs_num)]
        #print( (self._x_eNB, self._y_eNB) )
        for CS_point in CS_points:
            x_current_CS = CS_point._x_coord
            y_current_CS = CS_point._y_coord

            print( (x_current_CS, y_current_CS) )

            current_position_on_map_status = points_matrix[y_current_CS][x_current_CS]._status

            # SIDE-EFFECT on 'points_matrix':

            # If the selected map position is FREE, then that position will be occupied by a CS.
            if (current_position_on_map_status == FREE):
                #print( (current_position_on_map._x_coord, current_position_on_map._y_coord, current_position_on_map._z_coord, current_position_on_map._status) )
                points_matrix[y_current_CS][x_current_CS] = CS_point
                #print( (current_position_on_map._x_coord, current_position_on_map._y_coord, current_position_on_map._z_coord, current_position_on_map._status) )
            # Otherwise, namely if an obstacle is in here, the CS will be placed on that obstacle with by keeping the height of the obstacle;
            # In this case the obstacle will be removed from 'obs_points' list.
            elif (current_position_on_map_status == OBS_IN):

                points_matrix[y_current_CS][x_current_CS]._status = CS_IN
                #print("eccolo")

                for obs in obs_points:
                    if ( (obs._x_coord == x_current_CS) and (obs._y_coord == y_current_CS) ):
                        obs_points.remove(obs) # SIDE-EFFECT on 'obs_points'
                        break
        
        return CS_points

    def set_eNB_on_map(self, points_matrix, obs_points):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Set eNodeB on map (represented by 'points_matrix'). #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        if (points_matrix[self._y_eNB][self._x_eNB]._status == FREE):
            z_enb = self._z_eNB
        # If the desired position for the eNodeB is occupied by an obstacle, then
        # use the height of the obstacle as eNodeB height:
        elif (points_matrix[self._y_eNB][self._x_eNB]._status == OBS_IN):
            z_enb = points_matrix[self._y_eNB][self._x_eNB]._z_coord

        for obs in obs_points:
            if ( (obs._x_coord == self._x_eNB) and (obs._y_coord == self._y_eNB) ):
                obs_points.remove(obs) # SIDE-EFFECT on 'obs_points'
                break

        eNB_point = []
        eNB_point.append(Point(self._enb_in, self._x_eNB, self._y_eNB, z_enb, []) )

        if (CREATE_ENODEB == True):
            points_matrix[self._y_eNB][self._x_eNB] = eNB_point[0]
        # Otherwise eNodeB will not set on map, but its coordinates will be used to spread the Charging Stations around the centre of the map. 

        return eNB_point

    def compute_cell_matrix(self, points_matrix, actual_z_enb, area_width, area_height, cell_res_row, cell_res_col):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Similarly to SET_OBSTACLES_ON_MAP, COMPUTE_CELL_MATRIX returns a matrix made by Cell elements;    #
        # height an width cell resolutions have to be chosen and every cell which contains an obstacle      #
        # will have on obstacle height (i.e. 'Cell._z_coord') equal to the maximum height of the obstacles  #
        # contained in it.                                                                                  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        cells = []

        cell_idx = 0
        for i in range(0, self._area_height, cell_res_row):
            for j in range(0, self._area_width, cell_res_col):
                current_cell = [submatrix[j:j+cell_res_row] for submatrix in points_matrix[i:i+cell_res_col]]
                cells.append(list(np.array(current_cell).flatten()))

                cell_idx += 1

        cells_matrix = [[None for col in range(self._cells_cols)] for row in range(self._cells_rows)]

        cell_idx = 0
        for i in range(self._cells_rows):
            for j in range(self._cells_cols):
                current_points = cells[cell_idx]
                
                status_points = [point._status for point in current_points]
                
                if (self._cs_in in status_points):
                    # Set the height of the cell equal to the possible CS contained in the current cell:
                    current_cell_status = self._cs_in
                    z_current_cell = self._cs_height
                elif (self._enb_in in status_points):
                    current_cell_status = self._enb_in
                    z_current_cell = actual_z_enb
                else:
                    # Select the maximum height among the ones which are present in the
                    # points inside the considered cell: 
                    z_current_cell = max([point._z_coord for point in current_points])
                    if (z_current_cell > 0):
                        current_cell_status = self._obs_in
                    else:
                        current_cell_status = self._free

                cells_matrix[i][j] = Cell(current_cell_status, current_points, j, i, z_current_cell, [])
                cell_idx += 1

        return cells_matrix

    def extracting_specific_cells_coordinates(self, cells_matrix, which_cell_type):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Extracting cell coordinates from 'cells_matrix' which contain 'which_cell_type';  #
        # this function is used to shape the cell in such a way to be plotted.              #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        #desired_cells = [[cells_matrix[i][j] for j in range(CELLS_COLS) if cells_matrix[i][j]._status == which_cell_type] for i in range(CELLS_ROWS)]

        desired_cells = [[(cells_matrix[i][j] if cells_matrix[i][j]._status == which_cell_type else None) for j in range(CELLS_COLS)] for i in range(CELLS_ROWS)]
        desired_cells = list(np.array(desired_cells).flatten())
        desired_cells = list(filter(lambda a: a != None, desired_cells))

        return desired_cells

    def pickle_MyClass(obj):
        assert type(obj) is Environment
        return scenario_objects.Environment, ('static_env')

if __name__ == '__main__':

    env = Environment(AREA_WIDTH, AREA_HEIGHT, MAXIMUM_AREA_HEIGHT, CELL_RESOLUTION_PER_ROW, CELL_RESOLUTION_PER_COL)
    obs_points = env.obstacles_generation()
    points_matrix = env.set_obstacles_on_map(obs_points)
    CS_points = env.set_CS_on_map(points_matrix, obs_points)
    eNB_point = env.set_eNB_on_map(points_matrix, obs_points)  
    #for point in CS_points:
    #    print( (point._x_coord, point._y_coord) )

    #print(points_matrix[0][0]._users)
    #users = [num for num in range(1, 10) if num not in points_matrix[0][0]._users] #points_matrix[0][0]._users.append("prova")
    # points_matrix[0][0]._users = users
    # print(print(points_matrix[0][0]._users))
    us = User
    centroids = us.centroids_user_cluster_generation(3, 3, 8, 8, FIXED_CLUSTERS_NUM) # --> CAMBIA I VALORI DI QUESTI ARGOMENTI PER OTTENERE CLUSTERS SPARSI IN MODO DIVERSO --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print("CENTROIDIIIII", centroids)
    users_clusters, users_xy = us.spread_users_around_clusters(centroids, 1.0, 1.0, 1.0, 1.0, 8, 16) # --> CAMBIA I VALORI DI QUESTI ARGOMENTI PER OTTENERE CLUSTERS SPARSI IN MODO DIVERSO --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    occurrences = [users_xy.count(user) for user in users_xy]
    #print(users_xy)
    #env.set_users_on_map(points_matrix, users_xy)
    #users_points_heights = []
    users_points_heights = us.users_heights(points_matrix, users_xy)
    '''
    for i in range(env._area_height):
        for j in range(env._area_width):
            current_point = points_matrix[i][j]
            if current_point._users != []:
                users_points_heights.append(env.users_heights(points_matrix[i][j]))
    '''
    n_users = len(users_xy)
    print(n_users)
    #print(len(users_points_heights))
    #print(len(users_xy))
    users_xyz = us.create_users(points_matrix, users_xy, users_points_heights, n_users)
    #print("Z USERS COORDS")
    #[print(user._z_coord) for user in users_xyz]
    print("XY USERS COORDS")
    [print((user._x_coord, user._y_coord)) for user in users_xyz]
    users_steps = us.k_random_walk(users_xyz, 10)
    initial_clusterer = us.compute_clusterer(users_xyz)
    initial_centroids = us.actual_users_clusters_centroids(initial_clusterer)
    initial_clusters_radiuses = us.actual_clusters_radiuses(initial_centroids, users_clusters, FIXED_CLUSTERS_NUM) # --> qui ovviamente poi varierai il numero di clusters a seconda che tu li voglia identificare in modo dinamico o meno -> !!!!!!!!!!!!!!!!!!!!!!!!!!
    print("CENTROIDS")
    print(initial_centroids)
    #print(users_steps)
    #print(users_steps)
    #print()
    #print(len(users_steps))
    #print()
    #print(len(users_steps[0]))
    #print(len(users_steps[0][0]))
    #print()
    #print(len(users_steps[1]))
    #print(len(users_steps[1][0]))
    #users_xyz = [User(users_xy[idx][0], users_xy[idx][1], users_points_heights[idx], 0) for idx in range(n_users)]
    #print(len(users_xyz))
    #print(len(users_points_heights))
    #print(len(users_xy))

    cells_matrix = env.compute_cell_matrix(points_matrix, eNB_point[0]._z_coord, env._area_width, env._area_height, env._cell_res_row, env._cell_res_col)
    print(len(cells_matrix), len(cells_matrix[0]))
    
    '''
    print("CELLS")
    for pnts in cells_matrix[4][0]._points:
        print((pnts._x_coord, pnts._y_coord))
    '''

    # Extracting cells coordinates which contain obstacles:
    obs_cells = env.extracting_specific_cells_coordinates(cells_matrix, env._obs_in)
    # Extracting cells coordinates which charging stations:
    cs_cells = env.extracting_specific_cells_coordinates(cells_matrix, env._cs_in)
    # Extracting cells coordinates which contain eNodeB:
    eNB_cells = env.extracting_specific_cells_coordinates(cells_matrix, env._enb_in)


    # Create the directories 'map_data' and 'initial_users' to save data:
    directory = Directories() 
    directory.create_map_data_dir()
    directory.create_users_clusters_dir()

    # Saving:
    saver = Saver()
    saver.maps_data(obs_points, points_matrix, cells_matrix, obs_cells, cs_cells, eNB_cells, CS_points, eNB_point)
    saver.users_clusters(users_xyz, initial_centroids, initial_clusters_radiuses, initial_clusterer)