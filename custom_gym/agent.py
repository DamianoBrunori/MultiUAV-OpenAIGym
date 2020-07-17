# AGENT MAIN CLASSES AND METHODS DEFINITION RELATED TO IT.

# CENTROIDI, CLUSTER, UTENTI (e altro . . . ?) VA AGIORNATO AD OGNI ITERAZIONE --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

from my_utils import *
from scenario_objects import Cell, Point, Environment, User
from load_and_save_data import Loader
#import operator
import numpy as np
from math import tan, radians, ceil
from numpy import linalg as LA
import copy

load = Loader()
load.maps_data()
obs_cells = load.obs_cells

MAX_OBS_CELLS = max(obs._z_coord for obs in obs_cells) if DIMENSION_2D==False else 0
MAX_UAV_HEIGHT = int(MAX_OBS_CELLS)

class Agent:
    '''
    |--------------------------------------------------------------------------------------|
    |Define the agent by its coordinates, occupied cell, performing action and its distance|
    |from charging stations and users clusters.                                            |
    |--------------------------------------------------------------------------------------|
    '''

    def __init__(self, pos, ID, toward, action, bandwidth, battery_level, footprint, max_uav_height, action_set, TR, EC, DG, d_ag_cc):
        #self._cell = cell # DA VEDERE SE INSERIRLO OPPURE NO (POTREBBE ESSERE DEDOTTO DALLE COORDINATE DEL DRONE) --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self._uav_ID = ID
        self._x_coord = pos[0] # cell._x_coord
        self._y_coord = pos[1] # cell._y_coord
        self._z_coord = pos[2] # cell._z_coord
        self._toward = toward # DA VEDERE SE INSERIRLO OPPURE NO --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self._action = action
        self._bandwidth = bandwidth
        self._battery_level = battery_level
        self._footprint = footprint
        self._max_uav_height = max_uav_height
        self._action_set = action_set 
        self._coming_home = False
        self._cs_goal = (None, None, None) if DIMENSION_2D==False else (None, None)
        self._path_to_the_closest_CS = []
        self._current_pos_in_path_to_CS = -1
        self._required_battery_to_CS = None
        self._users_in_footprint = []
        self._charging = False
        self._n_recharges = 0
        self._crashed = False
        self._current_consumption_to_go_cs = 1 # --> It is reset to 1 every time it reaches a value equal to PERC_CONSUMPTION_PER_ITERATION.
        self._throughput_request = TR
        self._edge_computing = EC
        self._data_gathering = DG 
        self._d_ag_cc = d_ag_cc

    @property
    def _vector(self):
        return np.array([self._x_coord, self._y_coord, self._z_coord])

    @property
    def _n_actions(self):
        return len(self._action_set)

    @property
    def _n_step_to_the_closest_cs(self):
        return len(self._path_to_the_closest_CS)

    def move_2D_limited_battery(self, old_agent_pos, move_action):

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # 2D motion;                                                            #
        # LIMITED UAV battery;                                                  #
        # constant battery consumption for both UAV motion and services;        #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        next_cell_x = self._x_coord
        next_cell_y = self._y_coord

        if (move_action == CHARGE):
            self._charging = True
            self.charging_battery1()
            new_agent_pos = (next_cell_x, next_cell_y)
            #print("CI SONOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")

            return new_agent_pos

        elif (move_action == GO_TO_CS):
            self._coming_home = True
            self.residual_battery_when_come_home()
            
            #print("_current_pos_in_path_to_CS", self._current_pos_in_path_to_CS)
            #print(self._path_to_the_closest_CS)
            new_agent_pos = self._path_to_the_closest_CS[self._current_pos_in_path_to_CS]
            new_agent_pos = (new_agent_pos[0], new_agent_pos[1]) 
            self._x_coord = new_agent_pos[0]
            self._y_coord = new_agent_pos[1]

            #print(new_agent_pos)
            return new_agent_pos

        else:
            
            if (move_action == HOVERING):
                self.residual_battery1(move_action)
                return (next_cell_x, next_cell_y)

            elif (move_action == LEFT):
                next_cell_x -= UAV_XY_STEP

            elif (move_action == RIGHT):
                next_cell_x += UAV_XY_STEP

            elif (move_action == UP):
                next_cell_y += UAV_XY_STEP

            elif (move_action == DOWN):
                next_cell_y -= UAV_XY_STEP

            self._charging = False
            self._coming_home = False
            self._cs_goal = (None, None)


        new_agent_pos = (next_cell_x, next_cell_y)

        # SOLO PER VEDERE SE FUNZIONA IL PLOT DELLA MAPPA PER L'ANIMAZIONE: --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #if self._battery_level <= 0:
            #self._battery_level = 100

        if (self.off_map_move_2D(new_agent_pos)):
            new_agent_pos = old_agent_pos
            # Reduction battery level due to the agent motion: 

        # Constant reduction battery level due to UAV motion and the provided service:
        self.residual_battery1(move_action)

        self._x_coord = new_agent_pos[0]
        self._y_coord = new_agent_pos[1]

        return new_agent_pos

    def move_2D_unlimited_battery(self, old_agent_pos, move_action):
        # # # # # # # # # # # # # # 
        # 2D motion;              #
        # UNLIMITED UAV battery;  #                          
        # # # # # # # # # # # # # #

        next_cell_x = self._x_coord
        next_cell_y = self._y_coord

        if (move_action == HOVERING):
            #print("HOVERING")
            return (next_cell_x, next_cell_y)

        elif (move_action == LEFT):
            #print("LEFT")
            next_cell_x -= UAV_XY_STEP

        elif (move_action == RIGHT):
            #print("RIGHT")
            next_cell_x += UAV_XY_STEP

        elif (move_action == UP):
            #print("UP")
            next_cell_y += UAV_XY_STEP

        elif (move_action == DOWN):
            #print("DOWN")
            next_cell_y -= UAV_XY_STEP

        new_agent_pos = (next_cell_x, next_cell_y)

        if (self.off_map_move_2D(new_agent_pos)):
            new_agent_pos = old_agent_pos

        self._x_coord = new_agent_pos[0]
        self._y_coord = new_agent_pos[1]

        return new_agent_pos

    def move_3D_limited_battery(self, old_agent_pos, move_action, cells_matrix): 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # 3D motion;                                                            #
        # LIMITED UAV battery                                                   #
        # constant battery consumption for both UAV motion and services;        #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        next_cell_x = self._x_coord
        next_cell_y = self._y_coord
        next_cell_z = self._z_coord

        if (move_action == CHARGE):
            #print("CHARGE")
            self._charging = True
            self.charging_battery1()
            new_agent_pos = (next_cell_x, next_cell_y, next_cell_z)

            return new_agent_pos
        
        elif (move_action == GO_TO_CS):
            #print("GO TO CS")
            self._coming_home = True
            self.residual_battery_when_come_home()
            
            #print(self._x_coord, self._y_coord, self._z_coord)
            #print(self._path_to_the_closest_CS)
            #print("QUIIIIIIIIIIIIII", self._current_pos_in_path_to_CS)
            new_agent_pos = self._path_to_the_closest_CS[self._current_pos_in_path_to_CS]
            self._x_coord = new_agent_pos[0]
            self._y_coord = new_agent_pos[1]
            self._z_coord = new_agent_pos[2]

            return new_agent_pos

        else:
            
            if (move_action == HOVERING):
                #print("HOVERING")
                self.residual_battery1(move_action)
                return (next_cell_x, next_cell_y, next_cell_z)

            elif (move_action == LEFT):
                #print("LEFT")
                next_cell_x -= UAV_XY_STEP

            elif (move_action == RIGHT):
                #print("RIGHT")
                next_cell_x += UAV_XY_STEP

            elif (move_action == UP):
                #print("UP")
                next_cell_y += UAV_XY_STEP

            elif (move_action == DOWN):
                #print("DOWN")
                next_cell_y -= UAV_XY_STEP

            elif (move_action == DROP):
                #print("DROP")
                next_cell_z -= UAV_Z_STEP

            elif (move_action == RISE):
                #print("RISE")
                next_cell_z += UAV_Z_STEP

            self._charging = False
            self._coming_home = False
            self._cs_goal = (None, None, None)

        new_agent_pos = (next_cell_x, next_cell_y, next_cell_z)

        # SOLO PER VEDERE SE FUNZIONA IL PLOT DELLA MAPPA PER L'ANIMAZIONE: --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #if self._battery_level <= 0:
            #self._battery_level = 100

        if (self.off_map_move_3D(new_agent_pos, cells_matrix)):
            new_agent_pos = old_agent_pos
        
        # Constant reduction battery level due to UAV motion and the provided service:
        #print("DECREMENTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        self.residual_battery1(move_action)

        if (MULTI_SERVICE==True):
            # Reduction battery level due to service provided by the agent:
            #self.residual_battery_after_service()
            pass

        self._x_coord = new_agent_pos[0]
        self._y_coord = new_agent_pos[1]
        self._z_coord = new_agent_pos[2]

        return new_agent_pos

    def move_3D_unlimited_battery(self, old_agent_pos, move_action, cells_matrix): 
        # # # # # # # # # # # # # # 
        # 3D motion;              #
        # UNLIMITED UAV battery;  #
        # # # # # # # # # # # # # #

        next_cell_x = self._x_coord
        next_cell_y = self._y_coord
        next_cell_z = self._z_coord

        if (move_action == HOVERING):
            return (next_cell_x, next_cell_y, next_cell_z)

        elif (move_action == LEFT):
            next_cell_x -= UAV_XY_STEP

        elif (move_action == RIGHT):
            next_cell_x += UAV_XY_STEP

        elif (move_action == UP):
            next_cell_y += UAV_XY_STEP

        elif (move_action == DOWN):
            next_cell_y -= UAV_XY_STEP

        elif (move_action == DROP):
            next_cell_z -= UAV_Z_STEP

        elif (move_action == RISE):
            next_cell_z += UAV_Z_STEP

        new_agent_pos = (next_cell_x, next_cell_y, next_cell_z)

        if (self.off_map_move_3D(new_agent_pos, cells_matrix)):
            #print("CI SONOOOOOOOOOOOOOOOOOOOOO")
            new_agent_pos = old_agent_pos

        self._x_coord = new_agent_pos[0]
        self._y_coord = new_agent_pos[1]
        self._z_coord = new_agent_pos[2]

        return new_agent_pos

    def move(self, old_agent_pos, move_action, cells_matrix): 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Move the agent according to 'move_action' and return the 'next_cell'  #
        # on which the considered agent (i.e, UAV) will be placed.              #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        #next_cell_x = self._cell._x_coord
        #next_cell_y = self._cell._y_coord
        #next_cell_z = self._cell._z_coord

        next_cell_x = self._x_coord
        next_cell_y = self._y_coord
        next_cell_z = self._z_coord

        if (move_action == HOVERING):
            return (next_cell_x, next_cell_y, next_cell_z)

        elif (move_action == LEFT):
            next_cell_x -= UAV_XY_STEP

        elif (move_action == RIGHT):
            next_cell_x += UAV_XY_STEP

        elif (move_action == UP):
            next_cell_y += UAV_XY_STEP

        elif (move_action == DOWN):
            next_cell_y -= UAV_XY_STEP

        elif (move_action == DROP):
            next_cell_z -= UAV_Z_STEP

        elif (move_action == RISE):
            next_cell_z += UAV_Z_STEP

        new_agent_pos = (next_cell_x, next_cell_y, next_cell_z)

        if (self.off_map_move_3D(new_agent_pos, cells_matrix)):
            new_agent_pos = old_agent_pos
            # Reduction battery level due to the agent motion: 
            self.residual_battery_after_propulsion(HOVERING)
        else:
            self.residual_battery_after_propulsion(move_action)

        if self._battery_level <= 0:
            self._battery_level = 100

        # DEVI DEFINIRE 'users' --> !!!!!!!!!!!!!!!!!!!!
        #users_in_footprint = self.users_in_uav_footprint(users) # --> IMPORTA GLI USERS --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #n_served_users = self.n_served_users_in_foot_and_type_of_service(users_in_footprint)
        
        # # Reduction battery level due to service provided by the agent:
        self.residual_battery_after_service()

        # DEVI DEFINIRE 'centroids' --> !!!!!!!!!!!!!!!!!!!!!!!!
        #cc_reference_ = self.centroid_cluster_reference(centroids) # Selected centroid cluster as reference for the current agent --> IMPORTA I CENTROIDS --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #d_ag_cc_ = LA.norm(np.array(agent_pos_) - np.array(cc_reference_)) # Distance betwen the agent position and the centroid cluster selected as reference
        #self._d_ag_cc = d_ag_cc_

        self._x_coord = new_agent_pos[0]
        self._y_coord = new_agent_pos[1]
        self._z_coord = new_agent_pos[2]

        return new_agent_pos

    def off_map_move_2D(self, new_agent_pos, cells_matrix=None):
        # agent_pos is a tuple (x,y)

        agent_x = new_agent_pos[0]
        agent_y = new_agent_pos[1]

        if \
        ( (agent_x < LOWER_BOUNDS) or \
        (agent_y < LOWER_BOUNDS) or \
        (agent_x >= CELLS_COLS) or \
        (agent_y >= CELLS_ROWS) ):

            return True

        else:

            return False

    def off_map_move_3D(self, new_agent_pos, cells_matrix):
        # agent_pos is a tuple (x,y,z)

        agent_x = new_agent_pos[0]
        agent_y = new_agent_pos[1]
        agent_z = new_agent_pos[2]

        cell_x = int(agent_x-0.5)
        cell_y = int(agent_y-0.5)
        
        if \
        ( (agent_x < LOWER_BOUNDS) or \
        (agent_y < LOWER_BOUNDS) or \
        (agent_z < MIN_UAV_HEIGHT) or \
        (agent_x >= CELLS_COLS) or \
        (agent_y >= CELLS_ROWS) or \
        (agent_z >= MAX_UAV_HEIGHT) or \
        (cells_matrix[cell_y][cell_x]==OBS_IN) ):
            
            return True
        
        else:
            
            return False
        
        '''
        (cells_matrix[cell_y][cell_x]==OBS_IN) ):
            #print("CI SONOOOOOOOOOOOOOOOOOOOOO")
            return True

        else:

            return False
        '''

    def compute_UAV_footprint_radius(self, z_coord, theta=30):
        # Compute the radius of the UAV footprint taking as input the
        # z coordinate of the UAV and its 'angle of view' 'theta'.

        theta_radians = radians(theta)
        radius = z_coord*tan(theta_radians)

        return radius

    def compute_distances(self, desired_cells):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Compute the distance between the agent position and the position of specific cells; #
        # it returns a list of tuple in which the first item represents a cell and the second #
        # item is the distance of the considered cell from the current agent position.        #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        distances_from_current_position = [(cell, LA.norm(cell._vector - self._vector)) for cell in desired_cells]
        # Order the list of tuples according to their second item (i.e. the distance from the current agent position): 
        distances_from_current_position.sort(key=lambda x: x[1])

        # Set the closest CS (for the current UAV) equal to the first elem of the first tuple of the previous computed ordered list:
        closest_cs_cell = distances_from_current_position[0][0]
        if (DIMENSION_2D == False):
            z_cs = closest_cs_cell._z_coord+0.5
            self._cs_goal = (closest_cs_cell._x_coord+0.5, closest_cs_cell._y_coord+0.5, z_cs) # --> SIDE-EFFECT on attribute 'cs_goal'
        else:
            self._cs_goal = (closest_cs_cell._x_coord+0.5, closest_cs_cell._y_coord+0.5) # --> SIDE-EFFECT on attribute 'cs_goal'

        return distances_from_current_position

    @staticmethod
    def setting_agents_pos(cs_points_or_cells):

        UAVS_ON_EACH_CS = N_UAVS//N_CS
        REMAINING_UAVS_ON_LAST_CS = N_UAVS%N_CS
        last_CS_index = N_CS - 1
        start_uav_idx = 0
        end_uav_idx = UAVS_ON_EACH_CS
        uavs_initial_pos = []

        # Each UAV position is set according to the following rules:
        #       - IF 'N_CS' == 'N_UAVS', THEN each UAV will be placed on a different CS;
        #       - IF 'N_CS' < 'N_UAVS', THEN
        #                           IF 'N_CS' is divisible by 'N_UAVS', THEN equal number of UAVS will be placed on each CS;
        #                           OTHERWISE an equal number of UAV will be set on 'N_CS -1' charging stations, and the remaining UAVs will be placed on the last CS;
        #       - 'N_CS' can not be >= 'N_UAVS'.

        if N_CS > N_UAVS:
            print("Invalid Setting: Number of charging stations exceeds number of drones!")

        for CS_i in range(N_CS):

            if (CS_i == last_CS_index):
                end_uav_idx += REMAINING_UAVS_ON_LAST_CS
            
            for UAV_i in range(start_uav_idx, end_uav_idx):
                uavs_initial_pos.append((cs_points_or_cells[CS_i]._x_coord, cs_points_or_cells[CS_i]._y_coord, cs_points_or_cells[CS_i]._z_coord)) # The value 0.5 indicates that the UAV is assumed to be in the middle of a point or a cell.

            #print((start_uav_idx, end_uav_idx))
            start_uav_idx = end_uav_idx

            end_uav_idx += UAVS_ON_EACH_CS 

        return uavs_initial_pos

    @staticmethod
    def initialize_agents(agents_pos, max_uav_height, action_set):
        # # # # # # # # # # # # # # # # # # # # # # # # 
        # Initialize the agents on their first start; #
        # 'agents_pos' is a list of tuple (x,y,z).    #
        # # # # # # # # # # # # # # # # # # # # # # # #

        # 'x' and 'y' are derived from the integer part division used with the derired resolution cell (because we only know where the drone is according to the selected resolution): 
        agents = [Agent((pos[0]+0.5, pos[1]+0.5, pos[2]+0.5), 1, 0, 1, UAV_BANDWIDTH, FULL_BATTERY_LEVEL, ACTUAL_UAV_FOOTPRINT, max_uav_height, action_set, False, False, False, 2) for pos in agents_pos]
        
        return agents

    @staticmethod
    def moving_time(a, b, a_max, v_max):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Compute the moving time of the each UAV by assuming that the UAV timing law has #
        # has a conventional trapezoidal speed profile and a bang-coast-bang acceleration #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        N = LA.norm(a - b)*a_max - pow(v_max, 2)
        D = a_max*v_max
        Tm = N/D

        return Tm

    @staticmethod
    def residual_battery_after_charging(current_residual_battery):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Compute residual battery autonomy after charging; every charge has a minimum  #
        # time of X minutes and thus every time a charge is performed, the UAV has to   #
        # charge for at least X minutes.                                                #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  

        percentage_of_gained_battery = MINIMUM_CHARGING_TIME/BATTERY_AUTONOMY_TIME
        gained_battery = percentage_of_gained_battery*100
        new_residual_battery = current_residual_battery + int(round(gained_battery))
        
        return new_residual_battery

    def charging_battery1(self):

        self._battery_level += BATTERY_CHARGED_PER_IT
        if (self._battery_level > FULL_BATTERY_LEVEL):
            self._battery_level = FULL_BATTERY_LEVEL

    @staticmethod
    def residual_battery_after_moving(elapsed_time, current_residual_battery):
        # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Compute residual battery autonomy after moving. #
        # # # # # # # # # # # # # # # # # # # # # # # # # #

        percentage_of_battery_loss = elapsed_time/BATTERY_AUTONOMY_TIME
        battery_loss = percentage_of_battery_loss*100
        new_residual_battery = battery_loss - int(round(battery_loss))

        return new_residual_battery

    def centroid_cluster_reference(self, centroids):

        # TO DO . . . taking into account the SE_avg per cluster, the minimum distance, the amount of service requests and if the cluster is served yet or if it is free --> !!!!!!!!!!!!!!!

        return centroids[0]

    def users_in_uav_footprint(self, users, uav_footprint, discovered_users):
        # Only not served users are considered inside each UAV footprint.

        uav_x = self._x_coord
        uav_y = self._y_coord
        #space_to_check = [(uav_x-uav_footprint, uav_x+uav_footprint), (uav_y-uav_footprint, uav_y+uav_footprint)] # (x_min, x_max), (y_min, y_max) are the extremes coordinates inside the current agent can find a user

        users_in_footprint = []
        for user in users:
            user_x = user._x_coord
            user_y = user._y_coord
            #user_z = user._z_coord

            # Update the info related to the current user
            #user.user_info_update()

            # Check who are the users inside the uav footprint (the radius of the footprint has been used):
            #print(self._vector, "-", np.array([user_x, user_y])) #, user_z
            #print("DISTANCE USER-UAV:", LA.norm(np.array([uav_x, uav_y]) - np.array([float(user_x), float(user_y)])))
            #print(self._footprint)
            #print("UAV:", (uav_x, uav_y), "USER", (float(user_x), float(user_y)))
            if ( LA.norm(np.array([uav_x, uav_y]) - np.array([float(user_x), float(user_y)])) < self._footprint ): #, user_z
                if (user not in discovered_users):
                    discovered_users.append(user) # --> SIDE-EFFECT on 'discovered_users'
                # Check if the current user inside the UAV footprint is not served OR if it is served yet; in both cases the current agent will serve this user.
                if ( (user._info[0]) and (user in self._users_in_footprint)): #and (not user in users_in_footprint) ): # --> E' corretto l'AND, ma usandolo ottengo che uno dei due droni (se ne uso due) ottiene un reward massimo di 0.50 anche se sta facendo tutto correttamente come si puo' vedere dall'animazione --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
                    users_in_footprint.append(user)
                elif ( (not user._info[0]) ):
                    users_in_footprint.append(user)
                    #user._info[0] = True
            '''
            # Check if the X coord of the considered user is inside the current UAV footprint:
            if ( (user_x >= space_to_check[0][0]) and (user_x <= space_to_check[0][1]) ):
                # Check if the Y coord of the considered user is inside the current UAV footprint:
                if ( (user_y >= space_to_check[1][0]) and (user_y <= space_to_check[1][1]) ):
                    if not user._info[0]:
                        users_in_footprint.append(user)
            '''

        return users_in_footprint

    def users_in_uav_footprint_lim_band(self, users, uav_footprint, discovered_users):
        # Only not served users are considered inside each UAV footprint.

        uav_x = self._x_coord
        uav_y = self._y_coord
        #space_to_check = [(uav_x-uav_footprint, uav_x+uav_footprint), (uav_y-uav_footprint, uav_y+uav_footprint)] # (x_min, x_max), (y_min, y_max) are the extremes coordinates inside the current agent can find a user

        users_in_footprint = []
        self._bandwidth = UAV_BANDWIDTH
        bandwidth_request_in_current_footprint = 0
        for user in users:
            user_x = user._x_coord
            user_y = user._y_coord
            #user_z = user._z_coord

            # Update the info related to the current user
            #user.user_info_update()

            # Check who are the users inside the uav footprint (the radius of the footprint has been used):
            #print(self._vector, "-", np.array([user_x, user_y])) #, user_z
            #print("DISTANCE USER-UAV:", LA.norm(np.array([uav_x, uav_y]) - np.array([float(user_x), float(user_y)])))
            #print(self._footprint)
            #print("UAV:", (uav_x, uav_y), "USER", (float(user_x), float(user_y)))
            # (Virtually) set to 0 all the service for this current footprint in such a way to know which service is not provide when all users will be scrolled:
            self._throughput_request = False
            self._edge_computing = False
            self._data_gathering = False
            if ( LA.norm(np.array([uav_x, uav_y]) - np.array([float(user_x), float(user_y)])) < self._footprint ): #, user_z
                if (user not in discovered_users):
                    discovered_users.append(user) # --> SIDE-EFFECT on 'discovered_users'
                # Check if the current user inside the UAV footprint is not served OR if it is served yet; in both cases the current agent will serve this user.
                if ( ((user._info[0]) and (user in self._users_in_footprint) and (self._bandwidth>=user._info[5])) or ((not user._info[0]) and (self._bandwidth>=user._info[5])) ): #and (not user in users_in_footprint) ): # --> E' corretto l'AND, ma usandolo ottengo che uno dei due droni (se ne uso due) ottiene un reward massimo di 0.50 anche se sta facendo tutto correttamente come si puo' vedere dall'animazione --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
                    # The users inside the current UAV footprint are also the ones which are not requesting for a service:
                    users_in_footprint.append(user)
                    self._bandwidth -= user._info[5]
                    bandwidth_request_in_current_footprint += user._info[5]
                    if (user._info[1]==THROUGHPUT_REQUEST):
                        self._throughput_request = True
                    elif (user._info[1]==EDGE_COMPUTING):
                        self._edge_computing = True
                    elif (user._info[1]==DATA_GATHERING):
                        self._data_gathering = True
                    #user._info[0] = True
                
                '''
                elif ( (not user._info[0]) and (self._bandwidth>=user._info[5])):
                    users_in_footprint.append(user)
                    self._bandwidth -= user._info[5]
                    if (user._info[1]==THROUGHPUT_REQUEST):
                        self._throughput_request = True
                    elif (user._info[1]==EDGE_COMPUTING):
                        self._edge_computing = True
                    elif (user._info[1]==DATA_GATHERING):
                        self._data_gathering = True
                '''
            '''
            # Check if the X coord of the considered user is inside the current UAV footprint:
            if ( (user_x >= space_to_check[0][0]) and (user_x <= space_to_check[0][1]) ):
                # Check if the Y coord of the considered user is inside the current UAV footprint:
                if ( (user_y >= space_to_check[1][0]) and (user_y <= space_to_check[1][1]) ):
                    if not user._info[0]:
                        users_in_footprint.append(user)
            '''

            #print("\nRIMANENTE", self._bandwidth, "\n")

        return users_in_footprint, bandwidth_request_in_current_footprint

    def check_if_on_CS(self):

        if (DIMENSION_2D==False):
            if ( (self._cs_goal[0]==self._x_coord) and (self._cs_goal[1]==self._y_coord) and (self._cs_goal[2]==self._z_coord) ):
                return True
            else:
                return False

        else:
            if ( (self._cs_goal[0]==self._x_coord) and (self._cs_goal[1]==self._y_coord)):
                return True
            else:
                return False

    @staticmethod
    def n_served_users_in_foot(users_in_foot):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # In this method it is assumed that all the users inside the UAV footprint are served;              #
        # every user has the same priority and ask for the same service by using an infinite UAV bandwidth. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Set the provided services and served users:
        for user in users_in_foot:
            user._info[0] = True

        # UAV serves all the users inside its footprint:
        served_users = len(users_in_foot)

        return served_users

    @staticmethod
    def n_served_users_in_foot_and_type_of_service(users_in_foot):

        TS_service = 0
        EC_service = 0
        DG_service = 0

        # Set the provided services and served users:
        for user in users_in_foot:
            if (True):
                user._info[0] = True # TO DO . . . --> Stabilisci una logica/euristica secondo la quale servire un utente oppure no, tenendo conto dei tipi di servizi che andrai ad erogare --> !!!!!!!!!!!!!!!!!!!
                if (user._info[1] == THROUGHPUT_REQUEST): TS_service += 1
                elif (user._info[1] == EDGE_COMPUTING): CS_service += 1
                elif (user._info[1] == DATA_GATHERING): DG_service += 1

            else:
                user._info[0] = False

        # PER IL MOMENTO ASSUMO CHE VENGANO SERVITI TUTTI GLI UTENTI ALL'INTERNO DEL FOOTPRINT DELL'AGENT E CHE VENGANO EROGATI TUTTI I SERVIZI DISPONIBILI: --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        served_users = len(users_in_foot)
        #self._throughput_request = True
        #self._edge_computing = True
        #self._data_gathering = True

        return served_users

    @staticmethod
    def set_not_served_users(users, all_users_in_all_foots, current_provided_services, serving_uav_id, QoEs_store, current_iteration): #users_served_time, user_request_service_elapsed_time
        # 'all_users_in_all_foots' is a list containing all the users inside all the footprints of each UAVs. 

        # Set the users not served (during a single iteration of each UAV):
        for user in users:
            if (not user in all_users_in_all_foots):

                user._info[0] = False

            # Update the info related to the current user (only when all the UAVs have performed their actions):
            if (serving_uav_id==N_UAVS):
                user.user_info_update(QoEs_store, current_provided_services, current_iteration)
                #user.user_info_update(users_served_time, current_provided_services, current_iteration) # users_served_time, user_request_service_elapsed_time = user.user_info_ . . .

    @staticmethod
    def set_not_served_users_inf_request(users, all_users_in_all_foots, serving_uav_id, QoEs_store, current_iteration):
        # 'all_users_in_all_foots' is a list containing all the users inside all the footprints of each UAVs.

        # Set the users not served:
        for user in users:
            if (not user in all_users_in_all_foots):

                user._info[0] = False

            # Update the info related to the current user
            if (serving_uav_id==N_UAVS):
                user.user_info_update_inf_request(QoEs_store, current_iteration)
                #user.user_info_update(QoEs_store)
        #return users_served_time, user_request_service_elapsed_time


# ______________________________________________________________________________________________________________________________________________________________
# I) BATTERY CONSUMPTION: only if propulsion and UAVs services are considered together in the same and unique (average) consumption.

    def residual_battery1(self, move_action):

        self._battery_level -= PERC_CONSUMPTION_PER_ITERATION
        #battery_not_rounded = self._battery_level - PERC_CONSUMPTION_PER_ITERATION
        #self._battery_level = round(battery_not_rounded)

    def needed_battery_to_come_home(self):

        needed_battery_to_cs = self._n_step_to_the_closest_cs*PERC_BATTERY_TO_GO_TO_CS
        #n_step_with_ususal_consumption = ceil(self._n_step_to_the_closest_cs/STEP_REDUCTION_TO_GO_TO_CS)
        #needed_battery_to_cs = round(PERC_CONSUMPTION_PER_ITERATION*n_step_with_ususal_consumption, 2)

        return needed_battery_to_cs

    def residual_battery_when_come_home(self):

        #print("CONSUMPTION TO CS:", self._current_consumption_to_go_cs)
        if (self._current_consumption_to_go_cs == PERC_CONSUMPTION_PER_ITERATION):
            #print("CI SONOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
            self._battery_level -= PERC_CONSUMPTION_PER_ITERATION
            self._current_consumption_to_go_cs = 1
        else:
            self._current_consumption_to_go_cs += PERC_BATTERY_TO_GO_TO_CS
        #if ( (self._current_pos_in_path_to_CS+1)%STEP_REDUCTION_TO_GO_TO_CS ):
            #self._battery_level -= PERC_CONSUMPTION_PER_ITERATION
# ______________________________________________________________________________________________________________________________________________________________


# ______________________________________________________________________________________________________________________________________________________________
# II) BATTERY CONSUMPTION: only if propulsion and UAVs services have different battery consumption and a time variable is explicitly used during the training.

    def residual_battery_after_service(self):
        
        battery_consumption = 0

        if (self._edge_computing == True):
            self._battery_level -= battery_consumption
    
    def residual_battery_after_propulsion(self, action):
        
        if ( (action == LEFT) or (action == RIGHT) or (action == UP) or (action==DOWN) or (action == DROP) or (action == RISE) ):
            self._battery_level -= PERC_CONSUMPTION_PER_ITERATION
        # Case in which the agent perform an action among HOVERING:
        else:
            self._battery_level -= PERC_CONSUMPTION_IF_HOVERING

# _______________________________________________________________________________________________________________________________________________________________







# PROVA:

# Loading:
load = Loader()
load.maps_data()

cs_points = load.cs_points
cs_cells = load.cs_cells
enb_cells = load.enb_cells
cells_matrix = load.cells_matrix
#print([(cs_cell._x_coord, cs_cell._y_coord, cs_cell._z_coord) for cs_cell in cs_cells])

uavs_pos = Agent.setting_agents_pos(cs_cells) if UNLIMITED_BATTERY==False else UAVS_POS
#print("UAVS POS:", uavs_pos)
max_uav_height_test = 12
action_set_test = []
uavs_pos = Agent.initialize_agents(uavs_pos, max_uav_height_test, action_set_test)
#for uav in uavs_pos:
    #print((uav._x_coord, uav._y_coord, uav._z_coord))

cell = Cell(0, 0, 1, 0, 0, 2)
agent = Agent((1,0,0), 1, 0, 1, 4, 100, None, max_uav_height_test, action_set_test, False, False, False, 2)
#print("SERVICE BATTERY CONSUMPTION", agent.residual_battery_after_service())
#print(cell._vector)
next_cell_coords = agent.move((0,0,0), LEFT, cells_matrix)
CS_distances = agent.compute_distances(cs_cells) if UNLIMITED_BATTERY==False else print("Unlimited battery case --> No need to compute distance between agent and CS.")
if (CREATE_ENODEB == True):
    eNodeB_distance = agent.compute_distances(enb_cells)
#print("Disances between agent and charging stations:", CS_distances)
#print("Disances between agent and eNodeB:", eNodeB_distance)

#if next_cell_coords == -1:
    #print("Motion not allowed")
#else:
    #print("Actual cell coordinates", (cell._x_coord, cell._y_coord, cell._z_coord))
    #print("Next cell coordinates:", (next_cell_coords[0], next_cell_coords[1], next_cell_coords[2]))
