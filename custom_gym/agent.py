# AGENT MAIN CLASSES AND METHODS DEFINITION RELATED TO IT.

from my_utils import *
from scenario_objects import Cell, Point, Environment, User
from load_and_save_data import Loader
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
        self._uav_ID = ID
        self._x_coord = pos[0]
        self._y_coord = pos[1]
        self._z_coord = pos[2]
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
        self._standard_behav_forward = False

    @property
    def _vector(self):
        return np.array([self._x_coord, self._y_coord, self._z_coord])

    @property
    def _n_actions(self):
        return len(self._action_set)

    @property
    def _n_step_to_the_closest_cs(self):
        return len(self._path_to_the_closest_CS)

    def move(self, move_action, cells_matrix=None):

        next_cell_x = self._x_coord
        next_cell_y = self._y_coord
        next_cell_z = self._z_coord
        old_agent_pos = (next_cell_x, next_cell_y) if DIMENSION_2D==True else (next_cell_x, next_cell_y, next_cell_z)

        if (move_action == CHARGE):
            self._charging = True
            self.charging_battery1()

            return old_agent_pos

        elif (move_action == GO_TO_CS):
            self._coming_home = True
            self.residual_battery_when_come_home()
            new_agent_pos = self._path_to_the_closest_CS[self._current_pos_in_path_to_CS]
            new_agent_pos = (new_agent_pos[0], new_agent_pos[1]) if (DIMENSION_2D==True) else (new_agent_pos[0], new_agent_pos[1], new_agent_pos[2]) 
            self._x_coord = new_agent_pos[0]
            self._y_coord = new_agent_pos[1]
            if (DIMENSION_2D==False):
                self._z_coord = new_agent_pos[2]

            return new_agent_pos

        else:
            
            if (move_action == HOVERING):
                self.residual_battery1(move_action)
                return old_agent_pos

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

            self._charging = False
            self._coming_home = False
            self._cs_goal = (None, None) if DIMENSION_2D==True else (None, None, None)

            agent_is_off_map = self.off_map_move_2D((next_cell_x, next_cell_y)) if DIMENSION_2D==True else self.off_map_move_3D((next_cell_x, next_cell_y, next_cell_z), cells_matrix)
            
            if (agent_is_off_map):
                new_agent_pos = old_agent_pos
            
            else:
                new_agent_pos = (next_cell_x, next_cell_y) if DIMENSION_2D==True else (next_cell_x, next_cell_y, next_cell_z)

            if (UNLIMITED_BATTERY==False):
                # Constant reduction battery level due to UAV motion and the provided service:
                self.residual_battery1(move_action)

            self._x_coord = new_agent_pos[0]
            self._y_coord = new_agent_pos[1]
            if (DIMENSION_2D==False):
                self._z_coord = new_agent_pos[2]

            return new_agent_pos

    def move_standard_behaviour(self, move_action):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Motion for the standard behavior (2D, 3D, limited and unlimited battery). #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        next_cell_x = self._x_coord
        next_cell_y = self._y_coord
        if (DIMENSION_2D==False):
            next_cell_z = self._z_coord

        if (move_action == CHARGE):
            self._charging = True
            self._coming_home = False
            self.charging_battery1()
            
            if (DIMENSION_2D==False):
                new_agent_pos = (next_cell_x, next_cell_y, next_cell_z)
            else:
                new_agent_pos = (next_cell_x, next_cell_y)
            
            return new_agent_pos
        
        elif (move_action == GO_TO_CS):
            self._coming_home = True
            self._charging = False
            self.residual_battery_when_come_home()
            new_agent_pos = self._path_to_the_closest_CS[self._current_pos_in_path_to_CS]
            
            if (DIMENSION_2D==True):
                new_agent_pos = (new_agent_pos[0], new_agent_pos[1])
                self._x_coord = new_agent_pos[0]
                self._y_coord = new_agent_pos[1]
            else:
                self._z_coord = new_agent_pos[2]

            return new_agent_pos
        
        else:

            if (DIMENSION_2D==False):
                next_cell_z = self._z_coord

            if (move_action == LEFT):
                next_cell_x -= UAV_XY_STEP

            elif (move_action == RIGHT):
                next_cell_x += UAV_XY_STEP
                
            elif (move_action == UP):
                next_cell_y += UAV_XY_STEP

            elif (move_action == DOWN):
                next_cell_y -= UAV_XY_STEP

            elif (move_action == RISE):
                next_cell_z += UAV_Z_STEP

            if (next_cell_x <= 1):
                next_cell_x = 1.5
            if (next_cell_y <= 1):
                next_cell_y = 1.5
            if (next_cell_x >= CELLS_COLS):
                next_cell_x = CELLS_COLS-1
            if (next_cell_y >= CELLS_ROWS):
                next_cell_y = CELLS_ROWS-1

            if (DIMENSION_2D==False):
                new_agent_pos = (next_cell_x, next_cell_y, next_cell_z)
                self._x_coord = new_agent_pos[0]
                self._y_coord = new_agent_pos[1]
                self._z_coord = new_agent_pos[2]
            else:
                new_agent_pos = (next_cell_x, next_cell_y)
                self._x_coord = new_agent_pos[0]
                self._y_coord = new_agent_pos[1]

            if (UNLIMITED_BATTERY==False):
                self._charging = False
                self._coming_home = False
                if (DIMENSION_2D==True):
                    self._cs_goal = (None, None)
                else:
                    self._cs_goal = (None, None, None)

        self.residual_battery1(move_action)

        return new_agent_pos

    def action_for_standard_h(self, cells_matrix):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Horizontal trajectory action to perform when stadard behaviour is enable. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        if (self._x_coord==1.5):
            self._standard_behav_forward = True
            action = RIGHT

        elif (self._x_coord==CELLS_COLS-1):
            self._standard_behav_forward = False
            action = LEFT

        else:

            if (self._standard_behav_forward==True):
                action = RIGHT
            else:
                action = LEFT

        if (DIMENSION_2D==False):
            cell_x = int(self._x_coord)
            cell_y = int(self._y_coord)
            if (cells_matrix[cell_y][cell_x]._z_coord>=self._z_coord):
                action = RISE

        return action

    def action_for_standard_v(self, cells_matrix):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Vertical trajectory action to perform when stadard behaviour is enable. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        if (self._y_coord==1.5):
            self._standard_behav_forward = True
            action = UP

        elif (self._y_coord==CELLS_ROWS-1):
            self._standard_behav_forward = False
            action = DOWN

        else:

            if (self._standard_behav_forward==True):
                action = UP
            else:
                action = DOWN

        if (DIMENSION_2D==False):
            cell_x = int(self._x_coord)
            cell_y = int(self._y_coord)
            if (cells_matrix[cell_y][cell_x]._z_coord>=self._z_coord):
                action = RISE

        return action

    def action_for_standard_square_clockwise(self, cells_matrix):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Square trajectory action to perform when stadard behaviour is enable. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        if (self._x_coord==self._y_coord==1.5):
            action = RIGHT
        
        elif (self._x_coord==CELLS_COLS-1) and (self._y_coord==1.5):
            action = UP
        
        elif (self._x_coord==1.5) and (self._y_coord==CELLS_ROWS-1):
            action = DOWN
        
        elif ( (self._x_coord==CELLS_COLS-1) and (self._y_coord==CELLS_ROWS-1) ):
            action = LEFT

        # Upper side:
        elif ( (self._y_coord==1.5) and ((self._x_coord>=1.5) and (self._x_coord<=CELLS_COLS)) ):
            action = RIGHT
        # Lower side:
        elif ( (self._y_coord==CELLS_ROWS-1) and ((self._x_coord>=1.5) and (self._x_coord<=CELLS_COLS)) ):
            action = LEFT
        # Left side:
        elif ( (self._x_coord==1.5) and ((self._y_coord>=1.5) and (self._y_coord<=CELLS_ROWS)) ):
            action = DOWN
        # Right side:
        elif ( (self._x_coord==CELLS_COLS-1) and ((self._y_coord>=1.5) and (self._y_coord<=CELLS_ROWS)) ):
            action = UP
        else:
            min_x = min([self._x_coord, CELLS_COLS-1 - self._x_coord])
            min_y = min([self._y_coord, CELLS_ROWS-1 - self._y_coord])
            
            if (min_x <= min_y):
                action = LEFT if (min_x == self._x_coord) else RIGHT
            else:
                action = UP if (min_y == self._y_coord) else DOWN

        if (DIMENSION_2D==False):
            cell_x = int(self._x_coord)
            cell_y = int(self._y_coord)
            if (cells_matrix[cell_y][cell_x]._z_coord>=self._z_coord):
                action = RISE

        return action

    def off_map_move_2D(self, new_agent_pos, cells_matrix=None):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Check if the current agent (2D case) is outside the considered area of interest.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
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
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Check if the current agent (3D case) is outside the considered area of interest.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # agent_pos is a tuple (x,y,z)
        agent_x = new_agent_pos[0]
        agent_y = new_agent_pos[1]
        agent_z = new_agent_pos[2]

        cell_x = int(agent_x)
        cell_y = int(agent_y)

        if \
        ( (agent_x < LOWER_BOUNDS) or \
        (agent_y < LOWER_BOUNDS) or \
        (agent_z < MIN_UAV_HEIGHT) or \
        (agent_x >= CELLS_COLS) or \
        (agent_y >= CELLS_ROWS) or \
        (agent_z >= MAX_UAV_HEIGHT) or \
        (cells_matrix[cell_y][cell_x]==HOSP_IN) or \
        (cells_matrix[cell_y][cell_x]==HOSP_AND_CS_IN) or \
        (cells_matrix[cell_y][cell_x]==OBS_IN) ):
            
            return True
        
        else:
            
            return False

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
            z_cs = closest_cs_cell._z_coord
            self._cs_goal = (closest_cs_cell._x_coord, closest_cs_cell._y_coord, z_cs) # --> SIDE-EFFECT on attribute 'cs_goal'
        else:
            self._cs_goal = (closest_cs_cell._x_coord, closest_cs_cell._y_coord) # --> SIDE-EFFECT on attribute 'cs_goal'

        return distances_from_current_position

    @staticmethod
    def setting_agents_pos(cs_points_or_cells):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Each UAV position is set according to the following rules:                                                                                                    #
        #       - IF 'N_CS' == 'N_UAVS', THEN each UAV will be placed on a different CS;                                                                                #
        #       - IF 'N_CS' < 'N_UAVS', THEN:                                                                                                                           #
        #                           IF 'N_CS' is divisible by 'N_UAVS', THEN equal number of UAVS will be placed on each CS;                                            #
        #                           OTHERWISE an equal number of UAV will be set on 'N_CS -1' charging stations, and the remaining UAVs will be placed on the last CS;  #
        #       - 'N_CS' can not be >= 'N_UAVS'.                                                                                                                        #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

        UAVS_ON_EACH_CS = N_UAVS//N_CS
        REMAINING_UAVS_ON_LAST_CS = N_UAVS%N_CS
        last_CS_index = N_CS - 1
        start_uav_idx = 0
        end_uav_idx = UAVS_ON_EACH_CS
        uavs_initial_pos = []

        if N_CS > N_UAVS:
            print("Invalid Setting: Number of charging stations exceeds number of drones!")

        for CS_i in range(N_CS):

            if (CS_i == last_CS_index):
                end_uav_idx += REMAINING_UAVS_ON_LAST_CS
            
            for UAV_i in range(start_uav_idx, end_uav_idx):
                uavs_initial_pos.append((cs_points_or_cells[CS_i]._x_coord, cs_points_or_cells[CS_i]._y_coord, cs_points_or_cells[CS_i]._z_coord)) # The value 0.5 indicates that the UAV is assumed to be in the middle of a point or a cell.

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
        agents = [Agent((pos[0], pos[1], pos[2]), 1, 0, 1, UAV_BANDWIDTH, FULL_BATTERY_LEVEL, ACTUAL_UAV_FOOTPRINT, max_uav_height, action_set, False, False, False, 2) for pos in agents_pos]
        for id_num in range(N_UAVS):
            agents[id_num]._uav_ID = id_num

        return agents

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
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Battery level increment when the agent is charging. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        self._battery_level += BATTERY_CHARGED_PER_IT
        if (self._battery_level > FULL_BATTERY_LEVEL):
            self._battery_level = FULL_BATTERY_LEVEL

    def users_in_uav_footprint(self, users, uav_footprint, discovered_users):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Compute the users inside the UAVs footprints (case with infinite bandwidth).  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        uav_x = self._x_coord
        uav_y = self._y_coord

        users_in_footprint = []
        if (MULTI_SERVICE==True):
            self._bandwidth = UAV_BANDWIDTH
            bandwidth_request_in_current_footprint = 0
        for user in users:
            user_x = user._x_coord
            user_y = user._y_coord
            
            if (MULTI_SERVICE==True):
                self._throughput_request = False
                self._edge_computing = False
                self._data_gathering = False

            if ( LA.norm(np.array([uav_x, uav_y]) - np.array([float(user_x), float(user_y)])) < self._footprint ): #, user_zusers_in_uav_footprint_lim_band
                
                if (user not in discovered_users):
                    discovered_users.append(user) # --> SIDE-EFFECT on 'discovered_users'
                # Check if the current user inside the UAV footprint is not served OR if it is served yet (in both cases the current agent will serve this user):
                if ( (user._info[0]) and (user in self._users_in_footprint) ):

                    if (MULTI_SERVICE==False):
                        users_in_footprint.append(user)
                    
                    elif (MULTI_SERVICE==True):

                        # Check if the current user inside the UAV footprint is not served OR if it is served yet; in both cases the current agent will serve this user
                        if ( (self._bandwidth>=user._info[5]) or ((not user._info[0]) and (self._bandwidth>=user._info[5])) ):
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

                elif (not user._info[0]):
                    if (MULTI_SERVICE==False):
                        users_in_footprint.append(user)

        if MULTI_SERVICE==False:
            return users_in_footprint, None
        else:
            return users_in_footprint, bandwidth_request_in_current_footprint

    def check_if_on_CS(self):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Check if the current agent is on a charging station or not. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # In this method it is assumed that all the users inside the UAV footprint are served;                      #
        # every user has the same priority and ask for a single or multi-service (according to the selected case).  #
        # A possible limitaton on the bandwidth is taken into account in 'users_in_uav_footprint' method.           #
        # It simply returns the number of the users inside the current UAv footprint.                         #                                     
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # This method could be redundant if no heuristic function is provided
        if (MULTI_SERVICE==True):
            TS_service = 0
            EC_service = 0
            DG_service = 0

        # Set the provided services and served users:
        for user in users_in_foot:
            
            # You could also set a heuristic function to select the user to serve according to the requested service.            
            if (True): # --> assume that every user inside the UAV footprint is served and that every available service is provided.
                user._info[0] = True

                if (MULTI_SERVICE==True):
                    if (user._info[1] == THROUGHPUT_REQUEST): TS_service += 1
                    elif (user._info[1] == EDGE_COMPUTING): CS_service += 1
                    elif (user._info[1] == DATA_GATHERING): DG_service += 1
            
            else:
                user._info[0] = False

        # UAV serves all the users inside its footprint:
        served_users = len(users_in_foot)

        return served_users

    @staticmethod
    def set_not_served_users(users, all_users_in_all_foots, serving_uav_id, QoEs_store, current_iteration, current_provided_services):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Set the users which are not served in case of:                                                      #
        #   - continuous and infinite service request;                                                        #
        #   - discrete and variable service request;                                                          #
        # 'all_users_in_all_foots' is a list containing all the users inside all the footprints of each UAVs. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Set the users not served (during a single iteration of each UAV):
        for user in users:
            
            if (not user in all_users_in_all_foots):
                user._info[0] = False

            # Update the info related to the current user (only when all the UAVs have performed their actions):
            if (serving_uav_id==N_UAVS):
                if (MULTI_SERVICE==False):
                    user.user_info_update_inf_request(QoEs_store, current_iteration) # --> Obviously 'current_provided_services' is not used in this case
                else:
                    user.user_info_update(QoEs_store, current_iteration, current_provided_services)

# ______________________________________________________________________________________________________________________________________________________________
# I) BATTERY CONSUMPTION: only if propulsion and UAVs services are considered together in the same and unique (average) consumption.

    def residual_battery1(self, move_action):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Battery level decrement after motion (and service at the same time).  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        self._battery_level -= PERC_CONSUMPTION_PER_ITERATION

    def needed_battery_to_come_home(self):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Battery percentage needed to go to the closesest charging station.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        needed_battery_to_cs = self._n_step_to_the_closest_cs*PERC_BATTERY_TO_GO_TO_CS

        return needed_battery_to_cs

    def residual_battery_when_come_home(self):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Battery consumption when go to a charging station (without serving).  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        if (self._current_consumption_to_go_cs == PERC_CONSUMPTION_PER_ITERATION):
            self._battery_level -= PERC_CONSUMPTION_PER_ITERATION
            self._current_consumption_to_go_cs = 1
        
        else:
            self._current_consumption_to_go_cs += PERC_BATTERY_TO_GO_TO_CS
