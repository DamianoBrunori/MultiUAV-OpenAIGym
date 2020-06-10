# AGENT MAIN CLASSES AND METHODS DEFINITION RELATED TO IT.

# CENTROIDI, CLUSTER, UTENTI (e altro . . . ?) VA AGIORNATO AD OGNI ITERAZIONE --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

from my_utils import *
from scenario_objects import Cell, Point, Environment, User
from load_and_save_data import Loader
#import operator
import numpy as np
from math import tan, radians
from numpy import linalg as LA
import copy

class Agent:
    '''
    |--------------------------------------------------------------------------------------|
    |Define the agent by its coordinates, occupied cell, performing action and its distance|
    |from charging stations and users clusters.                                            |
    |--------------------------------------------------------------------------------------|
    '''

    def __init__(self, pos, ID, toward, action, bandwidth, battery_level, footprint, TR, EC, DG, d_ag_cc):
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
        self._users_in_footprint = [] 
        self._throughput_request = TR
        self._edge_computing = EC
        self._data_gathering = DG 
        self._d_ag_cc = d_ag_cc

    @property
    def _vector(self):
        return np.array([self._x_coord, self._y_coord, self._z_coord])

    def move_2D_limited_battery(self, old_agent_pos, move_action):

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # 2D motion;                                                            #
        # LIMITED UAV battery;                                                  #
        # constant battery consumption for both UAV motion and services;        #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        next_cell_x = self._x_coord
        next_cell_y = self._y_coord

        if (move_action == HOVERING):
            return (next_cell_x, next_cell_y)

        elif (move_action == LEFT):
            next_cell_x -= UAV_XY_STEP

        elif (move_action == RIGHT):
            next_cell_x += UAV_XY_STEP

        elif (move_action == UP):
            next_cell_y += UAV_XY_STEP

        elif (move_action == DOWN):
            next_cell_y -= UAV_XY_STEP

        new_agent_pos = (next_cell_x, next_cell_y)

        # SOLO PER VEDERE SE FUNZIONA IL PLOT DELLA MAPPA PER L'ANIMAZIONE: --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if self._battery_level <= 0:
            self._battery_level = 10

        if (self.off_map_move_2D(new_agent_pos)):
            new_agent_pos = old_agent_pos
            # Reduction battery level due to the agent motion: 

        # Constant reduction battery level due to UAV motion and the provided service:
        self.residual_battery()

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
            print("HOVERING")
            return (next_cell_x, next_cell_y)

        elif (move_action == LEFT):
            print("LEFT")
            next_cell_x -= UAV_XY_STEP

        elif (move_action == RIGHT):
            print("RIGHT")
            next_cell_x += UAV_XY_STEP

        elif (move_action == UP):
            print("UP")
            next_cell_y += UAV_XY_STEP

        elif (move_action == DOWN):
            print("DOWN")
            next_cell_y -= UAV_XY_STEP

        new_agent_pos = (next_cell_x, next_cell_y)

        if (self.off_map_move_2D(new_agent_pos)):
            new_agent_pos = old_agent_pos

        self._x_coord = new_agent_pos[0]
        self._y_coord = new_agent_pos[1]

        return new_agent_pos

    def move_3D_limited_battery(self, old_agent_pos, move_action): 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # 3D motion;                                                            #
        # LIMITED UAV battery                                                   #
        # constant battery consumption for both UAV motion and services;        #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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

        # SOLO PER VEDERE SE FUNZIONA IL PLOT DELLA MAPPA PER L'ANIMAZIONE: --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if self._battery_level <= 0:
            self._battery_level = 10

        if (self.off_map_move_3D(new_agent_pos)):
            new_agent_pos = old_agent_pos
        
        # Constant reduction battery level due to UAV motion and the provided service:
        self.residual_battery()

        self._x_coord = new_agent_pos[0]
        self._y_coord = new_agent_pos[1]
        self._z_coord = new_agent_pos[2]

        return new_agent_pos

    def move_3D_unlimited_battery(self, old_agent_pos, move_action): 
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

        if (self.off_map_move_3D(new_agent_pos)):
            new_agent_pos = old_agent_pos

        self._x_coord = new_agent_pos[0]
        self._y_coord = new_agent_pos[1]
        self._z_coord = new_agent_pos[2]

        return new_agent_pos

    def move(self, old_agent_pos, move_action): 
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

        if (self.off_map_move_3D(new_agent_pos)):
            new_agent_pos = old_agent_pos
            # Reduction battery level due to the agent motion: 
            self.residual_battery_after_propulsion(HOVERING)
        else:
            self.residual_battery_after_propulsion(move_action)

        if self._battery_level <= 0:
            self._battery_level = 10

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

    def off_map_move_2D(self, new_agent_pos):
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

    def off_map_move_3D(self, new_agent_pos):
        # agent_pos is a tuple (x,y,z)

        agent_x = new_agent_pos[0]
        agent_y = new_agent_pos[1]
        agent_z = new_agent_pos[2]

        if \
        ( (agent_x < LOWER_BOUNDS) or \
        (agent_y < LOWER_BOUNDS) or \
        (agent_z < MIN_UAV_HEIGHT) or \
        (agent_x >= CELLS_COLS) or \
        (agent_y >= CELLS_ROWS) or \
        (agent_z >= MAX_UAV_HEIGHT) ):

            return True

        else:

            return False 

    def compute_UAV_footprint_radius(self, z_coord, theta=30):
        # Compute the radius of the UAV footprint taking as input the
        # z coordinate of the UAV and its 'angle of view' 'theta'.

        theta_radians = radians(theta)
        radius = z_coord*tan(theta_radians)

        return radius

    def compute_distances(self, agent, desired_cells):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Compute the distance between the agent position and the position of specific cells; #
        # it returns a list of tuple in which the first item represents a cell and the second #
        # item is the distance of the considered cell from the current agent position.        #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        distances_from_current_position = [(cell, LA.norm(cell._vector - agent._vector)) for cell in desired_cells]
        # Order the list of tuples according to their second item (i.e. the distance from the current agent position): 
        distances_from_current_position.sort(key=lambda x: x[1])

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
    def initialize_agents(agents_pos):
        # # # # # # # # # # # # # # # # # # # # # # # # 
        # Initialize the agents on their first start; #
        # 'agents_pos' is a list of tuple (x,y,z).    #
        # # # # # # # # # # # # # # # # # # # # # # # #

        # 'x' and 'y' are derived from the integer part division used with the derired resolution cell (because we only know where the drone is according to the selected resolution): 
        agents = [Agent((pos[0]+0.5, pos[1]+0.5, pos[2]+0.5), 1, 0, 1, 4, N_BATTERY_LEVELS, UAV_FOOTPRINT, False, False, False, 2) for pos in agents_pos]
        
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

    def users_in_uav_footprint(self, users, uav_footprint):
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
            if ( LA.norm(np.array([uav_x, uav_y]) - np.array([float(user_x), float(user_y)])) < self._footprint ): #, user_z
                #if not user._info[0]:
                    users_in_footprint.append(user)
            '''
            # Check if the X coord of the considered user is inside the current UAV footprint:
            if ( (user_x >= space_to_check[0][0]) and (user_x <= space_to_check[0][1]) ):
                # Check if the Y coord of the considered user is inside the current UAV footprint:
                if ( (user_y >= space_to_check[1][0]) and (user_y <= space_to_check[1][1]) ):
                    if not user._info[0]:
                        users_in_footprint.append(user)
            '''

        return users_in_footprint

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
    def set_not_served_users(users, all_users_in_all_foots, users_served_time, user_request_service_elapsed_time):
        # 'all_users_in_all_foots' is a list containing all the users inside all the footprints of each UAVs. 

        # Set the users not served:
        for user in users:
            if not (user in all_users_in_all_foots):

                user._info[0] = False

            # Update the info related to the current user
            users_served_time, user_request_service_elapsed_time = user.user_info_update(users_served_time, user_request_service_elapsed_time)

    @staticmethod
    def set_not_served_users_inf_request(users, all_users_in_all_foots):
        # 'all_users_in_all_foots' is a list containing all the users inside all the footprints of each UAVs.

        # Set the users not served:
        for user in users:
            if not (user in all_users_in_all_foots):

                user._info[0] = False

            # Update the info related to the current user
            user.user_info_update_inf_request()
            
        #return users_served_time, user_request_service_elapsed_time


# ______________________________________________________________________________________________________________________________________________________________
# I) BATTERY CONSUMPTION: only if propulsion and UAVs services are considered together in the same and unique (average) consumption.

    def residual_battery(self):

        self._battery_level -= CONSUMPTION_PER_ITERATION

# ______________________________________________________________________________________________________________________________________________________________


# ______________________________________________________________________________________________________________________________________________________________
# II) BATTERY CONSUMPTION: only if propulsion and UAVs services have different battery consumption and a time variable is explicitly used during the training.

    def residual_battery_after_service(self):
        
        battery_consumption = 0

        if (self._throughput_request == True):
            battery_consumption += TR_BATTERY_CONSUMPTION
        if (self._edge_computing == True):
            battery_consumption += EC_BATTERY_CONSUMPTION

        self._battery_level -= battery_consumption

    
    def residual_battery_after_propulsion(self, action):
        
        if ( (action == LEFT) or (action == RIGHT) or (action == DROP) or (action == RISE) ):
            battery_consumption = 2
        # Case in which the agent perform an action among HOVERING, RISE and DROP:
        else:
            battery_consumption = 1

        self._battery_level -= battery_consumption

# _______________________________________________________________________________________________________________________________________________________________







# PROVA:

# Loading:
load = Loader()
load.maps_data()

cs_points = load.cs_points
cs_cells = load.cs_cells
enb_cells = load.enb_cells
print([(cs_cell._x_coord, cs_cell._y_coord, cs_cell._z_coord) for cs_cell in cs_cells])

uavs_pos = Agent.setting_agents_pos(cs_cells)
print("UAVS POS:", uavs_pos)
uavs_pos = Agent.initialize_agents(uavs_pos)
for uav in uavs_pos:
    print((uav._x_coord, uav._y_coord, uav._z_coord))

cell = Cell(0, 0, 1, 0, 0, 2)
agent = Agent((1,0,0), 1, 0, 1, 4, 100, None, False, False, False, 2)
print("SERVICE BATTERY CONSUMPTION", agent.residual_battery_after_service())
#print(cell._vector)
next_cell_coords = agent.move((0,0,0), LEFT)
CS_distances = agent.compute_distances(agent, cs_cells)
eNodeB_distance = agent.compute_distances(agent, enb_cells)
print("Disances between agent and charging stations:", CS_distances)
print("Disances between agent and eNodeB:", eNodeB_distance)

if next_cell_coords == -1:
    print("Motion not allowed")
else:
    print("Actual cell coordinates", (cell._x_coord, cell._y_coord, cell._z_coord))
    print("Next cell coordinates:", (next_cell_coords[0], next_cell_coords[1], next_cell_coords[2]))
