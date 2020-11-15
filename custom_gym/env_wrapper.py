import numpy as np
import sys
import pickle
from os import mkdir
from os.path import join, isdir
from numpy import linalg as LA
from math import sqrt, inf
from decimal import Decimal
import time
import gym
import envs
from gym import spaces, logger
from scenario_objects import Point, Cell, User, Environment
import plotting
from my_utils import *
import agent
from Astar import *

# _________ Main training parameters:_________

SHOW_EVERY = 30
LEARNING_RATE = 1.0
DISCOUNT = 0.95
EPSILON = 1.0
EPSILON_DECREMENT = 0.998
EPSILON_MIN = 0.01
EPSILON_MIN2 = 0.4

max_value_for_Rmax = 100
ITERATIONS_PER_EPISODE = 30

# _____________________________________________


# __________________ Main loadings: __________________

env = gym.make('UAVEnv-v0')
MAX_UAV_HEIGHT = env.max_uav_height
n_actions = env.nb_actions
actions_indeces = range(n_actions)
cs_cells = env.cs_cells
cs_cells_coords_for_UAVs = [(cell._x_coord, cell._y_coord) for cell in cs_cells] if DIMENSION_2D==True else [(cell._x_coord, cell._y_coord, cell._z_coord) for cell in cs_cells]
cells_matrix = env.cells_matrix
action_set_min = env.action_set_min
if (UNLIMITED_BATTERY==False):
    q_table_action_set = env.q_table_action_set
    charging_set = env.charging_set
    come_home_set = env.come_home_set

reset_uavs = env.reset_uavs
plot = plotting.Plot()
centroids = env.cluster_centroids
# Scale centroids according to the selected resolution:
env_centroids = [(centroid[0]/CELL_RESOLUTION_PER_COL, centroid[1]/CELL_RESOLUTION_PER_ROW) for centroid in centroids]

# _____________________________________________________

def show_and_save_info(q_table_init, q_table, dimension_space, battery_type, users_request, reward_func, case_directory):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Save the main settings in 'env_and_train_info.txt' file and show them before training start.  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    info = []

    info1 = "\n\n_______________________________________ENVIRONMENT AND TRAINING INFO: _______________________________________\n"
    info.append(info1)

    info2 = "\nTraining:\n"
    info.append(info2)
    info3 = "\nEPISODES: " + str(EPISODES)
    info.append(info3)
    info4 = "\nITERATIONS PER EPISODE: " + str(ITERATIONS_PER_EPISODE)
    info.append(info4)
    info5 = "\nINITIAL EPSILON: " + str(EPSILON)
    info.append(info5)
    info6 = "\nMINIMUM EPSILON: " + str(EPSILON_MIN)
    info.append(info6)
    info31 = "\nEPSILON DECREMENT: " + str(EPSILON_DECREMENT)
    info.append(info31)
    info7 = "\nLEARNING RATE: " + str(LEARNING_RATE)
    info.append(info7)
    info8 = "\nDISCOUNT RATE: " + str(DISCOUNT)
    info.append(info8)
    if (q_table_init=="Max Reward"):
        info9 = "\nQ-TABLE INITIALIZATION: " + q_table_init + " with a Rmax value equal to: " + str(max_value_for_Rmax)
    else:
        info9 = "\nQ-TABLE INITIALIZATION: " + q_table_init
    info.append(info9)
    info28 = "\nQ-TABLE DIMENSION PER UAV: " + str(len(q_table))
    info.append(info28)
    info29 = "\nREWARD FUNCTION USED: " + str(reward_func) + "\n\n"
    info.append(info29)

    info10 = "\nEnvironment:\n"
    info.append(info10)

    if dimension_space == "2D":
        info11 = "\nMAP DIMENSION AT MINIMUM RESOLUTION: " + str(AREA_WIDTH) + "X" + str(AREA_HEIGHT)
        info12 = "\nMAP DIMENSION AT DESIRED RESOLUTION: " + str(CELLS_ROWS) + "X" + str(CELLS_COLS)
        info.append(info11)
        info.append(info12)
    else:
        Z_DIM = MAX_UAV_HEIGHT-MIN_UAV_HEIGHT
        info11 = "\nMAP DIMENSION AT MINIMUM RESOLUTION: " + str(AREA_WIDTH) + "X" + str(AREA_HEIGHT) + "X" + str(Z_DIM)
        info12 = "\nMAP DIMENSION AT DESIRED RESOLUTION: " + str(CELLS_ROWS) + "X" + str(CELLS_COLS) + "X" + str(Z_DIM)
        info13 = "\nMINIMUM UAVs FLIGHT HEIGHT: " + str(MIN_UAV_HEIGHT)
        info14 = "\nMAXIMUM UAVs FLIGHT HEIGHT: " + str(MAX_UAV_HEIGHT)
        info32 = "\nMINIMUM COVERAGE PERCENTAGE OF OBSTACLES: " + str(MIN_OBS_PER_AREA*100) + " %"
        info33 = "\nMAXIMUM COVERAGE PERCENTAGE OF OBSTACELS: " + str(MAX_OBS_PER_AREA*100) + " %"
        info34 = "\nMAXIMUM FLIGHT HEIGHT OF A UAV: " + str(MAX_UAV_HEIGHT) + ", equal to the height of the highest obstacle"
        info35 = "\nMINIMUM FLIGHT HEIGHT OF A UAV: " + str(MIN_UAV_HEIGHT) + ", equal to the height of the Charging Stations"
        info36 = "\nUAV MOTION STEP ALONG Z-AXIS: " + str(UAV_Z_STEP)
        info.append(info36)
        info.append(info11)
        info.append(info12)
        info.append(info13)
        info.append(info14)
        info.append(info32)
        info.append(info33)
        info.append(info34)
        info.append(info35)
        info.append(info36)
    info15 = "\nUAVs NUMBER: " + str(N_UAVS)
    info.append(info15)
    if (dimension_space == "2D"):
        uavs_coords = ["UAV " + str(uav_idx+1) + ": " + str((env.agents[uav_idx]._x_coord, env.agents[uav_idx]._y_coord)) for uav_idx in range(N_UAVS)]
        info16 = "\nUAVs INITIAL COORDINATES: " + str(uavs_coords)
    else:
        uavs_coords = ["UAV " + str(uav_idx+1) + ": " + str((env.agents[uav_idx]._x_coord, env.agents[uav_idx]._y_coord, env.agents[uav_idx]._z_coord)) for uav_idx in range(N_UAVS)]
        info16 = "\nUAVs INITIAL COORDINATES: " + str(uavs_coords)
    info40 = "\n UAVs FOOTPRINT DIMENSION: " + str(ACTUAL_UAV_FOOTPRINT)
    info.append(info40) 
    info.append(info16)
    info17 = "\nUSERS CLUSTERS NUMBER: " + str(len(env.cluster_centroids))
    info30 = "\nUSERS INITIAL NUMBER: " + str(env.n_users)
    info.append(info17)
    info.append(info30)
    centroids_coords = ["CENTROIDS: " +  str(centroid_idx+1) + ": " + str((env.cluster_centroids[centroid_idx][0], env.cluster_centroids[centroid_idx][1])) for centroid_idx in range(len(env.cluster_centroids))]
    info18 = "\nUSERS CLUSTERS PLANE-COORDINATES: " + str(centroids_coords)
    info37 = "\nCLUSTERS RADIUSES: " + str(env.clusters_radiuses)
    info.append(info37)
    info.append(info18)
    info19 = "\nDIMENION SPACE: " + str(dimension_space)
    info.append(info19)
    info20 = "\nBATTERY: " + str(battery_type)
    info.append(info20)
    info21 = "\nUSERS SERVICE TIME REQUEST: " + str(users_request)
    info.append(info21)
    if (STATIC_REQUEST == True):
        info22 = "\nUSERS REQUEST: Static"
    else:
        info22 = "\nUSERS REQUEST: Dynamic"
    info.append(info22)
    if (USERS_PRIORITY == False):
        info23 = "\nUSERS ACCOUNTS: all the same"
    else:
        info23 = "\nUSERS ACCOUNTS: " + str(USERS_ACCOUNTS)
    info.append(info23)
    if (INF_REQUEST == True):
        # If the users service request is infite, then we assume that the UAVs are providing only one service.
        info24 = "\nNUMBER SERVICES PROVIDED BY UAVs: 1"
    else:
        info24 = "\nNUMBER SERVICES PROVIDED BY UAVs: 3"
    info.append(info24)
    if (UNLIMITED_BATTERY == True):
        info25 = "\nCHARGING STATIONS NUMBER: N.D."
    else:
        info25 = "\nCHARGING STATIONS NUMBER: " + str(N_CS)
        info_37 = "\nCHARGING STATIONS COORDINATES: " + str([(cell._x_coord, cell._y_coord, cell._z_coord) for cell in env.cs_cells])
        info.append(info_37)
        info38 = "\nTHRESHOLD BATTERY LEVEL PERCENTAGE CONSIDERED CRITICAL: " + str(PERC_CRITICAL_BATTERY_LEVEL)
        info.append(info38)
        info39 = "\nBATTERY LEVELS WHEN CHARGING SHOWED EVERY " + str(SHOW_BATTERY_LEVEL_FOR_CHARGING_INSTANT) + " CHARGES"
        info.append(info39)
    info.append(info25)
    if (CREATE_ENODEB == True):
        info26 = "\nENODEB: Yes"
    else:
        info26 = "\nENODEB: No"
    info.append(info26)
    info27 = "\n__________________________________________________________________________________________________________________\n\n"
    info.append(info27)

    file = open(join(saving_directory, "env_and_train_info.txt"), "w")

    for i in info:
        print(i)
        file.write(i)

    file.close()

    #time.sleep(5)

def compute_subareas(area_width, area_height, x_split, y_split):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Return 'xy_limits' and middle points for some subareas (oo the all considered map) which can be used for a Q-table initialization based on 'a priori knowledge'.  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    subW_min = subH_min = 0
    subW_max = area_width/x_split
    subH_max = area_height/y_split
    subareas_xy_limits = []
    subareas_middle_points = [] 

    for x_subarea in range(1, x_split+1):
        W_max = subW_max*x_subarea

        for y_subarea in range(1, y_split+1):
            H_max = subH_max*y_subarea

            x_limits = (subW_min, W_max)
            y_limits = (subH_min, H_max)
            subareas_xy_limits.append([x_limits, y_limits])
            # Compute the middle point of each subarea: 
            subareas_middle_points.append((subW_min + (W_max - subW_min)/2, subH_min + (H_max - subH_min)/2))
            
            subH_min = H_max

        subW_min = W_max
        subH_min = 0

    return subareas_xy_limits, subareas_middle_points

def compute_prior_rewards(agent_pos_xy, best_prior_knowledge_points):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Compute the values used to inizialized the Q-table based on 'a priori initialization'.  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    actions = env.q_table_action_set
    # Initialize a random agent just to use easily the 'move' methods for 2D and 3D cases:
    agent_test = agent.Agent((agent_pos_xy[0], agent_pos_xy[1], 0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # Compute all the possible positions according to the available actions
    if (DIMENSION_2D == True):
        # Prior knowledge obvoiusly does not take into account the battery level: 
        new_agent_pos_per_action = [agent_test.move_2D_unlimited_battery((agent_pos_xy[0], agent_pos_xy[1]), action) for action in actions]
    else:
        new_agent_pos_per_action = [agent_test.move_3D_unlimited_battery((agent_pos_xy[0], agent_pos_xy[1], agent_pos_xy[2]), action) for action in actions]
    prior_rewards = []
    for pos in new_agent_pos_per_action:
        current_distances_from_best_points = [LA.norm(np.array([pos[0], pos[1]]) - np.array(best_point)) for best_point in best_prior_knowledge_points]
        # The reference distance for the reward in the current state is based on the distance between the agent and the closer 'best point':
        current_reference_distance = min(current_distances_from_best_points)
        current_normalized_ref_dist = current_reference_distance/diagonal_area_value
        prior_rewards.append(1 - current_normalized_ref_dist)

    return prior_rewards

# _________________________________ Check if Q-table initialization is based on 'a prior knoledge' _________________________________

if (PRIOR_KNOWLEDGE == True):
    subareas_limits, subareas_middle_points = compute_subareas(CELLS_COLS, CELLS_ROWS, X_SPLIT, Y_SPLIT)
    best_prior_knowledge_points = []
    diagonal_area_value = sqrt(pow(CELLS_ROWS, 2) + pow(CELLS_COLS, 2)) # --> The diagonal is the maximum possibile distance between two points, and then it will be used to normalize the distance when the table is initialized under the assumption of 'prior knowledge'. 0.5 is used because the UAV is assumed to move from the middle of a cell to the middle of another one.

    for centroid in env_centroids:
        centroid_x = centroid[0]
        centroid_y = centroid[1]
        
        for subarea in range(N_SUBAREAS):
            current_subarea = subareas_limits[subarea]
            
            if ( ( (centroid_x >= current_subarea[0][0]) and (centroid_x < current_subarea[0][1]) ) and ( (centroid_y >= current_subarea[1][0]) and (centroid_y < current_subarea[1][1]) ) ):
                best_prior_knowledge_points.append(subareas_middle_points[subarea])

DEFAULT_CLOSEST_CS = (None, None, None) if DIMENSION_2D==False else (None, None)

# ___________________________________________________________________________________________________________________________________

def go_to_recharge(action, agent):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    # Perform SIDE EFFECT on '_path_to_the_closest_CS' and '_current_pos_in_path_to_CS'.  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # agent._coming_home is set equal to True inside 'move' method, which is called inside 'step' method (just after 'choose_action'). Thus, here it is not needed.
    closest_CS = agent._cs_goal
    # Compute the path to the closest CS only if needed:
    if (closest_CS == DEFAULT_CLOSEST_CS):
        _ = agent.compute_distances(cs_cells) # --> Update the closest CS just in case the agent need to go the CS (which is obviously the closest one).
        agent._path_to_the_closest_CS = astar(cells_matrix, get_agent_pos(agent), agent._cs_goal)
        agent._current_pos_in_path_to_CS = 0 # --> when compute the path, set to 0 the counter indicating the current position (which belongs to the computed path) you have to go. 

def choose_action(uavs_q_tables, which_uav, obs, agent, battery_in_CS_history, cells_matrix):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Return the chosen action according to the current set of action which depends on the considered scenario, battery level and other parameters:   #
    # SIDE EFFECT on different agents attributes are performed.                                                                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if ( (ANALYZED_CASE == 1) or (ANALYZED_CASE == 2) ):
        obs = tuple([round(ob, 1) for ob in obs])

    #obs = tuple([round(Decimal(ob), 1) for ob in obs])
    all_actions_values = [values for values in uavs_q_tables[which_uav][obs]]
    current_actions_set = agent._action_set
    
    if (UAV_STANDARD_BEHAVIOUR==False):

        if UNLIMITED_BATTERY == False:
            if (current_actions_set == action_set_min): # == ACTION_SPACE_3D_MIN
                all_actions_values[GO_TO_CS_INDEX] = -inf
                all_actions_values[CHARGE_INDEX] = -inf
            
            elif (current_actions_set == come_home_set): # == ACTION_SPACE_3D_COME_HOME

                if ( (agent._coming_home == True) and (get_agent_pos(agent)!=agent._cs_goal) ):
                    action = GO_TO_CS_INDEX
                    
                    if (Q_LEARNING==True):
                        agent._current_pos_in_path_to_CS += 1
                    
                    return action 
                
                elif (agent._coming_home == False):
                    all_actions_values[CHARGE_INDEX] = -inf
                    agent._required_battery_to_CS = agent.needed_battery_to_come_home()

                elif ( (agent._coming_home == True) and agent.check_if_on_CS()):
                    agent._n_recharges +=1
                    n_recharges = agent._n_recharges
                    
                    if (n_recharges%SHOW_BATTERY_LEVEL_FOR_CHARGING_INSTANT==0):
                        battery_in_CS_history.append(agent._battery_level)
                        
                    action = CHARGE_INDEX
                    
                    return action
            
            elif (current_actions_set == charging_set): # == ACTION_SPACE_3D_WHILE_CHARGING
                
                if ( (agent.check_if_on_CS()) and (agent._battery_level < FULL_BATTERY_LEVEL) ):
                    action = CHARGE_INDEX
                    # agent._coming_home is set equal to True inside 'move' method, which is called inside 'step' method (just after 'choose_action'). Thus, here it is not needed.
                    
                    return action
                
                elif (agent._battery_level >= FULL_BATTERY_LEVEL):
                    agent._battery_level = FULL_BATTERY_LEVEL
                    all_actions_values[CHARGE_INDEX] = -inf
                    all_actions_values[GO_TO_CS_INDEX] = -inf

    rand = np.random.random()

    if (UAV_STANDARD_BEHAVIOUR==False):
        if (rand > EPSILON):
            # Select the best action so far (based on the current action set of the agent, assign actions values to a reduce action set):
            action = np.argmax(all_actions_values)
            
        else:
            n_actions_to_not_consider = n_actions-agent._n_actions
            n_current_actions_to_consider = n_actions-n_actions_to_not_consider
            prob_per_action = 1/n_current_actions_to_consider
            probabilities = [prob_per_action if act_idx < n_current_actions_to_consider else 0.0 for act_idx in actions_indeces]
            # Select the action randomly:
            action = np.random.choice(actions_indeces, p=probabilities)
    else:
        
        if (UNLIMITED_BATTERY == False):
            
            if ( (agent._coming_home == True) and (get_agent_pos(agent) in cs_cells_coords_for_UAVs) ):
                action = CHARGE

            elif ( (agent._charging == True) and (agent._battery_level < FULL_BATTERY_LEVEL) ):
                action = CHARGE

            elif (agent._battery_level <= CRITICAL_BATTERY_LEVEL):
                action = GO_TO_CS
                agent._current_pos_in_path_to_CS += 1
                go_to_recharge(GO_TO_CS_INDEX, agent)

            elif (which_uav==2):
                action = agent.action_for_standard_h(cells_matrix)
                
            elif (which_uav==1):
                action = agent.action_for_standard_v(cells_matrix)
                
            elif (which_uav==0):
                action = agent.action_for_standard_square_clockwise(cells_matrix)

        else:

            if (which_uav==2):
                action = agent.action_for_standard_h(cells_matrix)
                
            elif (which_uav==1):
                action = agent.action_for_standard_v(cells_matrix)
                
            elif (which_uav==0):
                action = agent.action_for_standard_square_clockwise(cells_matrix)

        return action

    if (action == GO_TO_CS_INDEX):
        go_to_recharge(action, agent)

    return action

def obs_fun(obs1=None, obs2=None, obs3=None):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Return the observation according to the considered case.  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if (HOW_MANY_OBSERVATIONS == 1):
        return (obs1)
    elif (HOW_MANY_OBSERVATIONS == 2):
        return (obs1, obs2)
    elif (HOW_MANY_OBSERVATIONS == 3):
        return (obs1, obs2, obs3)
    else:
        return

# ________________________________________ Assign a number_id to each analyzed case in order to initialize the proper Q-table based on te current case: ________________________________________

ANALYZED_CASE = 0

if ( (USERS_PRIORITY == False) and (CREATE_ENODEB == False) ):
    # 2D case with UNLIMITED UAVs battery autonomy:
    if ( (DIMENSION_2D == True) and (UNLIMITED_BATTERY == True) ):
        ANALYZED_CASE = 1
        HOW_MANY_OBSERVATIONS = 1
        get_agent_pos = env.get_2Dagent_pos
        step = env.step_2D_unlimited_battery
        considered_case_directory = "2D_un_bat"
        dimension_space = "2D"
        battery_type = "Unlimited"
        reward_func = "Reward function 1"

    # 2D case with LIMITED UAVs battery autonomy:
    elif ( (DIMENSION_2D == True) and (UNLIMITED_BATTERY == False) ):
        ANALYZED_CASE = 2
        HOW_MANY_OBSERVATIONS = 2
        get_agent_pos = env.get_2Dagent_pos
        step = env.step_2D_limited_battery
        considered_case_directory = "2D_lim_bat"
        dimension_space = "2D"
        battery_type = "Limited"
        reward_func = "Reward function 2"

    # 3D case with UNLIMITED UAVs battery autonomy:
    elif ( (DIMENSION_2D == False) and (UNLIMITED_BATTERY == True) ):
        ANALYZED_CASE = 3
        HOW_MANY_OBSERVATIONS = 1
        get_agent_pos = env.get_3Dagent_pos
        step = env.step_3D_unlimited_battery
        considered_case_directory = "3D_un_bat"
        dimension_space = "3D"
        battery_type = "Unlimited"
        reward_func = "Reward function 1"

    # 3D case with LIMITED UAVs battery autonomy:
    elif ( (DIMENSION_2D == False) and (UNLIMITED_BATTERY == False) ):
        ANALYZED_CASE = 4
        HOW_MANY_OBSERVATIONS = 2
        get_agent_pos = env.get_3Dagent_pos
        step = env.step_3D_limited_battery
        considered_case_directory = "3D_lim_bat"
        dimension_space = "3D"
        battery_type = "Limited"
        reward_func = "Reward function 2"

    if (INF_REQUEST == True):
        setting_not_served_users = agent.Agent.set_not_served_users_inf_request
        service_request_per_epoch = env.n_users*ITERATIONS_PER_EPISODE
        considered_case_directory += "_inf_req"
        users_request = "Continue"
    else:
        setting_not_served_users = agent.Agent.set_not_served_users
        service_request_per_epoch = 0 # --> TO SET --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        considered_case_directory += "_lim_req"
        users_request = "Discrete"
        if (MULTI_SERVICE==True):
            considered_case_directory += "_multi_service_limited_bandwidth"
            step = env.step_3D_limited_battery_multi_service_limited_bandwidth
            reward_func = "Reward function 3"

else:

    #CREATE_ENODEB = False
    #DIMENSION_2D = False
    #UNLIMITED_BATTERY = True
    #INF_REQUEST = True
    #STATIC_REQUEST = True
    #USERS_PRIORITY = False
    assert False, "Environment parameters combination not implemented yet: STATIC_REQUEST: %s, DIMENSION_2D: %s, UNLIMITED_BATTERY: %s, INF_REQUEST: %s, USERS_PRIORITY: %s, CREATE_ENODEB: %s"%(STATIC_REQUEST, DIMENSION_2D, UNLIMITED_BATTERY, INF_REQUEST, USERS_PRIORITY, CREATE_ENODEB)
    pass # TO DO --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ______________________________________________________________________________________________________________________________________________________________________________________________


# ________________________________________ Directory cration to save images and data: ________________________________________

considered_case_directory += "_" + str(N_UAVS) + "UAVs" + "_" + str(len(env.cluster_centroids)) + "clusters"

cases_directory = "Cases"
if (R_MAX == True):
    sub_case_dir = "Max Initialization"
    q_table_init = "Max Reward"
elif (PRIOR_KNOWLEDGE == True):
    sub_case_dir = "Prior Initialization"
    q_table_init = "Prior Knowledge"
else:
    sub_case_dir = "Random Initialization"
    q_table_init = "Random Reward"

saving_directory = join(cases_directory, considered_case_directory, sub_case_dir)

if not isdir(cases_directory): mkdir(cases_directory)
if not isdir(join(cases_directory, considered_case_directory)): mkdir(join(cases_directory, considered_case_directory))
if not isdir(saving_directory): mkdir(saving_directory)

# _____________________________________________________________________________________________________________________________


# ________________________________________________________________________________ Q-table initialization: ________________________________________________________________________________________________

map_width = CELLS_COLS
map_length = CELLS_ROWS
map_height = MAXIMUM_AREA_HEIGHT

# Agents initialization:
agents = env.agents

# It is called Q-table, but if SARSA algorithm is used, it will be actually a SARSA-table:
uavs_q_tables = None

if uavs_q_tables is None:

    print("Q-TABLES INITIALIZATION . . .")
    uavs_q_tables = [None for uav in  range(N_UAVS)]
    explored_states_q_tables = [None for uav in range(N_UAVS)]
    uav_counter = 0
    
    for uav in range(N_UAVS):
        current_uav_q_table = {}
        current_uav_explored_table = {}
        
        for x_agent in np.arange(0, map_width+1, 0.2): #range(map_width)

            for y_agent in np.arange(0, map_length+1, 0.2): #range(map_length)

                x_agent = round(x_agent, 1)
                y_agent = round(y_agent, 1)

                # 2D case with UNLIMITED UAVs battery autonomy:
                if (ANALYZED_CASE == 1):
                    
                    if (PRIOR_KNOWLEDGE == True):
                        prior_rewards = compute_prior_rewards((x_agent, y_agent), best_prior_knowledge_points)
                        current_uav_q_table[(x_agent, y_agent)] = [prior_rewards[action] for action in range(n_actions)]
                    elif (R_MAX == True):
                        current_uav_q_table[(x_agent, y_agent)] = [max_value_for_Rmax for action in range(n_actions)]
                    else:
                        current_uav_q_table[(x_agent, y_agent)] = [np.random.uniform(0, 1) for action in range(n_actions)]                   

                    current_uav_explored_table[(x_agent, y_agent)] = [False for action in range(n_actions)]

                # 2D case with LIMITED UAVs battery autonomy:
                elif (ANALYZED_CASE == 2):
                    for battery_level in np.arange(0, FULL_BATTERY_LEVEL+1, PERC_CONSUMPTION_PER_ITERATION):

                        if (PRIOR_KNOWLEDGE == True):
                            prior_rewards = compute_prior_rewards((x_agent, y_agent), best_prior_knowledge_points)
                            current_uav_q_table[((x_agent, y_agent), battery_level)] = [(1 - prior_rewards) for action in range(n_actions)]
                        elif (R_MAX == True):
                            current_uav_q_table[((x_agent, y_agent), battery_level)] = [max_value_for_Rmax for action in range(n_actions)]
                        else:
                            current_uav_q_table[((x_agent, y_agent), battery_level)] = [np.random.uniform(0, 1) for action in range(n_actions)]

                        current_uav_explored_table[(x_agent, y_agent), battery_level] = [False for action in range(n_actions)]                     
                
                # 3D case with UNLIMITED UAVs battery autonomy:
                elif (ANALYZED_CASE == 3):
                    for z_agent in range(MIN_UAV_HEIGHT, MAX_UAV_HEIGHT, UAV_Z_STEP):

                        if (PRIOR_KNOWLEDGE == True):
                            prior_rewards = compute_prior_rewards((x_agent, y_agent), best_prior_knowledge_points)
                            current_uav_q_table[(x_agent, y_agent, z_agent)] = [(1 - prior_rewards) for action in range(n_actions)]
                        elif (R_MAX == True):
                            current_uav_q_table[(x_agent, y_agent, z_agent)] = [max_value_for_Rmax for action in range(n_actions)]
                        else:
                            current_uav_q_table[(x_agent, y_agent, z_agent)] = [np.random.uniform(0, 1) for action in range(n_actions)]

                        current_uav_explored_table[(x_agent, y_agent, z_agent)] = [False for action in range(n_actions)]

                # 3D case with LIMITED UAVs battery autonomy:
                elif (ANALYZED_CASE == 4):
                    if (UAV_STANDARD_BEHAVIOUR == False):
                        range_for_z = range(MIN_UAV_HEIGHT, MAX_UAV_HEIGHT, UAV_Z_STEP)
                    else:
                        range_for_z = range(MIN_UAV_HEIGHT, MAX_UAV_HEIGHT+UAV_Z_STEP+1, UAV_Z_STEP)
                    for z_agent in range_for_z:

                        for battery_level in np.arange(0, FULL_BATTERY_LEVEL+1, PERC_CONSUMPTION_PER_ITERATION):
                            
                            if (PRIOR_KNOWLEDGE == True):
                                prior_rewards = compute_prior_rewards((x_agent, y_agent), best_prior_knowledge_points)
                                current_uav_q_table[((x_agent, y_agent, z_agent), battery_level)] = [(1 - prior_rewards) for action in range(n_actions)]
                            elif (R_MAX == True):
                                current_uav_q_table[((x_agent, y_agent, z_agent), battery_level)] = [max_value_for_Rmax for action in range(n_actions)]
                            else:
                                current_uav_q_table[((x_agent, y_agent, z_agent), battery_level)] = [np.random.uniform(0, 1) for action in range(n_actions)]
                            
                            current_uav_explored_table[(x_agent, y_agent, z_agent), battery_level] = [False for action in range(n_actions)]

        uavs_q_tables[uav] = current_uav_q_table
        explored_states_q_tables[uav] = current_uav_explored_table
        print("Q-Table for Uav ", uav, " created")

    print("Q-TABLES INITIALIZATION COMPLETED.")

else:

    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

# _________________________________________________________________________________________________________________________________________________________________________________________________________

show_and_save_info(q_table_init, uavs_q_tables[0], dimension_space, battery_type, users_request, reward_func, saving_directory)

q_tables_dir = "QTables"
q_tables_directory = join(saving_directory, q_tables_dir)

uav_ID = "UAV"
uavs_directories = [0 for uav in range(N_UAVS)]
q_tables_directories = [0 for uav in range(N_UAVS)]
for uav in range(1, N_UAVS+1):
    current_uav_dir = join(saving_directory, uav_ID + str(uav))
    if not isdir(current_uav_dir): mkdir(current_uav_dir)
    uavs_directories[uav-1] = current_uav_dir
    current_q_table_dir = join(current_uav_dir, q_tables_dir)
    q_tables_directories[uav-1] = join(current_uav_dir, q_tables_dir) 
    if not isdir(current_q_table_dir): mkdir(current_q_table_dir)
uav_directory = uav_ID

# ________________________________________ Variable initialization before training loop: __________________________________________

uavs_episode_rewards = [[] for uav in range(N_UAVS)]
agents_paths = [[0 for iteration in range(ITERATIONS_PER_EPISODE)] for uav in range(N_UAVS)]
users_in_foots = [[] for uav in range(N_UAVS)]

avg_QoE1_per_epoch = [0 for ep in range(EPISODES)]
avg_QoE2_per_epoch = [0 for ep in range(EPISODES)]
avg_QoE3_per_epoch = [0 for ep in range(EPISODES)]

q_values = [[] for episode in range(N_UAVS)]

if (DIMENSION_2D == True):
    GO_TO_CS_INDEX = GO_TO_CS_2D_INDEX 
    CHARGE_INDEX = CHARGE_2D_INDEX
    set_action_set = env.set_action_set2D
else:
    CHARGE_INDEX = CHARGE_3D_INDEX
    CHARGE_INDEX_WHILE_CHARGING =  CHARGE_3D_INDEX_WHILE_CHARGING
    GO_TO_CS_INDEX = GO_TO_CS_3D_INDEX
    GO_TO_CS_INDEX_HOME_SPACE = GO_TO_CS_3D_INDEX_HOME_SPACE
    set_action_set = env.set_action_set3D

#time.sleep(5)

epsilon_history = [0 for ep in range(EPISODES)]
crashes_history = [0 for ep in range(EPISODES)] 
battery_in_CS_history = [[] for uav in range(N_UAVS)]
n_active_users_per_epoch = [0 for ep in range(EPISODES)]
provided_services_per_epoch = [[0, 0, 0] for ep in range(EPISODES)]
n_active_users_per_episode = [0 for ep in range(EPISODES)]
UAVs_used_bandwidth = [[0 for ep in range(EPISODES)] for uav in range(N_UAVS)]
users_bandwidth_request_per_UAVfootprint = [[0 for ep in range(EPISODES)] for uav in range(N_UAVS)]

MOVE_USERS = False

# ________________________________________________________________________________ Training start: _____________________________________________________________________________________________________________

print("\nSTART TRAINING . . .\n")
for episode in range(1, EPISODES+1):

    if (STATIC_REQUEST==False):
        if (episode%MOVE_USERS_EACH_N_EPOCHS==0):
            env.compute_users_walk_steps()
            MOVE_USERS = True
        else:
            MOVE_USERS = False

    if (INF_REQUEST==False):
        if (episode%UPDATE_USERS_REQUESTS_EACH_N_ITERATIONS==0):
            env.update_users_requests(env.users)

    epsilon_history[episode-1] = EPSILON

    print("| EPISODE: {ep:3d} | Epsilon: {eps:6f}".format(ep=episode, eps=EPSILON))

    QoEs_store = [[], []] # --> Store QoE1 and QoE2
    current_QoE3 = 0
    users_served_time = 0
    users_request_service_elapsed_time = 0
    current_provided_services = [0, 0, 0]
    current_UAV_bandwidth = [0 for uav in range(N_UAVS)]
    current_requested_bandwidth = [0 for uav in range(N_UAVS)]
    uavs_episode_reward = [0 for uav in range(N_UAVS)]
    crashes_current_episode = [False for uav in range(N_UAVS)]
    q_values_current_episode = [0 for uav in range(N_UAVS)] 
    n_active_users_current_it = 0 
    tr_active_users_current_it = 0
    ec_active_users_current_it = 0
    dg_active_users_current_it = 0
    
    # Each 30 minutes: 
    for i in range(ITERATIONS_PER_EPISODE):

        if (INF_REQUEST==True):
            n_active_users = env.n_users

        else:
            n_active_users, tr_active_users, ec_active_users, dg_active_users, n_tr_served, n_ec_served, n_dg_served = env.get_active_users()
            tr_active_users_current_it += tr_active_users
            ec_active_users_current_it += ec_active_users
            dg_active_users_current_it += dg_active_users

        n_active_users_current_it += n_active_users

        if (MOVE_USERS==True):
            env.move_users(i+1)

        all_users_in_all_foots = [] 
        for UAV in range(N_UAVS):
            # Skip analyzing the current UAV features until it starts to work (you can set a delayed start for each uav):
            current_iteration = i+1
            if (episode==1):
                # Case in which are considered the UAVs from the 2-th to the n-th:
                if (UAV>0):
                    if (current_iteration!=(DELAYED_START_PER_UAV*(UAV))):
                        pass
                        #continue

            agents_paths[UAV][i] = get_agent_pos(agents[UAV])
            if (NOISE_ON_POS_MEASURE==True):
                drone_pos = env.noisy_measure_or_not(get_agent_pos(agents[UAV]))
            else:
                drone_pos = agents_paths[UAV][i]
            
            obs = obs_fun(drone_pos, agents[UAV]._battery_level) #, agents[UAV]._battery_level) # --> CHANGE THE OBSERVATION WHEN SWITCH FROM 2D TO 3D ENV AND VICEVERSA!!!!!!!!!!!!!!!!!!!!!

            if (UNLIMITED_BATTERY==False):
                set_action_set(agents[UAV])

            action = choose_action(uavs_q_tables, UAV, obs, agents[UAV], battery_in_CS_history[UAV], cells_matrix)
            obs_, reward, done, info = step(agents[UAV], action, all_users_in_all_foots, env.users, setting_not_served_users, crashes_current_episode, cells_matrix)
            crashes_current_episode[UAV] = agents[UAV]._crashed
            
            print(" - Iteration: {it:1d} - Reward per UAV {uav:1d}: {uav_rew:6f}".format(it=i+1, uav=UAV+1, uav_rew=reward), end="\r", flush=True)

            if (UAV_STANDARD_BEHAVIOUR==True):
                action = ACTION_SPACE_STANDARD_BEHAVIOUR.index(action)

            '''
            if CONTINUOUS_TIME==True:
                obs_aux = obs
                obs_aux_ = obs_
                obs = tuple([round(Decimal(ob), 1) for ob in obs])
                obs_ = tuple([round(Decimal(ob), 1) for ob in obs_])
            '''

            obs = tuple([round(ob, 1) for ob in obs])
            obs_ = tuple([round(ob, 1) for ob in obs_])

            if not explored_states_q_tables[UAV][obs_][action]:
                explored_states_q_tables[UAV][obs_][action] = True

            if (info=="IS CHARGING"):
                continue
            
            # Set all the users which could be no more served after the current UAV action (use different arguments according to the considered case!!!!!!!!!!!!!!!!!!):
            #setting_not_served_users(env.users, all_users_in_all_foots, current_provided_services, UAV+1, QoEs_store, i+1)
            setting_not_served_users(env.users, all_users_in_all_foots, UAV+1, QoEs_store, i+1) # --> This make a SIDE_EFFECT on users by updating their info.

            if (Q_LEARNING==True):
                
                max_future_q = np.max(uavs_q_tables[UAV][obs_])
                current_q = uavs_q_tables[UAV][obs][action]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            
            else:
                
                if (SARSA==True):
                    action_ = choose_action(uavs_q_tables, UAV, obs_, agents[UAV], battery_in_CS_history[UAV], cells_matrix)
                    future_reward = uavs_q_tables[UAV][obs_][action_]
                    current_q = uavs_q_tables[UAV][obs][action]
                    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * future_reward)
                
                else:
                    assert False, "Invalid algorithm selection."            
            
            q_values_current_episode[UAV] = new_q
            uavs_q_tables[UAV][obs][action] = new_q
            uavs_episode_reward[UAV] += reward

            current_UAV_bandwidth[UAV] += UAV_BANDWIDTH - agents[UAV]._bandwidth
            current_requested_bandwidth[UAV] += env.current_requested_bandwidth

            #obs = obs_aux
            #obs_ = obs_aux_

            reset_uavs(agents[UAV])

        current_QoE3 += len(all_users_in_all_foots)/len(env.users) if len(env.users)!=0 else 0 # --> Percentage of covered users (including also the users which are not requesting a service but which are considered to have the communication with UAVs on) 
        n_active_users_per_episode[episode-1] = n_active_users

    if (INF_REQUEST==False): 
        tr_active_users_current_ep = tr_active_users_current_it/ITERATIONS_PER_EPISODE
        ec_active_users_current_ep = ec_active_users_current_it/ITERATIONS_PER_EPISODE
        dg_active_users_current_ep = dg_active_users_current_it/ITERATIONS_PER_EPISODE
        
        for service_idx in range(N_SERVICES):
            n_users_provided_for_current_service = current_provided_services[service_idx]/ITERATIONS_PER_EPISODE
            
            if (service_idx==0):
                n_active_users_per_current_service = tr_active_users_current_ep #tr_active_users
            elif (service_idx==1):
                n_active_users_per_current_service = ec_active_users_current_ep #ec_active_users
            elif (service_idx==2):
                n_active_users_per_current_service = dg_active_users_current_ep #dg_active_users
            
            perc_users_provided_for_current_service = n_users_provided_for_current_service/n_active_users_per_current_service if n_active_users_per_current_service!=0 else 0  
            provided_services_per_epoch[episode-1][service_idx] = perc_users_provided_for_current_service
        
    if (UNLIMITED_BATTERY==False):
        crashes_history[episode-1] = crashes_current_episode
    
    n_active_users_current_ep = n_active_users_current_it/ITERATIONS_PER_EPISODE
    users_served_time = sum(QoEs_store[0])/(len(QoEs_store[0])) if len(QoEs_store[0])!=0 else 0 # --> It is divided by its lenght, because here it is measured the percenteage according to which a service is completed (once it starts).
    users_request_service_elapsed_time = sum(QoEs_store[1])/n_active_users_current_ep if n_active_users_current_ep!=0 else 0 # --> It is divided by the # of active users, beacuse here it is measured the avg elapsed time (among the active users) between a service request and its provision.        
    
    QoE3_for_current_epoch = current_QoE3/ITERATIONS_PER_EPISODE
    User.avg_QoE(episode, users_served_time, users_request_service_elapsed_time, QoE3_for_current_epoch, avg_QoE1_per_epoch, avg_QoE2_per_epoch, avg_QoE3_per_epoch) # --> users_request_service_elapsed_time has to be divided by the iterations number if you do not have the mean value!!!

    print(" - Iteration: {it:1d} - Reward per UAV {uav:1d}: {uav_rew:6f}".format(it=i+1, uav=UAV+1, uav_rew=reward))
    
    for UAV in range(N_UAVS):
        UAVs_used_bandwidth[UAV][episode-1] = current_UAV_bandwidth[UAV]/ITERATIONS_PER_EPISODE
        users_bandwidth_request_per_UAVfootprint[UAV][episode-1] = current_requested_bandwidth[UAV]/ITERATIONS_PER_EPISODE
        current_mean_reward = uavs_episode_reward[UAV]/ITERATIONS_PER_EPISODE
        uavs_episode_rewards[UAV].append(current_mean_reward)
        current_q_mean = q_values_current_episode[UAV]/ITERATIONS_PER_EPISODE
        q_values[UAV].append(current_q_mean)
        print(" - Mean reward per UAV{uav:3d}: {uav_rew:6f}".format(uav=UAV+1, uav_rew=current_mean_reward), end=" ")
    print() 

    if ((episode%500)==0): #(episode%250)==0 # --> You can change the condition to show (to save actually) or not the scenario of the current episode.
        print("\nSaving animation for episode:", episode)
        env.render(agents_paths, saving_directory, episode)
        plot.users_wait_times(env.n_users, env.users, saving_directory, episode)
        print("Animation saved.\n")
        
        agents_paths = [[0 for iteration in range(ITERATIONS_PER_EPISODE)] for uav in range(N_UAVS)] # --> Actually is should be useless because the values are overwritten on the previous ones.

    n_discovered_users = len(env.discovered_users)
    
    if ( (n_discovered_users/env.n_users) >= 0.85):
        EPSILON = EPSILON*EPSILON_DECREMENT if EPSILON > EPSILON_MIN else EPSILON_MIN
    else:
        EPSILON = EPSILON*EPSILON_DECREMENT if EPSILON > EPSILON_MIN else EPSILON_MIN2

file = open(join(saving_directory, "env_and_train_info.txt"), "a")

print("\nTRAINING COMPLETED.\n")

# ________________________________________________________________________________ Training end ________________________________________________________________________________________________________________


# ________________________________________________________________________________ Results Saving: ______________________________________________________________________________________________________
 
for uav_idx in range(N_UAVS):
    if (UNLIMITED_BATTERY==False):
        print("\nSaving battery levels when start to charge . . .")
        plot.battery_when_start_to_charge(battery_in_CS_history, uavs_directories)
        print("Battery levels when start to charge saved.")
        print("Saving UAVs crashes . . .")
        plot.UAVS_crashes(EPISODES, crashes_history, saving_directory)
        print("UAVs crashes saved.")

    list_of_lists_of_actions = list(explored_states_q_tables[uav_idx].values())
    actions_values = [val for sublist in list_of_lists_of_actions for val in sublist]
    file.write("\nExploration percentage of the Q-Table for UAV:\n")
    actual_uav_id = uav_idx+1
    value_of_interest = np.mean(actions_values)
    file.write(str(actual_uav_id) + ": " + str(value_of_interest))
    print("Exploration percentage of the Q-Table for UAV:\n", actual_uav_id, ":", value_of_interest)

for qoe_num in range(1,4):
    file.write("\nQoE" + str(qoe_num) + " : ")
    
    if (qoe_num==1):
        file.write(str(np.mean(avg_QoE1_per_epoch)))
    
    if (qoe_num==2):
        file.write(str(np.mean(avg_QoE2_per_epoch)))
    
    if (qoe_num==3):
        file.write(str(np.mean(avg_QoE3_per_epoch)))

file.close()

print("\nSaving QoE charts, UAVs rewards and Q-values . . .")
legend_labels = []
plot.QoE_plot(avg_QoE1_per_epoch, EPISODES, join(saving_directory, "QoE1"), "QoE1")
plot.QoE_plot(avg_QoE2_per_epoch, EPISODES, join(saving_directory, "QoE2"), "QoE2")
plot.QoE_plot(avg_QoE3_per_epoch, EPISODES, join(saving_directory, "QoE3"), "QoE3")

if (INF_REQUEST==False):
    plot.bandwidth_for_each_epoch(EPISODES, saving_directory, UAVs_used_bandwidth, users_bandwidth_request_per_UAVfootprint)
    plot.users_covered_percentage_per_service(provided_services_per_epoch, EPISODES, join(saving_directory, "Services Provision"))

for uav in range(N_UAVS):
    plot.UAVS_reward_plot(EPISODES, uavs_episode_rewards, saving_directory)
    plot.UAVS_reward_plot(EPISODES, q_values, saving_directory, q_values=True)

print("Qoe charts, UAVs rewards and Q-values saved.")
print("\nSaving Epsilon chart trend . . .")
plot.epsilon(epsilon_history, EPISODES, saving_directory)
print("Epsilon chart trend saved.")

print("Saving Q-Tables for episode", episode, ". . .")
for uav in range(1, N_UAVS+1):
    saving_qtable_name = q_tables_directories[uav-1] + f"/qtable-ep{episode}.npy"
    np.save(saving_qtable_name, uavs_q_tables)
print("Q-Tables saved.\n")

with open(join(saving_directory, "q_tables.pickle"), 'wb') as f:
    pickle.dump(uavs_q_tables, f)

print("Saving Min and Max values related to the Q-Tables for episode", episode, ". . .")
for uav in range(1, N_UAVS+1):
    plot.actions_min_max_per_epoch(uavs_q_tables, q_tables_directories[uav-1], episode, uav)
print("Min and Max values related to the Q-Tables for episode saved.\n")

# ________________________________________________________________________________________________________________________________________________________________________________________________________