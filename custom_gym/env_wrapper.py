import numpy as np
import sys
import pickle
from os import mkdir
from os.path import join, isdir
from numpy import linalg as LA
from math import sqrt, inf
import time
import gym
import envs
from gym import spaces, logger
from scenario_objects import Point, Cell, User, Environment
import plotting
from my_utils import *
import agent
from Astar import *

# ---------------------------------------------------------------------------------------------------------------------------------------

WIDTH = 400
HEIGHT = 400


#WIDTH = 1200
#HEIGHT = 600

# ---------------------------------------------------------------------------------------------------------------------------------------

REFRESH_TIME = 10
collided_item_prec = 0

SHOW_EVERY = 30
LEARNING_RATE = 1.0
DISCOUNT = 0.95
EPSILON = 1.0
EPSILON_DECREMENT = 0.998
EPSILON_MIN = 0.01
EPSILON_MIN2 = 0.4

max_value_for_Rmax = 100

ITERATIONS_PER_EPISODE = 30

env = gym.make('UAVEnv-v0')
MAX_UAV_HEIGHT = env.max_uav_height
n_actions = env.nb_actions
actions_indeces = range(n_actions)
cs_cells = env.cs_cells
cells_matrix = env.cells_matrix
action_set_min = env.action_set_min
if (UNLIMITED_BATTERY==False):
    q_table_action_set = env.q_table_action_set
    charging_set = env.charging_set
    come_home_set = env.come_home_set
#uavs_initial_pos = env.initial_uavs_pos
reset_uavs = env.reset_uavs
plot = plotting.Plot()
centroids = env.cluster_centroids
# Scale centroids according to the selected resolution:
env_centroids = [(centroid[0]/CELL_RESOLUTION_PER_COL, centroid[1]/CELL_RESOLUTION_PER_ROW) for centroid in centroids]

def show_and_save_info(q_table_init, q_table, dimension_space, battery_type, users_request, reward_func, case_directory):

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
    info.append(info16)
    info17 = "\nUSERS CLUSTERS NUMBER: " + str(len(env.cluster_centroids))
    info30 = "\nUSERS INITIAL NUMBER: " + str(env.n_users)
    info.append(info17)
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
            #print("EEEEEEEEEEEEEE", [x_limits, y_limits])
            # Compute the middle point of each subarea: 
            
            subareas_middle_points.append((subW_min + (W_max - subW_min)/2, subH_min + (H_max - subH_min)/2))
            #print("OOOOOOOOOOOOOO", (subW_min + (W_max - subW_min)/2, subH_min + (H_max - subH_min)/2))
            
            subH_min = H_max

        subW_min = W_max
        subH_min = 0



    '''
    for subarea in range(N_SUBAREAS):
        x_limits = (subW_min, subW_max)
        y_limits = (subH_min, subH_max)
        subareas_xy_limits.append([x_limits, y_limits])
        print("EEEEEEEEEEEEEE", [x_limits, y_limits])
        # Compute the middle point of each subarea: 
        subareas_middle_points.append((subW_min + (subW_max - subW_min)/2, subH_min + (subH_max - subH_min)/2))
        subW_min = subW_max
        subH_min = subH_max
        subW_max *= subarea
        subH_max *= subarea
    '''

    return subareas_xy_limits, subareas_middle_points

#env = UAVENV()
#us = User()
#load = Loader()

def compute_prior_rewards(agent_pos_xy, best_prior_knowledge_points):

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

    current_uav_q_table[(x_agent+0.5, y_agent+0.5)] = [(1 - normalized_ref_dist) for action in range(n_actions)]

if (PRIOR_KNOWLEDGE == True):
    subareas_limits, subareas_middle_points = compute_subareas(CELLS_COLS, CELLS_ROWS, X_SPLIT, Y_SPLIT)
    #print("SUBAREASSSSSSSSSSSS", subareas_limits)
    #print("MIDDLEEEEEEEEEEEEEE", subareas_middle_points)
    best_prior_knowledge_points = []
    diagonal_area_value = sqrt(pow(CELLS_ROWS, 2) + pow(CELLS_COLS, 2))-0.5 # --> The diagonal is the maximum possibile distance between two points, and then it will be used to normalize the distance when the table is initialized under the assumption of 'prior knowledge'. 0.5 is used because the UAV is assumed to move from the middle of a cell to the middle of another one.

    for centroid in env_centroids:
        centroid_x = centroid[0]
        centroid_y = centroid[1]
        #print("CENTROIDE:", (centroid_x, centroid_y))
        for subarea in range(N_SUBAREAS):
            current_subarea = subareas_limits[subarea]
            #print("SUBAREA", current_subarea)
            #print("BOUNDS:", ( (current_subarea[0][0], current_subarea[0][1]), (current_subarea[1][0], current_subarea[1][1]) ) )
            if ( ( (centroid_x >= current_subarea[0][0]) and (centroid_x < current_subarea[0][1]) ) and ( (centroid_y >= current_subarea[1][0]) and (centroid_y < current_subarea[1][1]) ) ):
                #print("CI SIAMOOOOOOOOOOOO")
                best_prior_knowledge_points.append(subareas_middle_points[subarea])

#print("N_USERS:", env.n_users)

DEFAULT_CLOSEST_CS = (None, None, None) if DIMENSION_2D==False else (None, None)

def choose_action(uavs_q_tables, which_uav, obs, agent, battery_in_CS_history):

    all_actions_values = [values for values in uavs_q_tables[which_uav][obs]]
    current_actions_set = agent._action_set
    
    # CONSIDERA ANCHE IL CASO 2D CON BATTERIA LIMITATA --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if UNLIMITED_BATTERY == False:
        if (current_actions_set == action_set_min): # == ACTION_SPACE_3D_MIN
            all_actions_values[GO_TO_CS_INDEX] = -inf
            all_actions_values[CHARGE_INDEX] = -inf
        
        elif (current_actions_set == come_home_set): # == ACTION_SPACE_3D_COME_HOME

            if ( (agent._coming_home == True) and (get_agent_pos(agent)!=agent._cs_goal) ):
                action = GO_TO_CS_INDEX
                agent._current_pos_in_path_to_CS += 1
                #print("AOOOOOOOOOOOOOOOOOOOOHHHHH", agent._current_pos_in_path_to_CS)
                return action 
            elif (agent._coming_home == False):
                all_actions_values[CHARGE_INDEX] = -inf
                agent._required_battery_to_CS = agent.needed_battery_to_come_home()

            elif ( (agent._coming_home == True) and agent.check_if_on_CS()):
                agent._n_recharges +=1
                n_recharges = agent._n_recharges
                if (n_recharges%SHOW_BATTERY_LEVEL_FOR_CHARGING_INSTANT==0):
                    battery_in_CS_history.append(agent._battery_level)
                    #print("N RICARICA:", n_recharges, "BATTERY HISTORY:", battery_in_CS_history)
                action = CHARGE_INDEX
                return action
        
        elif (current_actions_set == charging_set): # == ACTION_SPACE_3D_WHILE_CHARGING
            
            if ( (agent.check_if_on_CS()) and (agent._battery_level < FULL_BATTERY_LEVEL) ): # MINIMUM_BATTERY_LEVEL_TO_STOP_CHARGING --> ?????????????
                action = CHARGE_INDEX
                # agent._charging = True # --> Non è necessario wui, poichè lo satto '._charging' viene settato all'interno di 'move', il quale viene chiamato in 'step' dopo 'choose_action'.
                return action
            elif (agent._battery_level >= FULL_BATTERY_LEVEL):
                agent._battery_level = FULL_BATTERY_LEVEL
                all_actions_values[CHARGE_INDEX] = -inf
                all_actions_values[GO_TO_CS_INDEX] = -inf
            '''
            else:
                all_actions_values[GO_TO_CS_INDEX] = -inf
            '''

    rand = np.random.random() 
    if (rand > EPSILON):
        #print("SELEZIONAAAAAAAAAAAAAAAAAAAAAAAAA")
        # Select the best action so far:
        # SULLA BASE DELL ACTION_SET DELL'AGENT, ASSEGNA I VALORI DELLE ACTION AD UN SET DI AZIONI RIDOTTO --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        action = np.argmax(all_actions_values)
        #action = np.argmax(uavs_q_tables[which_uav][obs])
    else:
        #print("CASUALEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        n_actions_to_not_consider = n_actions-agent._n_actions
        #print(n_actions, agent._action_set, agent._n_actions)
        n_current_actions_to_consider = n_actions-n_actions_to_not_consider
        prob_per_action = 1/n_current_actions_to_consider
        probabilities = [prob_per_action if act_idx < n_current_actions_to_consider else 0.0 for act_idx in actions_indeces]
        #print("PROOOOOOOOOOOOOOOOOOB", probabilities)
        # Select the action randomly:
        action = np.random.choice(actions_indeces, p=probabilities)

    #print("ECOOLOOOOOOOOOOOOOOOOO", action == GO_TO_CS_INDEX)
    
    if (action == GO_TO_CS_INDEX):
        # agent._coming_home = True # --> Non è necessario wui, poichè lo satto '._charging' viene settato all'interno di 'move', il quale viene chiamato in 'step' dopo 'choose_action'.
        closest_CS = agent._cs_goal
        # Compute the path to the closest CS only if needed:
        if (closest_CS == DEFAULT_CLOSEST_CS):
            _ = agent.compute_distances(cs_cells) # --> Update the closest CS just in case the agent need to go the CS (which is obviously the closest one).
            #print("START:", get_agent_pos(agent))
            #print("END:", agent._cs_goal)
            agent._path_to_the_closest_CS = astar(cells_matrix, get_agent_pos(agent), agent._cs_goal)
            #print("ASTAR RESULT:", agent._path_to_the_closest_CS)
            #print("START POS", get_agent_pos(agent))
            #print("END POS:", agent._cs_goal)
            #print("ASTAR RESULT", agent._path_to_the_closest_CS)
            agent._current_pos_in_path_to_CS = 0 # --> when compute the path, set to 0 the counter indicating the current position (which belongs to the computed path) you have to go.
    

    '''
    if (action == go_to_cs_index):
        action_for_current_space = action
    elif (action == charge_index):
        action_for_current_space = action - 2
        #agent._charging = True
        #agent.charging_battery1()
    else:
        action_for_current_space = action
        #agents[UAV]._charging = False
    '''
    
    #action_for_q_table = action

    #return action_for_current_space, action_for_q_table

    return action

def obs_fun(obs1=None, obs2=None, obs3=None):

    if (HOW_MANY_OBSERVATIONS == 1):
        return (obs1)
    elif (HOW_MANY_OBSERVATIONS == 2):
        return (obs1, obs2)
    elif (HOW_MANY_OBSERVATIONS == 3):
        return (obs1, obs2, obs3)
    else:
        return

ANALYZED_CASE = 0

if (STATIC_REQUEST == True) and (USERS_PRIORITY == False) and (CREATE_ENODEB == False):
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
        service_request_per_epoch = env.users*ITERATIONS_PER_EPISODE
        considered_case_directory += "_inf_req"
        users_request = "Continue"
    else:
        setting_not_served_users = agent.Agent.set_not_served_users
        service_request_per_epoch = 0 # --> TO SET --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        considered_case_directory += "_lim_req"
        users_request = "Discrete"

else:

    CREATE_ENODEB = False
    DIMENSION_2D = False
    UNLIMITED_BATTERY = True
    INF_REQUEST = True
    STATIC_REQUEST = True
    USERS_PRIORITY = False
    assert False, "Environment parameters combination not implemented yet: STATIC_REQUEST: %s, DIMENSION_2D: %s, UNLIMITED_BATTERY: %s, INF_REQUEST: %s, USERS_PRIORITY: %s, CREATE_ENODEB: %s"%(STATIC_REQUEST, DIMENSION_2D, UNLIMITED_BATTERY, INF_REQUEST, USERS_PRIORITY, CREATE_ENODEB)
    pass # TO DO --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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

'''
map_width = env.static_env._area_width
map_length = env.static_env._area_height
map_height = env.static_env._area_z
'''
map_width = CELLS_COLS
map_length = CELLS_ROWS
map_height = MAXIMUM_AREA_HEIGHT

# Agents initializing
agents = env.agents

'''
agents = [[]]*N_UAVS
for UAV in range(N_UAVS):
    agents[UAV] = agent.Agent((1,0,0), 1, 0, 1, 4, N_BATTERY_LEVELS, 6, False, False, False, 2)
'''

uavs_q_tables = None

if uavs_q_tables is None:

    print("Q-TABLES INITIALIZATION . . .")
    uavs_q_tables = [None for uav in  range(N_UAVS)]
    explored_states_q_tables = [None for uav in range(N_UAVS)]
    uav_counter = 0
    
    for uav in range(N_UAVS):
        current_uav_q_table = {}
        current_uav_explored_table = {}
        
        for x_agent in range(map_width):
            x_agent += 0.5

            for y_agent in range(map_length):
                y_agent += 0.5

                # 2D case with UNLIMITED UAVs battery autonomy:
                if (ANALYZED_CASE == 1):
                    
                    #print((x_agent+0.5, y_agent+0.5))
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
                    #for battery_level in range(N_BATTERY_LEVELS+1):
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
                        z_agent += 0.5

                        if (PRIOR_KNOWLEDGE == True):
                            prior_rewards = compute_prior_rewards((x_agent, y_agent), best_prior_knowledge_points)
                            current_uav_q_table[(x_agent, y_agent, z_agent)] = [(1 - prior_rewards) for action in range(n_actions)]
                        elif (R_MAX == True):
                            current_uav_q_table[(x_agent, y_agent, z_agent)] = [max_value_for_Rmax for action in range(n_actions)]
                        else:
                            #print((x_agent+0.5, y_agent+0.5, z_agent+0.5))
                            current_uav_q_table[(x_agent, y_agent, z_agent)] = [np.random.uniform(0, 1) for action in range(n_actions)]

                        current_uav_explored_table[(x_agent, y_agent, z_agent)] = [False for action in range(n_actions)]

                # 3D case with LIMITED UAVs battery autonomy:
                elif (ANALYZED_CASE == 4):
                    for z_agent in range(MIN_UAV_HEIGHT, MAX_UAV_HEIGHT, UAV_Z_STEP):
                        z_agent += 0.5
                        #for battery_level in range(N_BATTERY_LEVELS+1):
                        for battery_level in np.arange(0, FULL_BATTERY_LEVEL+1, PERC_CONSUMPTION_PER_ITERATION):
                            
                            if (PRIOR_KNOWLEDGE == True):
                                prior_rewards = compute_prior_rewards((x_agent, y_agent), best_prior_knowledge_points)
                                current_uav_q_table[((x_agent, y_agent, z_agent), battery_level)] = [(1 - prior_rewards) for action in range(n_actions)]
                            elif (R_MAX == True):
                                current_uav_q_table[((x_agent, y_agent, z_agent), battery_level)] = [max_value_for_Rmax for action in range(n_actions)] # current_uav_q_table[((x_agent, y_agent), battery_level, timeslot)] = [np.random.uniform(0, 1) for action in range(n_actions)]
                            else:
                                #for timeslot in range(N_TIMESLOTS_PER_DAY):
                                current_uav_q_table[((x_agent, y_agent, z_agent), battery_level)] = [np.random.uniform(0, 1) for action in range(n_actions)] # current_uav_q_table[((x_agent, y_agent), battery_level, timeslot)] = [np.random.uniform(0, 1) for action in range(n_actions)]
                            
                            current_uav_explored_table[(x_agent, y_agent, z_agent), battery_level] = [False for action in range(n_actions)]
                            #print( ((x_agent, y_agent, z_agent), battery_level) )

        uavs_q_tables[uav] = current_uav_q_table
        explored_states_q_tables[uav] = current_uav_explored_table
        print("Q-Table for Uav ", uav, " created")

    print("Q-TABLES INITIALIZATION COMPLETED.")

else:

    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

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

'''
if not isdir(q_tables_directory): mkdir(q_tables_directory)
uav_directory = "UAV"
for uav in range(1, N_UAVS+1):
    if not isdir(join(q_tables_directory, uav_directory+str(uav))): mkdir(join(q_tables_directory, uav_directory+str(uav)))
'''

#print(uavs_q_tables[0])

#print("Q-TABELS:\n")
#print(uavs_q_tables[0][((4,4), 10, 2)], uavs_q_tables[1][((4,4), 10, 2)])
#print("\n")

#for uav in range(N_UAVS):
#    uavs_q_tables[uav] = np.random.uniform(low=, high=, size=[N_UC+N_CS, N_BATTERY_LEVELS, N_TIMESLOTS_PER_DAY] + [n_actions])

'''
saving_qtable_name = q_tables_directory + f"/qtable-ep1.npy"
np.save(saving_qtable_name, uavs_q_tables)
for uav in range(1, N_UAVS+1):
    current_dir = join(q_tables_directory, uav_directory+str(uav))
    print("AOOOOOOH", current_dir + f"/qtable_graph-ep1.png")
    plot.actions_min_max_per_epoch(np.load(saving_qtable_name), current_dir, n_actions, 1, uav)
'''

'''
print()
print()
print()
print()
print("CHECK")

for uav in range(N_UAVS):
    for k, v in uavs_q_tables[uav].items(): 
        for elem in v:
            if elem > 1:
                print("BECCATOOOOO")

print()
print()
print()
print()
'''

# can look up from Q-table with: print(q_table[((-9, -2), (3, 9))]) for example

uavs_episode_rewards = [[] for uav in range(N_UAVS)]
agents_paths = [[(0,0,0) for iteration in range(ITERATIONS_PER_EPISODE)] for uav in range(N_UAVS)]
print("fagentPaths:",agents_paths)
users_in_foots = [[] for uav in range(N_UAVS)]

avg_QoE1_per_epoch = [0 for ep in range(EPISODES)]
avg_QoE2_per_epoch = [0 for ep in range(EPISODES)]

q_values = [[] for episode in range(N_UAVS)]

#users_served_time = 0
#users_request_service_elapsed_time = 0

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

print("\nSTART TRAINING . . .\n")
for episode in range(1, EPISODES+1):

    epsilon_history[episode-1] = EPSILON

    print("| EPISODE: {ep:3d} | Epsilon: {eps:6f}".format(ep=episode, eps=EPSILON))

    users_served_time = 0
    users_request_service_elapsed_time = 0
    #USERS_SERVED_TIME = 0
    #USERS_REQUEST_SERVICE_ELAPSED_TIME = 0
    
    '''
    player = Blob()
    food = Blob()
    enemy = Blob()
    '''

    #for UAV in range(N_UAVS):
        #if episode % SHOW_EVERY == 0:
            #print(f"on #{episode}, epsilon is {epsilon}")
            #print(f"{SHOW_EVERY} ep mean: {np.mean(uavs_episode_rewards[UAV][-SHOW_EVERY:])}")
            #show = True
        #else:
            #show = False

    uavs_episode_reward = [0 for uav in range(N_UAVS)]
    crashes_current_episode = [False for uav in range(N_UAVS)]
    q_values_current_episode = [0 for uav in range(N_UAVS)] 
    #battery_in_CS_current_episode = [[] for uav in range(N_UAVS)]

    '''
    print()
    print()
    print()
    print()
    print("FUORI ITERAZIONI")

    prova = uavs_q_tables[0]
    for uav in range(N_UAVS):
                for k, v in uavs_q_tables[uav].items(): 
                    for elem in v:
                        if elem > 1:
                            print("BECCATOOOOOOOOOOOOOOOOOOOOOOO")
    '''
    
    prova = uavs_q_tables[0]
    # Each 30 minutes: 
    for i in range(ITERATIONS_PER_EPISODE):

        '''
        print()
        print()
        print()
        print()
        print("DENTRO ITERAZIONI")

        for uav in range(N_UAVS):
            print(uavs_q_tables == prova)
            for k, v in uavs_q_tables[uav].items(): 
                for elem in v:
                    if elem > 1:
                        print("BECCATOOOOOOOOOOOOOOOOOOOOOOO", i)
        '''
        

        all_users_in_all_foots = [] 
        '''
        obs = (player-food, player-enemy)
        #print(obs)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        '''

        #print(agents[0], agents[1])
        #print(env.get_agent_pos(agents[0]), env.get_agent_pos(agents[1]))
        for UAV in range(N_UAVS):
            #obs = (agents[UAV].pos[:2], agents[UAV]._battery_level, i)
            #obs = (env.get_agent_pos(agents[UAV])[:2], agents[UAV]._battery_level, i)
            #obs = (env.get_agent_pos(agents[UAV]), agents[UAV]._battery_level)

            # Skip analyzing the current UAV features until it starts to work (you can set a delayed start for each uav):
            current_iteration = i+1
            if (episode==1):
                # Case in which are considered the UAVs from the 2-th to the n-th:
                if (UAV>0):
                    if (current_iteration!=(DELAYED_START_PER_UAV*(UAV))):
                        continue

            aPos = get_agent_pos(agents[UAV])
            if(len(aPos)==1):
                raise Exception("ERROR")
            agents_paths[UAV][i] = aPos 
            obs = obs_fun(get_agent_pos(agents[UAV]), agents[UAV]._battery_level) #, agents[UAV]._battery_level) # --> CHANGE THE OBSERVATION WHEN SWITCH FROM 2D TO 3D ENV AND VICEVERSA !!!!!!!!!!!!!!!!!!!!!
            #print("AGENT_POS:", obs)

            if (UNLIMITED_BATTERY==False):
                set_action_set(agents[UAV])

            action = choose_action(uavs_q_tables, UAV, obs, agents[UAV], battery_in_CS_history[UAV])

            #print("AOHHHHHHHHHHHHHHHHHHHHH", battery_in_CS_current_episode[UAV])
            
            '''
            for elem in uavs_q_tables[UAV][obs]:
                print(uavs_q_tables[UAV][obs])
                if elem > 1:
                    print("AKSJFHASKJFHAKJFHAKJSFHAKJSFHAKJSFHKJASHFAKJSFHKJASFHKJASDHFKJASDFHJK")
                    print(elem)
            '''
            
            '''
            #print("AZIONEEEEE", action, uavs_q_tables[UAV][obs])
            # New users inside the current UAV agent: 
            current_users_in_footprint = agents[UAV].users_in_uav_footprint(env.users, UAV_FOOTPRINT)
            agents[UAV]._users_in_footprint = current_users_in_footprint
            #print(len(agents[UAV]._users_in_footprint))
            #print("Users in foot:", len(agents[UAV]._users_in_footprint))
            # Compute the number of users which are served by the current UAV agent:
            n_served_users = agent.Agent.n_served_users_in_foot(agents[UAV]._users_in_footprint) # --> This mainly performs a SIDE-EFFECT on the info 'served or not served' related to the users.
            # For the current iteration, add the users inside the footprint of the current UAV agent:  
            for user_per_agent_foot in current_users_in_footprint:
                all_users_in_all_foots.append(user_per_agent_foot)
            '''
            #obs_, reward, done, info = env.step(agents[UAV], action, i)
            # Add argument 'env.cells_matrix' in 'step', when choose 3D environment:
            obs_, reward, done, info = step(agents[UAV], action, all_users_in_all_foots, env.users, setting_not_served_users, crashes_current_episode, cells_matrix)
            #print("OOOOOBSSSSS", obs_)
            #print(done[1], reward)
            crashes_current_episode[UAV] = agents[UAV]._crashed
            #print(crashes_current_episode)
            #print(info)
            #print(info, "\n")
            #print("DOPOOOOOOO", obs_)
            print(" - Iteration: {it:1d} - Reward per UAV {uav:1d}: {uav_rew:6f}".format(it=i+1, uav=UAV+1, uav_rew=reward), end="\r", flush=True)

            if not explored_states_q_tables[UAV][obs_][action]:
                explored_states_q_tables[UAV][obs_][action] = True

            if (info=="IS CHARGING"):
                continue
            #print("USERS IN FOOT:", len(agents[UAV]._users_in_footprint), reward)
            #print("UAV Position:", obs_)
            #print("Users Positions:")
            #for us in env.users:
            #    print(us._x_coord, us._y_coord, us._z_coord)

            # Collect in a list ('all_users_in_all_foots') the users inside all the UAVs footprints: 
            #users_in_foots[UAV] = agents[UAV]._users_in_footprint
            #for user_per_agent_foot in users_in_foots:
                #all_users_in_all_foots += user_per_agent_foot
        
            # Set all the users which could be no more served after the current UAV action:
            setting_not_served_users(env.users, all_users_in_all_foots) # --> This make a SIDE_EFFECT on users by updating their info.
            #USERS_SERVED_TIME, USERS_REQUEST_SERVICE_ELAPSED_TIME = setting_not_served_users(env.users, all_users_in_all_foots) # --> This make also a SIDE_EFFECT on users by updating their info.

            # Take the QoE parameters only after a complete iteration performed by all the UAVs:
            #if (UAV < N_UAVS-1):
            #    USERS_SERVED_TIME, USERS_REQUEST_SERVICE_ELAPSED_TIME = 0, 0
            
            #print("SERVED_TIME", USERS_SERVED_TIME)        

            #print("ECCOLOOOOOOOOOOOOOOOOOOOOOOOO", USERS_SERVED_TIME)
            # Q-Table update:

            max_future_q = np.max(uavs_q_tables[UAV][obs_])
            '''
            if max_future_q > 1:
                print("MA COMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
                print("MAX FUTURE Q:", max_future_q, "VALUES:", uavs_q_tables[UAV][obs_])
            '''
            current_q = uavs_q_tables[UAV][obs][action]
            '''
            if current_q > 1:
                print("MA COMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
                print("CURRENT Q:", current_q, "VALUES:", uavs_q_tables[UAV][obs])
            '''
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_values_current_episode[UAV] = new_q
            #print(new_q)
            #print("NEW Q:", new_q)

            '''
            if new_q > 1:
                print("MA COMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
                print("REWARD:", reward, "CURRENT Q", current_q, "NEW Q", new_q, "MAX FUTURE Q", max_future_q)
            '''
            uavs_q_tables[UAV][obs][action] = new_q

            uavs_episode_reward[UAV] += reward

            reset_uavs(agents[UAV])

            # Agents paths for current episode:
            #print(env.get_agent_pos(agents[UAV]))
            #agents_paths[UAV][i] = env.get_agent_pos(agents[UAV]) # --> Potrebbe essere 'pesante' perchè è un'assegnazione che viene fatta ad ogni iterazione e che viene usata solo quando siamo in un episodio in cui si richiede il 'render' dello scenario --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #print("AGENT POS:", env.get_agent_pos(agents[UAV]))
            #agents_paths[UAV][i] = get_agent_pos(agents[UAV])

        #print("reward:", reward)

        current_QoE1, current_QoE2 = User.QoE(env.users)
        #print("QOE:", current_QoE1, current_QoE2)
        #print("QoE1:", current_QoE1)

        users_served_time += current_QoE1
        users_request_service_elapsed_time += current_QoE2
        #users_served_time += USERS_SERVED_TIME
        #users_request_service_elapsed_time += USERS_REQUEST_SERVICE_ELAPSED_TIME
        #if uavs_q_tables[UAV] != prova:
            #print("CHANGE QUIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII", i)

    crashes_history[episode-1] = crashes_current_episode
    #battery_in_CS_history[episode-1] = battery_in_CS_current_episode
    #print("SERVED_TIME", USERS_SERVED_TIME)
    User.avg_QoE(episode, users_served_time, env.n_users*ITERATIONS_PER_EPISODE, users_request_service_elapsed_time, avg_QoE1_per_epoch, avg_QoE2_per_epoch)
    #print("QoE1:", avg_QoE1_per_epoch)
    #avg_QoE1_per_epoch[episode-1] = (users_served_time)/(env.n_users*ITERATIONS_PER_EPISODE) # --> Start from the first epoch, without considering the obviously 0 values for the instant immediately before the first epoch.
    #avg_QoE2_per_epoch[episode-1] = users_request_service_elapsed_time/ITERATIONS_PER_EPISODE # --> Start from the first epoch, without considering the obviously 0 values for the instant immediately before the first epoch.

    #print(agents_paths)

    print(" - Iteration: {it:1d} - Reward per UAV {uav:1d}: {uav_rew:6f}".format(it=i+1, uav=UAV+1, uav_rew=reward))
    #print("____________________________________________ EPISODE:", episode , "____________________________________________")
    for UAV in range(N_UAVS):
        current_mean_reward = uavs_episode_reward[UAV]/ITERATIONS_PER_EPISODE
        uavs_episode_rewards[UAV].append(current_mean_reward)
        current_q_mean = q_values_current_episode[UAV]/ITERATIONS_PER_EPISODE
        q_values[UAV].append(current_q_mean)
        print(" - Mean reward per UAV{uav:3d}: {uav_rew:6f}".format(uav=UAV+1, uav_rew=current_mean_reward), end=" ")
        #print()
        #print("EPISODE:", episode, " | Episode Reward:", current_mean_reward, "per UAV:", UAV+1, end="\r", flush=True)
        #print("| Episode Reward:", current_mean_reward, "per UAV:", UAV+1, "| Epsilon: ", EPSILON, "\n") #np.mean(uavs_episode_rewards[UAV])
        #print("Reward:", uavs_episode_rewards[UAV])
        #print(f"{SHOW_EVERY} ep mean: {np.mean(uavs_episode_rewards[UAV][-SHOW_EVERY:])}")
    print() 

    if ((episode%1)==0): #(episode%250)==0 # --> poi cambierai la CONDIZIONE che mostra o meno lo scenario per l'episodio corrente --> !!!!!!!!!!!!!!!!!!!
        #print("AGENTS_PATHS", agents_paths)
        #plot.UAVS_crashes(EPISODES, crashes_history, saving_directory)
        print("\nSaving animation for episode:", episode)
        env.render(agents_paths, saving_directory, episode)
        print("Animation saved.\n")
        #print("Saving UAVs crashes . . .")
        #plot.UAVS_crashes(EPISODES, crashes_history, saving_directory)
        #print("UAVs crashes saved.")
        #saving_qtable_name = q_tables_directory + f"/qtable-ep{episode}.npy"
        #np.save(saving_qtable_name, uavs_q_tables)
        '''
        print("Saving Q-Tables for episode", episode, ". . .")
        if ((episode%EPISODES)==0):
            for uav in range(1, N_UAVS+1):
                saving_qtable_name = q_tables_directories[uav-1] + f"/qtable-ep{episode}.npy"
                np.save(saving_qtable_name, uavs_q_tables)
                current_dir = q_tables_directories[uav-1]
                #current_dir = join(q_tables_directory, uav_directory+str(uav))
                #print("AOOOOOOH", current_dir + f"/qtable_graph-ep{episode}.png")
                plot.actions_min_max_per_epoch(uavs_q_tables, current_dir, episode, uav)
        print("Q-Tables saved.\n")
        '''
        #plot.plt_map_views(obs_cells=env.obs_cells, cs_cells=env.cs_cells, enb_cells=env.eNB_cells, users=env.users, centroids=env.cluster_centroids, clusters_radiuses=env.clusters_radiuses, agents_paths=agents_paths, path_animation=True)
        agents_paths = [[0 for iteration in range(ITERATIONS_PER_EPISODE)] for uav in range(N_UAVS)] # --> IN realtà non serve perchè i valori vengono comunque sovrascritti ai precedenti --> !!!!!!!!!!!!!!!!!!!!!
        
        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

    '''    
    if player.x == enemy.x and player.y == enemy.y:
        reward = -ENEMY_PENALTY
    elif player.x == food.x and player.y == food.y:
        reward = FOOD_REWARD
    else:
        reward = -MOVE_PENALTY
    ## NOW WE KNOW THE REWARD, LET'S CALC YO
    # first we need to obs immediately after the move.
    new_obs = (player-food, player-enemy)
    max_future_q = np.max(q_table[new_obs])
    current_q = q_table[obs][action]

    if reward == FOOD_REWARD:
        new_q = FOOD_REWARD
    else:
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    q_table[obs][action] = new_q
    '''

    '''
    if show:
        env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
        env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue
        env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    '''

    '''
    episode_reward += reward
    if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
        break

    #print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    '''

    n_discovered_users = len(env.discovered_users)
    if ( (n_discovered_users/env.n_users) >= 0.85):

        EPSILON = EPSILON*EPSILON_DECREMENT if EPSILON > EPSILON_MIN else EPSILON_MIN

    else:

        EPSILON = EPSILON*EPSILON_DECREMENT if EPSILON > EPSILON_MIN else EPSILON_MIN2

file = open(join(saving_directory, "env_and_train_info.txt"), "a")

print("\nTRAINING COMPLETED.\n")
for uav_idx in range(N_UAVS):
    print("\nSaving battery levels when start to charge . . .")
    plot.battery_when_start_to_charge(battery_in_CS_history, uavs_directories[UAV])
    print("Battery levels when start to charge saved.")
    print("Saving UAVs crashes . . .")
    plot.UAVS_crashes(EPISODES, crashes_history, saving_directory)
    print("UAVs crashes saved.")
    # Rendering the scenario --> TO DO --> !!!!!!!!!!
    #print("\nSaving animation for episode:", episode)
    list_of_lists_of_actions = list(explored_states_q_tables[uav_idx].values())
    actions_values = [val for sublist in list_of_lists_of_actions for val in sublist]
    file.write("\nExploration percentage of the Q-Table for UAV:\n")
    actual_uav_id = uav_idx+1
    value_of_interest = np.mean(actions_values)
    file.write(str(actual_uav_id) + ": " + str(value_of_interest))
    print("Exploration percentage of the Q-Table for UAV:\n", actual_uav_id, ":", value_of_interest)
file.close()
#print("dfnsdfnsdjfnsjkdnfkjsdfnjkn")
#print(len(uavs_episode_reward))
#print(len(uavs_episode_rewards))
#print(len(uavs_episode_rewards[0]))
#print("GINOOOOOOO", avg_QoE1_per_epoch)
print("\nSaving QoE charts, UAVs rewards and Q-values . . .")
legend_labels = []
for uav in range(N_UAVS):
    plot.QoE_plot(avg_QoE1_per_epoch, EPISODES, join(saving_directory, "QoE1"), "QoE1", uav, legend_labels)
    plot.QoE_plot(avg_QoE2_per_epoch, EPISODES, join(saving_directory, "QoE2"), "QoE2", uav, legend_labels)
    plot.UAVS_reward_plot(EPISODES, uavs_episode_rewards, saving_directory)
    plot.UAVS_reward_plot(EPISODES, q_values, saving_directory, q_values=True)
print("Qoe charts, UAVs rewards and Q-values saved.")
print("\nSaving Epsilon chart trend . . .")
plot.epsilon(epsilon_history, EPISODES, saving_directory)
print("Epsilon chart trend saved.")

print("Saving Min and Max values related to the Q-Tables for episode", episode, ". . .")
for uav in range(1, N_UAVS+1):
    #current_dir = q_tables_directories[uav-1]
    #current_dir = join(uav_directory+str(uav), q_tables_directory)
    plot.actions_min_max_per_epoch(uavs_q_tables, q_tables_directories[uav-1], episode, uav)
    #saving_qtable_name = q_tables_directories[uav-1] + f"/qtable-ep{episode}.npy"
    #np.save(saving_qtable_name, uavs_q_tables)
    #current_dir = q_tables_directories[uav-1]
    #current_dir = join(q_tables_directory, uav_directory+str(uav))
    #print("AOOOOOOH", current_dir + f"/qtable_graph-ep{episode}.png")
print("Min and Max values related to the Q-Tables for episode saved.\n")

#saving_qtable_name = q_tables_directories[uav-1] + f"/qtable-ep{episode}.npy"
#np.save(saving_qtable_name, uavs_q_tables)
#current_dir = q_tables_directories[uav-1]
#current_dir = join(q_tables_directory, uav_directory+str(uav))

print("Saving Q-Tables for episode", episode, ". . .")
for uav in range(1, N_UAVS+1):
    saving_qtable_name = q_tables_directories[uav-1] + f"/qtable-ep{episode}.npy"
    np.save(saving_qtable_name, uavs_q_tables)
    #print("AOOOOOOH", current_dir + f"/qtable_graph-ep{episode}.png")
print("Q-Tables saved.\n")

'''
moving_avgs = []
for UAV in range(N_UAVS):
    moving_avgs.append(np.convolve(uavs_episode_rewards[UAV], np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid'))
'''

with open(join(saving_directory, "q_tables.pickle"), 'wb') as f:
    pickle.dump(uavs_q_tables, f)


'''
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")

plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
'''
