import numpy as np
import sys
import pickle
from os import mkdir
from os.path import join, isdir
from numpy import linalg as LA
from math import sqrt
import gym
import envs
from gym import spaces, logger
from scenario_objects import Point, Cell, User, Environment
import plotting
from my_utils import *
import agent

# ---------------------------------------------------------------------------------------------------------------------------------------

WIDTH = 400
HEIGHT = 400


#WIDTH = 1200
#HEIGHT = 600

# ---------------------------------------------------------------------------------------------------------------------------------------

REFRESH_TIME = 10
collided_item_prec = 0

SHOW_EVERY = 30
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 1.0
EPSILON_DECREMENT = 0.995
EPSILON_MIN = 0.01

R_MAX = False
max_value_for_Rmax = 100

ITERATIONS_PER_EPISODE = 30


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

env = gym.make('UAVEnv-v0')
plot = plotting.Plot()
centroids = env.cluster_centroids
# Scale centroids according to the selected resolution:
env_centroids = [(centroid[0]/CELL_RESOLUTION_PER_COL, centroid[1]/CELL_RESOLUTION_PER_ROW) for centroid in centroids]

def compute_prior_rewards(agent_pos_xy, best_prior_knowledge_points):

    actions = env.action_set
    # Initialize an random agent just to use easily the 'move' methods for 2D and 3D cases:
    agent_test = agent.Agent((agent_pos_xy[0], agent_pos_xy[1], 0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # Compute all the possible positions according to the available actions
    if (DIMENSION_2D == True):
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

    current_uav_q_table[(x_agent+0.5, y_agent+0.5)] = [(1 - normalized_ref_dist) for action in range(env.nb_actions)]

if (PRIOR_KNOWLEDGE == True):
    subareas_limits, subareas_middle_points = compute_subareas(CELLS_COLS, CELLS_ROWS, X_SPLIT, Y_SPLIT)
    #print("SUBAREASSSSSSSSSSSS", subareas_limits)
    #print("MIDDLEEEEEEEEEEEEEE", subareas_middle_points)
    best_prior_knowledge_points = []
    diagonal_area_value = sqrt(pow(CELLS_ROWS, 2) + pow(CELLS_COLS, 2))-0.5 # --> The diagonal is the maximum possibile distance between two points, and then it will be used to normalize the distance when the table is initialized under the assumption of 'prior knowledge'. 0.5 is used because the UAV is assumed to move from the middle of a cell to the middle of another one.

    for centroid in env_centroids:
        centroid_x = centroid[0]
        centroid_y = centroid[1]
        print("CENTROIDE:", (centroid_x, centroid_y))
        for subarea in range(N_SUBAREAS):
            current_subarea = subareas_limits[subarea]
            #print("SUBAREA", current_subarea)
            print("BOUNDS:", ( (current_subarea[0][0], current_subarea[0][1]), (current_subarea[1][0], current_subarea[1][1]) ) )
            if ( ( (centroid_x >= current_subarea[0][0]) and (centroid_x < current_subarea[0][1]) ) and ( (centroid_y >= current_subarea[1][0]) and (centroid_y < current_subarea[1][1]) ) ):
                print("CI SIAMOOOOOOOOOOOO")
                best_prior_knowledge_points.append(subareas_middle_points[subarea])

print("N_USERS:", env.n_users)

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

# 2D case with UNLIMITED UAVs battery autonomy:
if ( (DIMENSION_2D == True) and (UNLIMITED_BATTERY == True) ):
    ANALYZED_CASE = 1
    HOW_MANY_OBSERVATIONS = 1
    get_agent_pos = env.get_2Dagent_pos
    step = env.step_2D_unlimited_battery
    considered_case_directory = "2D_un_bat"

# 2D case with LIMITED UAVs battery autonomy:
elif ( (DIMENSION_2D == True) and (UNLIMITED_BATTERY == False) ):
    ANALYZED_CASE = 2
    HOW_MANY_OBSERVATIONS = 2
    get_agent_pos = env.get_2Dagent_pos
    step = env.step_2D_limited_battery
    considered_case_directory = "2D_lim_bat"

# 3D case with UNLIMITED UAVs battery autonomy:
elif ( (DIMENSION_2D == False) and (UNLIMITED_BATTERY == True) ):
    ANALYZED_CASE = 3
    HOW_MANY_OBSERVATIONS = 1
    get_agent_pos = env.get_3Dagent_pos
    step = env.step_3D_unlimited_battery
    considered_case_directory = "3D_un_bat"

# 3D case with LIMITED UAVs battery autonomy:
elif ( (DIMENSION_2D == False) and (UNLIMITED_BATTERY == False) ):
    ANALYZED_CASE = 4
    HOW_MANY_OBSERVATIONS = 2
    get_agent_pos = env.get_3Dagent_pos
    step = env.step_3D_limited_battery
    considered_case_directory = "3D_lim_bat"

if (INF_REQUEST == True):
    setting_not_served_users = agent.Agent.set_not_served_users_inf_request
    service_request_per_epoch = env.users*ITERATIONS_PER_EPISODE
    considered_case_directory += "_inf_req" 
else:
    setting_not_served_users = agent.Agent.set_not_served_users
    service_request_per_epoch = 0 # --> TO SET --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    considered_case_directory += "_lim_req"

considered_case_directory += "_" + str(N_UAVS) + "UAVs" + "_" + str(len(env.cluster_centroids)) + "clusters"

cases_directory = "Cases"
saving_directory = join(cases_directory, considered_case_directory)

if not isdir(saving_directory): 
    print("Creating saving dir")
    mkdir(cases_directory)
    mkdir(saving_directory)

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

    print("Q-Table initializing . . .")
    uavs_q_tables = [None for uav in  range(N_UAVS)]
    uav_counter = 0
    
    for uav in range(N_UAVS):
        current_uav_q_table = {}
        
        for x_agent in range(map_width):
            for y_agent in range(map_length):
                
                # 2D case with UNLIMITED UAVs battery autonomy:
                if (ANALYZED_CASE == 1):
                    
                    #print((x_agent+0.5, y_agent+0.5))
                    if (PRIOR_KNOWLEDGE == True):
                        prior_rewards = compute_prior_rewards((x_agent+0.5, y_agent+0.5), best_prior_knowledge_points)
                        current_uav_q_table[(x_agent+0.5, y_agent+0.5)] = [prior_rewards[action] for action in range(env.nb_actions)]
                    elif (R_MAX == True):
                        current_uav_q_table[(x_agent+0.5, y_agent+0.5)] = [max_value_for_Rmax for action in range(env.nb_actions)]
                    else:
                        current_uav_q_table[(x_agent+0.5, y_agent+0.5)] = [np.random.uniform(0, 1) for action in range(env.nb_actions)]

                # 2D case with LIMITED UAVs battery autonomy:
                elif (ANALYZED_CASE == 2):
                    for battery_level in range(N_BATTERY_LEVELS+1):

                        if (PRIOR_KNOWLEDGE == True):
                            prior_rewards = compute_prior_rewards((x_agent+0.5, y_agent+0.5), best_prior_knowledge_points)
                            current_uav_q_table[((x_agent+0.5, y_agent+0.5), battery_level)] = [(1 - prior_rewards) for action in range(env.nb_actions)]
                        elif (R_MAX == True):
                            current_uav_q_table[((x_agent+0.5, y_agent+0.5), battery_level)] = [max_value_for_Rmax for action in range(env.nb_actions)]
                        else:
                            current_uav_q_table[((x_agent+0.5, y_agent+0.5), battery_level)] = [np.random.uniform(0, 1) for action in range(env.nb_actions)]                        
                
                # 3D case with UNLIMITED UAVs battery autonomy:
                elif (ANALYZED_CASE == 3):
                    for z_agent in range(MIN_UAV_HEIGHT, MAX_UAV_HEIGHT, UAV_Z_STEP):

                        if (PRIOR_KNOWLEDGE == True):
                            prior_rewards = compute_prior_rewards((x_agent+0.5, y_agent+0.5), best_prior_knowledge_points)
                            current_uav_q_table[(x_agent+0.5, y_agent+0.5, z_agent+0.5)] = [(1 - prior_rewards) for action in range(env.nb_actions)]
                        elif (R_MAX == True):
                            current_uav_q_table[(x_agent+0.5, y_agent+0.5, z_agent+0.5)] = [max_value_for_Rmax for action in range(env.nb_actions)]
                        else:
                            print((x_agent+0.5, y_agent+0.5, z_agent+0.5))
                            current_uav_q_table[(x_agent+0.5, y_agent+0.5, z_agent+0.5)] = [np.random.uniform(0, 1) for action in range(env.nb_actions)]

                # 3D case with LIMITED UAVs battery autonomy:
                elif (ANALYZED_CASE == 4):
                    for z_agent in range(MIN_UAV_HEIGHT, MAX_UAV_HEIGHT, UAV_Z_STEP):
                        for battery_level in range(N_BATTERY_LEVELS+1):
                            
                            if (PRIOR_KNOWLEDGE == True):
                                prior_rewards = compute_prior_rewards((x_agent+0.5, y_agent+0.5), best_prior_knowledge_points)
                                current_uav_q_table[((x_agent+0.5, y_agent+0.5, z_agent+0.5), battery_level)] = [(1 - prior_rewards) for action in range(env.nb_actions)]
                            elif (R_MAX == True):
                                current_uav_q_table[((x_agent+0.5, y_agent+0.5, z_agent+0.5), battery_level)] = [max_value_for_Rmax for action in range(env.nb_actions)] # current_uav_q_table[((x_agent, y_agent), battery_level, timeslot)] = [np.random.uniform(0, 1) for action in range(env.nb_actions)]
                            else:
                                #for timeslot in range(N_TIMESLOTS_PER_DAY):
                                current_uav_q_table[((x_agent+0.5, y_agent+0.5, z_agent+0.5), battery_level)] = [np.random.uniform(0, 1) for action in range(env.nb_actions)] # current_uav_q_table[((x_agent, y_agent), battery_level, timeslot)] = [np.random.uniform(0, 1) for action in range(env.nb_actions)]

        uavs_q_tables[uav] = current_uav_q_table
        print("First Uav Completed")

    print("Q-Table initialized")

else:

    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

q_tables_dir = "QTables"
q_tables_directory = join(saving_directory, q_tables_dir)
if not isdir(q_tables_directory): mkdir(q_tables_directory)
uav_directory = "UAV"
for uav in range(1, N_UAVS+1):
    if not isdir(join(q_tables_directory, uav_directory+str(uav))): mkdir(join(q_tables_directory, uav_directory+str(uav)))

#print(uavs_q_tables[0])

#print("Q-TABELS:\n")
#print(uavs_q_tables[0][((4,4), 10, 2)], uavs_q_tables[1][((4,4), 10, 2)])
#print("\n")

#for uav in range(N_UAVS):
#    uavs_q_tables[uav] = np.random.uniform(low=, high=, size=[N_UC+N_CS, N_BATTERY_LEVELS, N_TIMESLOTS_PER_DAY] + [env.nb_actions])

'''
saving_qtable_name = q_tables_directory + f"/qtable-ep1.npy"
np.save(saving_qtable_name, uavs_q_tables)
for uav in range(1, N_UAVS+1):
    current_dir = join(q_tables_directory, uav_directory+str(uav))
    print("AOOOOOOH", current_dir + f"\qtable_graph-ep1.png")
    plot.q_tables_plot(np.load(saving_qtable_name), current_dir, env.nb_actions, 1, uav)
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
agents_paths = [[0 for iteration in range(ITERATIONS_PER_EPISODE)] for uav in range(N_UAVS)]
users_in_foots = [[] for uav in range(N_UAVS)]

avg_QoE1_per_epoch = [0 for ep in range(EPISODES)]
avg_QoE2_per_epoch = [0 for ep in range(EPISODES)]

actions_indeces = range(env.nb_actions)

#users_served_time = 0
#users_request_service_elapsed_time = 0

for episode in range(1, EPISODES+1):

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
            agents_paths[UAV][i] = get_agent_pos(agents[UAV])
            obs = obs_fun(get_agent_pos(agents[UAV]), agents[UAV]._battery_level)
            #print("AGENT_POS:", obs)
            rand = np.random.random()
            if (rand > EPSILON):
                # Select the best action so far:
                action = np.argmax(uavs_q_tables[UAV][obs])
            else:
                # Select the action randomly:
                action = np.random.choice(actions_indeces)
            
            '''
            for elem in uavs_q_tables[UAV][obs]:
                print(uavs_q_tables[UAV][obs])
                if elem > 1:
                    print("AKSJFHASKJFHAKJFHAKJSFHAKJSFHAKJSFHKJASHFAKJSFHKJASFHKJASDHFKJASDFHJK")
                    print(elem)
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
            #obs_, reward, done, info = env.step(agents[UAV], action, i)
            obs_, reward, done, info = step(agents[UAV], action)

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
            #print("NEW Q:", new_q)

            '''
            if new_q > 1:
                print("MA COMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
                print("REWARD:", reward, "CURRENT Q", current_q, "NEW Q", new_q, "MAX FUTURE Q", max_future_q)
            '''
            uavs_q_tables[UAV][obs][action] = new_q

            uavs_episode_reward[UAV] += reward

            # Agents paths for current episode:
            #print(env.get_agent_pos(agents[UAV]))
            #agents_paths[UAV][i] = env.get_agent_pos(agents[UAV]) # --> Potrebbe essere 'pesante' perchè è un'assegnazione che viene fatta ad ogni iterazione e che viene usata solo quando siamo in un episodio in cui si richiede il 'render' dello scenario --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #print("AGENT POS:", env.get_agent_pos(agents[UAV]))
            #agents_paths[UAV][i] = get_agent_pos(agents[UAV])

        #print("reward:", reward)
        EPSILON = EPSILON*EPSILON_DECREMENT if EPSILON < EPSILON_MIN else EPSILON_MIN 

        current_QoE1, current_QoE2 = User.QoE(env.users)
        #print("QoE1:", current_QoE1)

        users_served_time += current_QoE1
        users_request_service_elapsed_time += current_QoE2
        #users_served_time += USERS_SERVED_TIME
        #users_request_service_elapsed_time += USERS_REQUEST_SERVICE_ELAPSED_TIME
        #if uavs_q_tables[UAV] != prova:
            #print("CHANGE QUIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII", i)

    #print("SERVED_TIME", USERS_SERVED_TIME)
    User.avg_QoE(episode, users_served_time, env.n_users*ITERATIONS_PER_EPISODE, users_request_service_elapsed_time, avg_QoE1_per_epoch, avg_QoE2_per_epoch)
    #avg_QoE1_per_epoch[episode-1] = (users_served_time)/(env.n_users*ITERATIONS_PER_EPISODE) # --> Start from the first epoch, without considering the obviously 0 values for the instant immediately before the first epoch.
    #avg_QoE2_per_epoch[episode-1] = users_request_service_elapsed_time/ITERATIONS_PER_EPISODE # --> Start from the first epoch, without considering the obviously 0 values for the instant immediately before the first epoch.

    #print(agents_paths)

    print("EPISODE:", episode)
    for UAV in range(N_UAVS):
        print()
        uavs_episode_rewards[UAV].append(uavs_episode_reward[UAV]/ITERATIONS_PER_EPISODE)
        print("mean reward:", np.mean(uavs_episode_rewards[UAV]), "per UAV:", UAV)
        #print("Reward:", uavs_episode_rewards[UAV])
        #print(f"{SHOW_EVERY} ep mean: {np.mean(uavs_episode_rewards[UAV][-SHOW_EVERY:])}")

    if ( episode == 1): #episode % 500 == 0 # --> poi cambierai la CONDIZIONE che mostra o meno lo scenario per l'episodio corrente --> !!!!!!!!!!!!!!!!!!!
        # Rendering the scenario --> TO DO --> !!!!!!!!!!
        print("Showing Scenario for episode:", episode)
        #print("AGENTS_PATHS", agents_paths)
        env.render(agents_paths, saving_directory)
        saving_qtable_name = q_tables_directory + f"/qtable-ep{episode}.npy"
        np.save(saving_qtable_name, uavs_q_tables)
        for uav in range(1, N_UAVS+1):
            current_dir = join(q_tables_directory, uav_directory+str(uav))
            print("AOOOOOOH", current_dir + f"\qtable_graph-ep{episode}.png")
            plot.q_tables_plot(np.load(saving_qtable_name), current_dir, episode, uav)
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

#print("dfnsdfnsdjfnsjkdnfkjsdfnjkn")
print(len(uavs_episode_reward))
print(len(uavs_episode_rewards))
print(len(uavs_episode_rewards[0]))
#print("GINOOOOOOO", avg_QoE1_per_epoch)
plot.QoE_plot(avg_QoE1_per_epoch, EPISODES, join(saving_directory, "QoE1"), "QoE1")
plot.QoE_plot(avg_QoE2_per_epoch, EPISODES, join(saving_directory, "QoE2"), "QoE2")
plot.UAVS_reward_plot(EPISODES, uavs_episode_rewards, saving_directory)

moving_avgs = []
for UAV in range(N_UAVS):
    moving_avgs.append(np.convolve(uavs_episode_rewards[UAV], np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid'))

with open(join(saving_directory, "q_tables.pickle"), 'wb') as f:
    pickle.dump(uavs_q_tables, f)


'''
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
# plt.savefig(CURRENT_DIR+"/figures/env_wrapper.png")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
'''
