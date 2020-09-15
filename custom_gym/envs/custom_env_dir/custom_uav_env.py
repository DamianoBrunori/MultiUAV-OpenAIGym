import gym
import sys
import numpy as np
from decimal import Decimal
#sys.path.append('C:/Users/damia/Desktop/MAGISTRALE/TESI/Scripts/custom_gym/envs/custom_env_dir')
from my_utils import *
from load_and_save_data import *
import scenario_objects
import agent
import plotting
from gym import spaces, logger
#from scenario_objects import Point, Cell, User
import os

#DEVE STARE DENTRO UN main, se no non ti trova gli objects --> !!!!!!!!!!!!!!!!!!!!!!!!!!???????????????????????!!!!!!!!!!!!!
load = Loader()
#print("QUI:", os.listdir())
load.maps_data()
load.users_clusters()
load.maps_status()
plot = plotting.Plot()

'''
obs_points = load.obs_points
points_matrix = load._points_matrix
cs_points = load.cs_points
eNB_point = load.enb_point

cells_matrix = load.cells_matrix
obs_cells = load.obs_cells
cs_cells = load.cs_cells
eNB_cells = load.enb_cells
'''

'''
class UAVEnv(gym.Env):

    def __init__(self):
        print('Environment initialized')
    def step(self):
        print('Step successful!')
    def reset(self):
        print('Environment reset')
'''

class UAVEnv(gym.Env):

    def __init__(self):
        #self.action_space = ["nope", "up", "down"]
        #self.nb_actions = len(self.action_space)
        #self._obs_points = obs_points
        # Setting upper and lower bounds for actions values:
        upper_limits = np.array([val[0] for val in LIMIT_VALUES_FOR_ACTION])
        lower_limits = np.array([val[1] for val in LIMIT_VALUES_FOR_ACTION])

        self.action_set_min = ACTION_SPACE_3D_MIN if DIMENSION_2D==False else ACTION_SPACE_2D_MIN
        if (UNLIMITED_BATTERY==True):
            self.q_table_action_set = self.action_set_min
        else:
            if (self.action_set_min==ACTION_SPACE_2D_MIN):
                self.q_table_action_set = ACTION_SPACE_2D_TOTAL
                self.charging_set = ACTION_SPACE_2D_WHILE_CHARGING
                self.come_home_set = ACTION_SPACE_2D_COME_HOME
            else:
                self.q_table_action_set = ACTION_SPACE_3D_TOTAL
                self.charging_set = ACTION_SPACE_3D_WHILE_CHARGING
                self.come_home_set = ACTION_SPACE_3D_COME_HOME
        #self.q_table_action_set = ACTION_SPACE_3D_TOTAL if DIMENSION_2D==False else ACTION_SPACE_2D_TOTAL
        self.action_space = spaces.Discrete(len(self.q_table_action_set))
        self.nb_actions = self.action_space.n # --> ???????????????????????????????????????????????
        self.observation_space = spaces.Box(lower_limits, upper_limits, dtype=np.float32)
        # self.agents = agent.Agent.initialize_agents(UAVS_POS) # Agents initialization
        self.state = None
        #self.static_env = scenario_objects.Environment(AREA_WIDTH, AREA_HEIGHT, MAXIMUM_AREA_HEIGHT, CELL_RESOLUTION_PER_ROW, CELL_RESOLUTION_PER_COL) # --> Al posto di questo magari carica gli ostacoli, la cell_matrix, . . . --> !!!!!!!!!!!!!!!
        self.obs_points = load.obs_points
        self.points_matrix = load._points_matrix
        self.cs_points = load.cs_points
        #for CS in self.cs_points:
        #    print("CS POINTS", (CS._x_coord, CS._y_coord, CS._z_coord))
        #uavs_pos = agent.Agent.setting_agents_pos(self.cs_points)
        #self.agents = agent.Agent.initialize_agents(uavs_pos) # Agents initialization
        #for ag in self.agents:
        #    print("UAV_POSooooooooooooooooooo", (ag._x_coord, ag._y_coord, ag._z_coord))
        self.eNB_point = load.enb_point
        self.cells_matrix = load.cells_matrix
        self.obs_cells = load.obs_cells
        self.max_uav_height = max([obs._z_coord for obs in self.obs_cells]) if DIMENSION_2D==False else 0
        #print("MAX UAV HEIGHT:", self.max_uav_height)
        self.cs_cells = load.cs_cells
        # Set the CS position according to the desired resolution cells:
        self.initial_uavs_pos = agent.Agent.setting_agents_pos(self.cs_cells) if UNLIMITED_BATTERY==False else UAVS_POS
        self.agents = agent.Agent.initialize_agents(self.initial_uavs_pos, self.max_uav_height, self.action_set_min) # Agents initialization
        #for CS in self.cs_points:
        #    print("CS CELLS", (CS._x_coord, CS._y_coord, CS._z_coord))
        #for ag in self.agents:
        #    print("UAV_POSooooooooooooooooooo", (ag._x_coord, ag._y_coord, ag._z_coord))
        self.eNB_cells = load.enb_cells
        self.points_status_matrix = load.points_status_matrix
        self.cells_status_matrix = load.cells_status_matrix
        self.clusterer = load.initial_clusterer
        self.cluster_centroids = load.initial_centroids
        self.users_clusters = load.initial_usr_clusters
        self.clusters_radiuses = load.initial_clusters_radiuses
        initial_users = load.initial_users
        for user in initial_users:
            # Set the users coordinates according to the desired resolution cell:
            user._x_coord /= CELL_RESOLUTION_PER_COL 
            user._y_coord /= CELL_RESOLUTION_PER_ROW
        self.users = initial_users
        self.users_walk_steps = []
        self.k_steps_to_walk = 0
        self.uav_footprint = ACTUAL_UAV_FOOTPRINT 
        self.n_users = len(self.users)
        self.discovered_users = []
        self.current_requested_bandwidth = 0 # --> bandwidth requested from all the users belonging to the current (considered) UAV footprint.
        #self.n_features = 2

    def step_2D_limited_battery(self, agent, action, all_users_inside_foots, users, setting_not_served_users, crashes_current_episode, cells_matrix=None):
        # - 2D case;
        # - LIMITED battery;
        # - Constat battery consumption wich includes both services and motions of the UAVs;
        # - All the users have the same priority (each of the is served) and and ask for the same service;
        # - Infinite UAV bandwidth;

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        agent_pos = self.get_2Dagent_pos(agent)
        info = ""
        #current_action = self.action_set_min[action]
        #current_action = agent._action_set[action]
        current_action = self.q_table_action_set[action]

        agent_pos_ = agent.move_2D_limited_battery(agent_pos, current_action) # --> 'move_2D_limited_battery' method includes also 'off_map_move_2D' check.

        if ( (action==CHARGE_2D_INDEX) or (action==CHARGE_2D_INDEX) ):
            agent._users_in_footprint = []
            current_users_in_footprint = []
        else:
            current_users_in_footprint = agent.users_in_uav_footprint(users, self.uav_footprint, self.discovered_users)
            agent._users_in_footprint = current_users_in_footprint
        # Compute the number of users which are served by the current UAV agent:
        n_served_users = agent.n_served_users_in_foot(agent._users_in_footprint) # --> This mainly performs a SIDE-EFFECT on the info 'served or not served' related to the users.
        # Set all the users which could be no more served after the current UAV action:
        #setting_not_served_users(users, all_users_inside_foots) # --> This make a SIDE_EFFECT on users by updating their info.
        # For the current iteration, add the users inside the footprint of the current UAV agent:  
        for user_per_agent_foot in current_users_in_footprint:
            all_users_inside_foots.append(user_per_agent_foot) # --> SIDE-EFFECT on 'all_users_inside_foots'

        agent._x_coord = agent_pos_[0]
        agent._y_coord = agent_pos_[1]

        #reward = self.reward_function_1(agent._users_in_footprint)
        reward = self.reward_function_2(agent._users_in_footprint, agent._battery_level, agent._required_battery_to_CS)

        done, info = self.is_terminal_state(agent, crashes_current_episode)

        s_ = (agent_pos_, agent._battery_level)

        #print(agent._battery_level)

        if (done):

            if (info=="IS CRASHED"):
                reward = 0.0
            else:
                reward = 0.0 # --> UAV is charging
            #reward = 1.0 # --> da decidere --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #done = True
            #info = "TERMINAL"

        return s_, reward, done, info

    def step_2D_unlimited_battery(self, agent, action, all_users_inside_foots, users, setting_not_served_users, crashes_current_episode=None, cells_matrix=None):
        # - 2D case;
        # - UNLIMITED battery;
        # - Constat battery consumption wich includes both services and motions of the UAVs;
        # - All the users have the same priority (each of the is served) and and ask for the same service; 
        # - Infinite UAV bandwidth;

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        agent_pos = self.get_2Dagent_pos(agent)
        info = ""
        #current_action = self.action_set_min[action]
        current_action = self.q_table_action_set[action]
        #print("ACTION:", current_action)

        agent_pos_ = agent.move_2D_unlimited_battery(agent_pos, current_action)

        current_users_in_footprint = agent.users_in_uav_footprint(users, self.uav_footprint, self.discovered_users)
        #print("QUANTIIIIII", len(current_users_in_footprint))
        agent._users_in_footprint = current_users_in_footprint
        # Compute the number of users which are served by the current UAV agent:
        n_served_users = agent.n_served_users_in_foot(agent._users_in_footprint) # --> This mainly performs a SIDE-EFFECT on the info 'served or not served' related to the users.
        # Set all the users which could be no more served after the current UAV action:
        #setting_not_served_users(users, all_users_inside_foots) # --> This make a SIDE_EFFECT on users by updating their info.
        # For the current iteration, add the users inside the footprint of the current UAV agent:  
        for user_per_agent_foot in current_users_in_footprint:
            all_users_inside_foots.append(user_per_agent_foot) # --> SIDE-EFFECT on 'all_users_inside_foots'

        agent._x_coord = agent_pos_[0]
        agent._y_coord = agent_pos_[1]

        reward = self.reward_function_1(agent._users_in_footprint)

        s_ = (agent_pos_)

        done, info = self.is_terminal_state(agent, crashes_current_episode)

        if (done):
            
            if (info=="IS CRASHED"):
                reward = 0.0
            else:
                reward = 0.0 # --> UAV is charging
            #reward = 1.0 # --> da decidere --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            ##done = True
            #info = "TERMINAL" 

        return s_, reward, done, info

    def step_3D_limited_battery(self, agent, action, all_users_inside_foots, users, setting_not_served_users, crashes_current_episode, cells_matrix):
        # - 3D case;
        # - LIMITED battery;
        # - Constat battery consumption wich includes both services and motions of the UAVs;
        # - All the users have the same priority (each of the is served) and and ask for the same service; 
        # - Infinite UAV bandwidth;

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        agent_pos = self.get_3Dagent_pos(agent)
        info = "" # --> lo utilizzo per capire chi effettivamente ha vinto, così al termine dell' episodio posto stampare il massimo punteggio raggiungibile (10) per il vincitore (altrimenti avrei stampato sempre 9 per il vincitore)
        #current_action = self.action_set_min[action]
        current_action = self.q_table_action_set[action]

        agent_pos_ = agent.move_3D_limited_battery(agent_pos, current_action, cells_matrix) # --> 'move_3D_unlimited_battery' method includes also 'off_map_move_3D' check.

        #print("BATTERY:", agent._battery_level)

        if ( (action==GO_TO_CS_3D_INDEX) or (action==CHARGE_3D_INDEX) ):
            agent._users_in_footprint = []
            current_users_in_footprint = []
        else:
            current_users_in_footprint = agent.users_in_uav_footprint(users, self.uav_footprint, self.discovered_users)
            agent._users_in_footprint = current_users_in_footprint
        # Compute the number of users which are served by the current UAV agent:
        n_served_users = agent.n_served_users_in_foot(agent._users_in_footprint) # --> This mainly performs a SIDE-EFFECT on the info 'served or not served' related to the users.
        # Set all the users which could be no more served after the current UAV action:
        #setting_not_served_users(users, all_users_inside_foots) # --> This make a SIDE_EFFECT on users by updating their info.
        # For the current iteration, add the users inside the footprint of the current UAV agent:  
        for user_per_agent_foot in current_users_in_footprint:
            all_users_inside_foots.append(user_per_agent_foot) # --> SIDE-EFFECT on 'all_users_inside_foots'

        agent._x_coord = agent_pos_[0]
        agent._y_coord = agent_pos_[1]
        agent._z_coord = agent_pos_[2]
        
        #reward = self.reward_function_1(agent._users_in_footprint)
        reward = self.reward_function_2(agent._users_in_footprint, agent._battery_level, agent._required_battery_to_CS)

        done, info = self.is_terminal_state(agent, crashes_current_episode)

        s_ = (agent_pos_, agent._battery_level)

        #print("BATTERY LEVEL:", agent._battery_level)

        if (done):
            
            if (info=="IS CRASHED"):
                reward = 0.0
            else:
                reward = 0.0 # --> UAV is charging

            #reward = 0.0 # --> da decidere --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #done = True
            #info = "IS CRASHED" 

        return s_, reward, done, info

    def step_3D_unlimited_battery(self, agent, action, all_users_inside_foots, users, setting_not_served_users, crashes_current_episode, cells_matrix):
        # - 3D case;
        # - UNLIMITED battery;
        # - Constat battery consumption wich includes both services and motions of the UAVs;
        # - All the users have the same priority (each of the is served) and and ask for the same service; 
        # - Infinite UAV bandwidth;

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        agent_pos = self.get_3Dagent_pos(agent)
        info = "" # --> lo utilizzo per capire chi effettivamente ha vinto, così al termine dell' episodio posto stampare il massimo punteggio raggiungibile (10) per il vincitore (altrimenti avrei stampato sempre 9 per il vincitore)
        #current_action = self.action_set_min[action]
        current_action = self.q_table_action_set[action]

        agent_pos_ = agent.move_3D_unlimited_battery(agent_pos, current_action, cells_matrix) # --> 'move_3D_unlimited_battery' method includes also 'off_map_move_3D' check.

        current_users_in_footprint = agent.users_in_uav_footprint(users, self.uav_footprint, self.discovered_users)
        agent._users_in_footprint = current_users_in_footprint
        # Compute the number of users which are served by the current UAV agent:
        n_served_users = agent.n_served_users_in_foot(agent._users_in_footprint) # --> This mainly performs a SIDE-EFFECT on the info 'served or not served' related to the users.
        # Set all the users which could be no more served after the current UAV action:
        #setting_not_served_users(users, all_users_inside_foots) # --> This make a SIDE_EFFECT on users by updating their info.
        # For the current iteration, add the users inside the footprint of the current UAV agent:  
        for user_per_agent_foot in current_users_in_footprint:
            all_users_inside_foots.append(user_per_agent_foot) # --> SIDE-EFFECT on 'all_users_inside_foots'

        #print("PRIMAAAAAAAAA", (agent._x_coord, agent._y_coord, agent._z_coord))
        #agent._x_coord = round(Decimal(agent_pos_[0]), 1)
        #agent._y_coord = round(Decimal(agent_pos_[1]), 1)
        #agent._z_coord = round(Decimal(agent_pos_[2]), 1)
        
        reward = self.reward_function_1(agent._users_in_footprint)

        s_ = (agent_pos_)
        #print("DOPOOOOOOOOOO", s_)

        done, info = self.is_terminal_state(agent, crashes_current_episode)

        if (done):
            
            if (info=="IS CRASHED"):
                reward = 0.0
            else:
                reward = 0.0 # --> UAV is charging

            #reward = 1.0 # --> da decidere --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            ##done = True
            #info = "TERMINAL" 

        return s_, reward, done, info

    def step_3D_limited_battery_multi_service_limited_bandwidth(self, agent, action, all_users_inside_foots, users, setting_not_served_users, crashes_current_episode, cells_matrix, n_tr_active, n_ec_active, n_dg_active):
            # - 3D case;
            # - LIMITED battery;
            # - Constat battery consumption wich includes both services and motions of the UAVs;
            # - All the users have the same priority (each of the is served) and and ask for the same service; 
            # - Infinite UAV bandwidth;

            assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
            
            if (DIMENSION_2D==False):
                agent_pos = self.get_3Dagent_pos(agent)
            else:
                agent_pos = self.get_2Dagent_pos(agent)
            info = "" # --> lo utilizzo per capire chi effettivamente ha vinto, così al termine dell' episodio posto stampare il massimo punteggio raggiungibile (10) per il vincitore (altrimenti avrei stampato sempre 9 per il vincitore)
            #current_action = self.action_set_min[action]
            current_action = self.q_table_action_set[action]

            if (DIMENSION_2D==False):
                agent_pos_ = agent.move_3D_limited_battery(agent_pos, current_action, cells_matrix) # --> 'move_3D_unlimited_battery' method includes also 'off_map_move_3D' check.
            else:
                agent_pos_ = agent.move_2D_limited_battery(agent_pos, current_action)
            
            #print("BATTERY:", agent._battery_level)

            if ( (action==GO_TO_CS_3D_INDEX) or (action==CHARGE_3D_INDEX) ):
                agent._users_in_footprint = []
                current_users_in_footprint = []
            else:
                self.current_requested_bandwidth = 0
                current_users_in_footprint, bandwidth_request_in_current_footprint = agent.users_in_uav_footprint_lim_band(users, self.uav_footprint, self.discovered_users)
                self.current_requested_bandwidth = bandwidth_request_in_current_footprint
                bandwidth_request_in_current_footprint
                agent._users_in_footprint = current_users_in_footprint
            # Compute the number of users which are served by the current UAV agent:
            n_served_users = agent.n_served_users_in_foot(agent._users_in_footprint) # --> This mainly performs a SIDE-EFFECT on the info 'served or not served' related to the users.
            # Set all the users which could be no more served after the current UAV action:
            #setting_not_served_users(users, all_users_inside_foots) # --> This make a SIDE_EFFECT on users by updating their info.
            # For the current iteration, add the users inside the footprint of the current UAV agent:  
            for user_per_agent_foot in current_users_in_footprint:
                all_users_inside_foots.append(user_per_agent_foot) # --> SIDE-EFFECT on 'all_users_inside_foots'

            agent._x_coord = agent_pos_[0]
            agent._y_coord = agent_pos_[1]
            if DIMENSION_2D==False: agent._z_coord = agent_pos_[2]
            
            #reward = self.reward_function_1(agent._users_in_footprint)
            reward = self.reward_function_3(agent._users_in_footprint, agent._battery_level, agent._required_battery_to_CS, n_tr_active, n_ec_active, n_dg_active)
            #print("\nDOPOOOOOOOOO", self.current_requested_bandwidth, "\n")

            done, info = self.is_terminal_state(agent, crashes_current_episode)

            s_ = (agent_pos_, agent._battery_level)

            #print("BATTERY LEVEL:", agent._battery_level)

            if (done):
                
                if (info=="IS CRASHED"):
                    reward = 0.0
                else:
                    reward = 0.0 # --> UAV is charging

                #reward = 0.0 # --> da decidere --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #done = True
                #info = "IS CRASHED" 

            return s_, reward, done, info

    def step(self, agent, action, timeslot, current_iteration):

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        #s = self.state
        agent_pos = self.get_3Dagent_pos(agent)
        #agent_pos = agent.pos
        #cc_reference = agent.centroid_cluster_reference(centroids) # selected centroid cluster as reference for the current agent
        #d_ag_cc = LA.norm(np.array(agent_pos) - np.array(cc_reference)) # distance betwen the agent position and the centroid cluster selected as reference
        #current_battery_level = agent.battery_level
        
        # Current State:
        #s = (agent._d_ag_cc, agent._battery_level, timeslot)
        #s = (d_ag_cc, current_battery_level, timeslot)
        #s = (self.player_score, self.computer_score)
        #time.sleep(0.5)
        info = "" # --> lo utilizzo per capire chi effettivamente ha vinto, così al termine dell' episodio posto stampare il massimo punteggio raggiungibile (10) per il vincitore (altrimenti avrei stampato sempre 9 per il vincitore)
        current_action = self.action_set_min[action]

        #self.move(current_action)
        # Next agent position:
        agent_pos_ = agent.move(agent_pos, current_action) # --> 'move' method includes also 'off_map_move' check.

        '''
        if (self.off_map_move(new_agent_pos)):
            agent_pos_ = agent_pos
            # Reduction battery level due to the agent motion: 
            self.residual_battery_after_propulsion(HOVERING)
        else:
            self.residual_battery_after_propulsion(move_action)
        '''

        agent._x_coord = agent_pos_[0]
        agent._y_coord = agent_pos_[1]
        agent._z_coord = agent_pos_[2]

        # DEVI DEFINIRE 'users' --> !!!!!!!!!!!!!!!!!!!!
        users_in_footprint = agent.users_in_uav_footprint(self.users, self.uav_footprint, self.discovered_users) # --> IMPORTA GLI USERS --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        agent._users_in_footprint = users_in_footprint
        n_served_users = agent.n_served_users_in_foot_and_type_of_service(users_in_footprint)
        agent.set_not_served_users(self.users, current_iteration)
        
        # # Reduction battery level due to service provided by the agent:
        agent.residual_battery_after_service()

        # DEVI DEFINIRE 'centroids' --> !!!!!!!!!!!!!!!!!!!!!!!!
        # QUESTE TRE RIGHE SERVONO PER CALCOLARE LA DISTANZA DAL CENTROIDE "selezionato con un'euristica":
        #cc_reference_ = agent.Agent.centroid_cluster_reference(cluster_centroids) # Selected centroid cluster as reference for the current agent --> IMPORTA I CENTROIDS --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #d_ag_cc_ = LA.norm(np.array(agent_pos_) - np.array(cc_reference_)) # Distance betwen the agent position and the centroid cluster selected as reference
        #self._d_ag_cc = d_ag_cc_

        #reward = self.reward_function(agent, users_in_footprint) # --> vecchio reward --< !!!!!!!!!!!!!!!!!!!!!!!
        reward = self.reward_function(agent)

        s_ = (agent_pos_, agent._battery_level) # , timeslot+1)
        
        '''
        if (self.off_map_move(agent_pos_)):
            agent_pos_ = agent_pos
            # Reduction battery level due to the agent motion: 
            agent.residual_battery_after_propulsion(HOVERING)
        else:
            agent.residual_battery_after_propulsion(current_action)
        '''
        
        '''
        agent._x_coord = agent_pos_[0]
        agent._y_coord = agent_pos_[1]
        agent._z_coord = agent_pos_[2]
        '''

        '''
        users_in_footprint = agent.users_in_uav_footprint(users)
        n_served_users = agent.n_served_users_in_foot_and_type_of_service(users_in_footprint)
        # # Reduction battery level due to service provided by the agent:
        agent.residual_battery_after_service()
        '''
        
        #cc_reference_ = agent.centroid_cluster_reference(centroids)
        #d_ag_cc_ = LA.norm(np.array(agent_pos_) - np.array(cc_reference_))

        # Next state: # --> VEDI COME AGGIORNARE IL TIMESLOT --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (timeslot+1 ???)
        #agent_pos_xy = self.get_agent_pos(agent)[:2]
        # agent_pos_xy = agent.pos[:2]
        # agent._d_ag_cc --> Poi proverai ad usare questo come observation --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #s_ = (agent_pos_xy, agent._battery_level, timeslot+1) # next_state --> INSERISCI I VALORI PER LO state corrente (saranno cambiati dato che hai eseguito 'move(..)') -> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        #next_ = [self.player_score, self.computer_score]  # next state

        # DECIDI LA/LE CONDIZIONI CHE STABILISCONO CHE HO RAGGIUNTO IL GOAL O L'ENVIRONMENT VA RESETTATO --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
        done = self.is_terminal_state(agent)

        if (done):

            if (info=="IS CRASHED"):
                reward = 0.0
            else:
                reward = 0.0 # --> UAV is charging

            #reward = 0.0 # --> da decidere --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            ##done = True
            #info = "IS CRASHED" 

        return s_, reward, done, info

    def cost_reward(self, battery_level, needed_battery):

        alpha_s = 0
        alpha_c = 0

        if (battery_level > CRITICAL_BATTERY_LEVEL):
            alpha_s = 1
            alpha_c = 0
        elif ( (battery_level >= CRITICAL_BATTERY_LEVEL_2) and (battery_level > needed_battery) ):
            alpha_s = 0.8
            alpha_c = 0.2
        elif ( (battery_level >= CRITICAL_BATTERY_LEVEL_3) and (battery_level > needed_battery) ):
            alpha_s = 0.5
            alpha_c = 0.5
        elif ( (battery_level >= CRITICAL_BATTERY_LEVEL_4) and (battery_level > needed_battery) ):
            alpha_s = 0.2
            alpha_c = 0.8
        elif (battery_level <= needed_battery):
            alpha_s = 0
            alpha_c = 1
            
        reward_for_cost = needed_battery/battery_level if battery_level != 0 else 1

        return reward_for_cost, alpha_s, alpha_c

    def discount_for_user_wait(self):

        n_discovered_users = len(self.discovered_users)
        all_wait_times = sum([wait_time_for_cur_user._info[3] for wait_time_for_cur_user in self.discovered_users])
        avg_wait_time_for_disc_users = all_wait_times/n_discovered_users if n_discovered_users!=0 else 0.0

        discount_factor = 0.0
        if (avg_wait_time_for_disc_users>CRITICAL_WAITING_TIME_FOR_SERVICE):
            discount_factor = CRITICAL_WAITING_TIME_FOR_SERVICE/NORMALIZATION_FACTOR_WAITING_TIME_FOR_SERVIE if avg_wait_time_for_disc_users<=NORMALIZATION_FACTOR_WAITING_TIME_FOR_SERVIE else 1.0
        else:
            discount_factor = 0.0

        return discount_factor


    # Reward function which takes into account only the percentage of covered users:
    def reward_function_1(self, users_in_footprint):

        n_users_in_footprint = len(users_in_footprint)
        reward = n_users_in_footprint/(self.n_users/N_UAVS)
        # Case in which a UAV is covering a number of user greater than the one (hypothetically) assigned to each UAV: 
        if (reward>1):
            reward = 1.0
        #print("\n")
        #print(n_users_in_footprint, self.n_users/N_UAVS, reward)
        #print("\n")
        #print("N Users:", n_users_in_footprint, "Reward:", reward)
        discount_for_wait_time = self.discount_for_user_wait()
        reward -= discount_for_wait_time
        if (reward<0.0):
            reward = 0.0

        #print("AOOOOOOOOOOOOOOOOHHHH", reward)

        return reward

    def reward_function_2(self, users_in_footprint, battery_level, needed_battery):

        # There is no need to take into account the case in which the agent is going to CS, because when it is 'coming home'
        # the method 'Agent.users_in_uav_footprint' returns always 0 (the agent is only focused on going to CS withouth serving anyone else).

        reward_for_users = self.reward_function_1(users_in_footprint)

        alpha_s = 1
        alpha_c = 0

        if (needed_battery==None):
            
            reward = alpha_s*reward_for_users

        else:

            reward_for_cost, alpha_s, alpha_c = self.cost_reward(battery_level, needed_battery)
            reward = alpha_s*reward_for_users + alpha_c*reward_for_cost

        return reward

        # TO DO . . . --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def reward_function_3(self, users_in_footprint, battery_level, needed_battery, n_tr_active_users, n_ec_active_users, n_dg_active_users):

        reward_for_users = self.reward_function_1(users_in_footprint)

        served_users_asking_for_service = 0
        n_served_tr_users = 0
        n_served_ec_users = 0
        n_served_dg_users = 0
        #self.current_requested_bandwidth = 0
        for user in users_in_footprint:
            
            if (user._info[1]!=NO_SERVICE):
                
                if (user._info[1]==THROUGHPUT_REQUEST):
                    n_served_tr_users += 1
                elif (user._info[1]==EDGE_COMPUTING):
                    n_served_ec_users += 1
                elif (user._info[1]==DATA_GATHERING):
                    n_served_dg_users += 1
                
                served_users_asking_for_service += 1
                #self.current_requested_bandwidth += user._info[5]
                #print("\nPRIMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", self.current_requested_bandwidth, "\n")
                #print("\nRICHIESTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", self.current_requested_bandwidth, "\n")

        n_tr_served_perc = n_served_tr_users/n_tr_active_users if n_tr_active_users!=0 else 0
        n_ec_served_perc = n_served_ec_users/n_ec_active_users if n_ec_active_users!=0 else 0
        n_dg_served_perc = n_served_dg_users/n_dg_active_users if n_dg_active_users!=0 else 0

        alpha_u = 0.5
        alpha_tr = 0.2
        alpha_ec = 0.2
        alpha_dg = 0.1

        reward_for_services_and_users = alpha_u*reward_for_users + alpha_tr*n_tr_served_perc + alpha_ec*n_ec_served_perc + alpha_dg*n_dg_served_perc
        reward = reward_for_services_and_users

        if (needed_battery!=None):

            reward_for_cost, alpha_s, alpha_c = self.cost_reward(battery_level, needed_battery)
            reward = alpha_s*reward_for_services_and_users + alpha_c*reward_for_cost

        #print(reward_for_users, n_tr_served_perc, n_ec_served_perc, n_dg_served_perc)

        discount_for_wait_time = self.discount_for_user_wait()
        reward -= discount_for_wait_time
        if (reward<0.0):
            reward = 0.0
        
        return reward

    # _____________________________________________________________________________________________________________________________________________
    # Reward function provvisoria con grandezze non scalate/normalizzate: 
    def reward_function(self, agent):

        reward = scenario_objects.User.bitrate_request() + scenario_objects.User.edge_computing_request() + scenario_objects.User.data_gathering()

        return reward
    # _____________________________________________________________________________________________________________________________________________

    def get_active_users(self):

        n_active_users = 0
        n_inactive_users = 0 
        tr_users = 0
        ec_users = 0
        dg_users = 0

        n_tr_served = 0
        n_ec_served = 0
        n_dg_served = 0        

        for user in self.users:
            if (user._info[1]==NO_SERVICE):
                n_inactive_users += 1
            elif (user._info[1]==THROUGHPUT_REQUEST):
                tr_users += 1
                if (user._info[0]==True):
                    n_tr_served += 1
            elif (user._info[1]==EDGE_COMPUTING):
                ec_users += 1
                if (user._info[0]==True):
                    n_ec_served += 1
            elif (user._info[1]==DATA_GATHERING):
                dg_users += 1
                if (user._info[0]==True):
                    n_dg_served += 1

        n_active_users = tr_users + ec_users + dg_users

        return n_active_users, tr_users, ec_users, dg_users, n_tr_served, n_ec_served, n_dg_served

    def set_action_set2D(self, agent):

        if ( (agent._battery_level <= CRITICAL_BATTERY_LEVEL) and (agent._charging == False) ):
            agent._action_set = ACTION_SPACE_2D_COME_HOME
        elif (agent._charging == True):
            agent._action_set = ACTION_SPACE_2D_WHILE_CHARGING
        else:
            agent._action_set = ACTION_SPACE_2D_MIN
            agent._path_to_the_closest_CS = []
            agent._current_pos_in_path_to_CS = -1
            agent._required_battery_to_CS = None

    def set_action_set3D(self, agent):

        if ( (agent._battery_level <= CRITICAL_BATTERY_LEVEL) and (agent._charging == False) ):
            #print("COME HOME SPACE")
            agent._action_set = ACTION_SPACE_3D_COME_HOME
        elif (agent._charging == True):
            #print("WHILE CHARGING SPACE")
            agent._action_set = ACTION_SPACE_3D_WHILE_CHARGING
        elif ( (agent._coming_home == False) and (agent._charging == False) and (agent._battery_level > CRITICAL_BATTERY_LEVEL) ):
            #print("MIN SPACE")
            agent._action_set = ACTION_SPACE_3D_MIN
            agent._path_to_the_closest_CS = []
            agent._current_pos_in_path_to_CS = -1
            agent._required_battery_to_CS = None

    def noisy_measure_or_not(self, values_to_warp):
        warped_values = []
        coord_idx = 1

        for value in values_to_warp:
            noise_prob = np.random.rand()
            if noise_prob < 0.1:
                #print("----------------------")
                #print("NOISE ADDED", noise_prob)
                #print("----------------------")
                gaussian_noise = np.random.normal(loc=0, scale=1)
                warped_value = round(value + gaussian_noise)
                # X coordinate case:
                if (coord_idx==1):
                    if (warped_value>=AREA_WIDTH):
                        warped_value = AREA_WIDTH - 0.5
                    elif (warped_value<=0):
                        warped_value = 0.5
                    else:
                        warped_value += 0.5
                # Y coordinates case:
                elif (coord_idx==2):
                    if (warped_value>=AREA_HEIGHT):
                        warped_value = AREA_HEIGHT - 0.5
                    elif (warped_value<=0):
                        warped_value = 0.5
                    else:
                        warped_value += 0.5
                # Z coordinate case (no error on this measurement):
                elif (coord_idx==3):
                    warped_value = value
                # Z coordinate has a larger step, thus a gaussian noise could lead to a 'KeyError'
                '''
                # Z coordinate case:
                elif (coord_idx==3):
                    if (warped_value>MAXIMUM_AREA_HEIGHT):
                        warped_value = MAXIMUM_AREA_HEIGHT - 0.5
                    elif (warped_value<MIN_UAV_HEIGHT):
                        warped_value = MIN_UAV_HEIGHT + 0.5
                    else:
                        print("ERRORE QUI")
                        warped_value += (UAV_Z_STEP + 0.5)
                        if (warped_value>=MAXIMUM_AREA_HEIGHT):
                            warped_value = MAXIMUM_AREA_HEIGHT - 0.5
                '''
            else:
                warped_value = value

            coord_idx += 1

            warped_values.append(warped_value)
        
        return tuple(warped_values)

    def show_scores(self):
        
        # TO DO . . . --> FORSE --> ?????????????????????????????????????????????????

        pass

    def is_terminal_state(self, agent, crashes_current_episode):

        # TO DO . . . --> Decidi quando uno stato è terminale, ossia se è stato raggiunto un eventuale goal oppure se l'environment deve essere resettato --> !!!!!!!!!!!!!!!!!!!!!!!

        if ( (agent._battery_level <= 0) and (not agent.check_if_on_CS()) ):
            #print("CI SONOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
            agent._battery_level = 0
            crashes_current_episode = True
            agent._crashed = True
            #agent._action_set = ACTION_SPACE_2D_MIN
            #agent._path_to_the_closest_CS = []
            #agent._current_pos_in_path_to_CS = -1
            #agent._required_battery_to_CS = None

            return True, "IS CRASHED"

        elif (agent._charging==True):

            agent._crashed = False

            return True, "IS CHARGING"

        else:

            agent._crashed = False
            
            return False, "IS WORKING"
        
        '''
        if (state[2] == 3600):
            return True
        else:
            return False
        ''' 
    def get_2Dagent_pos(self, agent):

        x = agent._x_coord
        y = agent._y_coord

        return (x, y)

    def get_3Dagent_pos(self, agent):

        x = agent._x_coord
        y = agent._y_coord
        z = agent._z_coord

        #print(x, y, z)
        return (x, y, z)

    '''
    def off_map_move(self, new_agent_pos):
        # agent_pos is a tuple (x,y,z)

        agent_x = new_agent_pos[0]
        agent_y = new_agent_pos[1]
        agent_z = new_agent_pos[2]

        if \
        ( (agent_x < LOWER_BOUNDS) or \
        (agent_y < LOWER_BOUNDS) or \
        (agent_z < MINIMUM_AREA_HEIGHT) or \
        (agent_x >= CELLS_COLS) or \
        (agent_y >= CELLS_ROWS) or \
        (agent_z >= MAXIMUM_AREA_HEIGHT) ):

            return True

        else:

            return False
    '''

    def render(self, agents_paths, where_to_save, episode):

        plot.plt_map_views(obs_cells=self.obs_cells, cs_cells=self.cs_cells, enb_cells=self.eNB_cells, points_status_matrix=self.points_status_matrix, cells_status_matrix=self.cells_status_matrix, users=self.users, centroids=self.cluster_centroids, clusters_radiuses=self.clusters_radiuses, area_height=AREA_HEIGHT, area_width=AREA_WIDTH, N_cells_row=CELLS_ROWS, N_cells_col=CELLS_COLS, agents_paths=agents_paths, path_animation=True, where_to_save=where_to_save, episode=episode)

    def reset(self, agents):

        # Agents (re)initialization:
        '''
        for agent in agents:
            agent._x_coord = 
            agent._y_coord = 
            agent._z_coord = 
            agent._bandwidth = 
            agent._battery_level = 
            agent._throughput_request = 
            agent._edge_computing = 
            agent._data_gathering =  
            agent._d_ag_cc = 
        '''

    def reset_uavs(self, agent,):

        if (agent._battery_level == 0):
            agent._battery_level = FULL_BATTERY_LEVEL
            arise_pos_idx = np.random.choice(range(N_UAVS))
            arise_pos = self.initial_uavs_pos[arise_pos_idx]
            agent._x_coord = arise_pos[0]+0.5
            agent._y_coord = arise_pos[1]+0.5
            agent._z_coord = arise_pos[2]+0.5

            agent._charging = False
            agent._coming_home = False

    def update_users_requests(self, users):

        for user in users:
            # Update current user request only if the current user is not already requesting a service:
            if (user._info[1]==NO_SERVICE):
                
                type_of_service = scenario_objects.User.which_service()

                if (type_of_service == THROUGHPUT_REQUEST):
                    service_quantity = scenario_objects.User.bitrate_request()
                elif (type_of_service == EDGE_COMPUTING):
                    service_quantity = scenario_objects.User.edge_computing_request()
                elif (type_of_service == DATA_GATHERING):
                    service_quantity = scenario_objects.User.data_gathering()
                else:
                    service_quantity = 0

                requested_service_life = scenario_objects.User.needed_service_life(type_of_service) if type_of_service!=NO_SERVICE else 0

                user._info[1] = type_of_service
                user._info[2] = requested_service_life
                user._info[5] = service_quantity

    def move_users(self, current_iteration):

        for user_idx in range(self.n_users):
            # Move all the users at each iteration step only if the current iteration step is lower than the number of steps to move the users:
            if (current_iteration<self.k_steps_to_walk):
                self.users[user_idx]._x_coord = self.users_walk_steps[user_idx][current_iteration][0]/CELL_RESOLUTION_PER_COL
                self.users[user_idx]._y_coord = self.users_walk_steps[user_idx][current_iteration][1]/CELL_RESOLUTION_PER_ROW
                self.users[user_idx]._z_coord = self.users_walk_steps[user_idx][current_iteration][2]

    def compute_users_walk_steps(self):
        
        min_steps = 2
        max_steps = 5
        k_steps = np.random.random_integers(min_steps, max_steps)
        self.k_steps_to_walk = k_steps
        # Users random walk:
        users_walks = scenario_objects.User.k_random_walk(self.users, k_steps)
        #print(users_walks)
        self.users_walk_steps = users_walks

        # New clusters detection:
        if (FIXED_CLUSTERS_NUM>0):
            self.clusterer, self.users_clusters = scenario_objects.User.compute_clusterer(self.users) # --> SE USI UN NUMERO FISSO DI CLUSTER, ALLORA VARIA QUEL NUMERO FISSO; SE USI UN NUMERO VARIABILE, ALLORA RITORERAI PIU' VALORI DA QUESTO METODO --> !!!!!!!!!!!!!!!!
        else:
            optimal_clusterer, users_clusters, optimal_clusters_num, current_best_silhoutte_score = scenario_objects.User.compute_clusterer(self.users, fixed_clusters=False)
            self.clusterer = optimal_clusterer
        
        self.cluster_centroids = scenario_objects.User.actual_users_clusters_centroids(self.clusterer)
        self.clusters_radiuses = scenario_objects.User.actual_clusters_radiuses(self.cluster_centroids, self.users_clusters, FIXED_CLUSTERS_NUM) # --> Volendo potresti anche usare un numero di cluster variabile in modo da fittare meglio in clusters gli utenti che si sono mossi sulla mappa (a quel punto però metti 'fixed_cluster=False' in 'compute_clusterer(..)') --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # TO DO . . . --> ????????????????????????????????????
