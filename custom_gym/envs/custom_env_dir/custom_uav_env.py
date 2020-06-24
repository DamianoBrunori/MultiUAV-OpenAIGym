import gym
import sys
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

        self.action_set = ACTION_SPACE_3D if DIMENSION_2D==False else ACTION_SPACE_2D
        self.action_space = spaces.Discrete(len(self.action_set))
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
        self.cs_cells = load.cs_cells
        # Set the CS position according to the desired resolution cells:
        uavs_pos = agent.Agent.setting_agents_pos(self.cs_cells)
        self.agents = agent.Agent.initialize_agents(uavs_pos) # Agents initialization
        for CS in self.cs_points:
            print("CS CELLS", (CS._x_coord, CS._y_coord, CS._z_coord))
        for ag in self.agents:
            print("UAV_POSooooooooooooooooooo", (ag._x_coord, ag._y_coord, ag._z_coord))
        self.eNB_cells = load.enb_cells
        self.points_status_matrix = load.points_status_matrix
        self.cells_status_matrix = load.cells_status_matrix
        self.clusterer = load.initial_clusterer
        self.cluster_centroids = load.initial_centroids 
        self.clusters_radiuses = load.initial_clusters_radiuses
        initial_users = load.initial_users
        for user in initial_users:
            # Set the users coordinates according to the desired resolution cell:
            user._x_coord /= CELL_RESOLUTION_PER_COL 
            user._y_coord /= CELL_RESOLUTION_PER_ROW
        self.users = initial_users
        self.n_users = len(self.users)
        #self.n_features = 2

    def step_2D_limited_battery(self, agent, action):
        # - 2D case;
        # - LIMITED battery;
        # - Constat battery consumption wich includes both services and motions of the UAVs;
        # - All the users have the same priority (each of the is served) and and ask for the same service;
        # - Infinite UAV bandwidth;

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        agent_pos = self.get_2Dagent_pos(agent)
        info = ""
        current_action = self.action_set[action]

        agent_pos_ = agent.move_2D_limited_battery(agent_pos, current_action) # --> 'move_2D_limited_battery' method includes also 'off_map_move_2D' check.

        agent._x_coord = agent_pos_[0]
        agent._y_coord = agent_pos_[1]

        reward = self.reward_function_1(agent._users_in_footprint)

        s_ = (agent_pos_, agent._battery_level)

        done = self.is_terminal_state(s_)

        if (done):
            reward = 1.0 # --> da decidere --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            done = True
            info = "TERMINAL" 
        else:
            #reward = 0.0
            info = "NOT TERMINAl"

        return s_, reward, done, info

    def step_2D_unlimited_battery(self, agent, action):
        # - 2D case;
        # - UNLIMITED battery;
        # - Constat battery consumption wich includes both services and motions of the UAVs;
        # - All the users have the same priority (each of the is served) and and ask for the same service; 
        # - Infinite UAV bandwidth;

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        agent_pos = self.get_2Dagent_pos(agent)
        info = ""
        current_action = self.action_set[action]
        #print("ACTION:", current_action)

        agent_pos_ = agent.move_2D_unlimited_battery(agent_pos, current_action)

        agent._x_coord = agent_pos_[0]
        agent._y_coord = agent_pos_[1]

        reward = self.reward_function_1(agent._users_in_footprint)

        s_ = (agent_pos_)

        done = self.is_terminal_state(s_)

        if (done):
            reward = 1.0 # --> da decidere --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            done = True
            info = "TERMINAL" 
        else:
            #reward = 0.0
            info = "NOT TERMINAl"

        return s_, reward, done, info

    def step_3D_limited_battery(self, agent, action):
        # - 3D case;
        # - LIMITED battery;
        # - Constat battery consumption wich includes both services and motions of the UAVs;
        # - All the users have the same priority (each of the is served) and and ask for the same service; 
        # - Infinite UAV bandwidth;

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        agent_pos = self.get_3Dagent_pos(agent)
        info = "" # --> lo utilizzo per capire chi effettivamente ha vinto, così al termine dell' episodio posto stampare il massimo punteggio raggiungibile (10) per il vincitore (altrimenti avrei stampato sempre 9 per il vincitore)
        current_action = self.action_set[action]

        agent_pos_ = agent.move_3D_limited_battery(agent_pos, current_action) # --> 'move_3D_unlimited_battery' method includes also 'off_map_move_3D' check.

        agent._x_coord = agent_pos_[0]
        agent._y_coord = agent_pos_[1]
        agent._z_coord = agent_pos_[2]
        
        reward = self.reward_function_1(agent._users_in_footprint)

        s_ = (agent_pos_, agent._battery_level)

        done = self.is_terminal_state(s_)

        if (done):
            reward = 1.0 # --> da decidere --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            done = True
            info = "TERMINAL" 
        else:
            #reward = 0.0
            info = "NOT TERMINAl"

        return s_, reward, done, info

    def step_3D_unlimited_battery(self, agent, action):
        # - 3D case;
        # - UNLIMITED battery;
        # - Constat battery consumption wich includes both services and motions of the UAVs;
        # - All the users have the same priority (each of the is served) and and ask for the same service; 
        # - Infinite UAV bandwidth;

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        agent_pos = self.get_3Dagent_pos(agent)
        info = "" # --> lo utilizzo per capire chi effettivamente ha vinto, così al termine dell' episodio posto stampare il massimo punteggio raggiungibile (10) per il vincitore (altrimenti avrei stampato sempre 9 per il vincitore)
        current_action = self.action_set[action]

        agent_pos_ = agent.move_3D_unlimited_battery(agent_pos, current_action) # --> 'move_3D_unlimited_battery' method includes also 'off_map_move_3D' check.

        agent._x_coord = agent_pos_[0]
        agent._y_coord = agent_pos_[1]
        agent._z_coord = agent_pos_[2]
        
        reward = self.reward_function_1(agent._users_in_footprint)

        s_ = (agent_pos_)

        done = self.is_terminal_state(s_)

        if (done):
            reward = 1.0 # --> da decidere --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            done = True
            info = "TERMINAL" 
        else:
            #reward = 0.0
            info = "NOT TERMINAl"

        return s_, reward, done, info

    def step(self, agent, action, timeslot):

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
        current_action = self.action_set[action]

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
        users_in_footprint = agent.users_in_uav_footprint(self.users, UAV_FOOTPRINT) # --> IMPORTA GLI USERS --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        agent._users_in_footprint = users_in_footprint
        n_served_users = agent.n_served_users_in_foot_and_type_of_service(users_in_footprint)
        agent.set_not_served_users(self.users)
        
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
        done = self.is_terminal_state(s_)

        if (done):
            reward = 1.0 # --> da decidere --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            done = True
            info = "TERMINAL" 
        else:
            #reward = 0.0
            info = "NOT TERMINAl"

        return s_, reward, done, info

    # Reward function which takes into account only the percentage of covered users:
    def reward_function_1(self, users_in_footprint):

        n_users_in_footprint = len(users_in_footprint)
        reward = n_users_in_footprint/self.n_users

        return reward

    # _____________________________________________________________________________________________________________________________________________
    # Reward function provvisoria con grandezze non scalate/normalizzate: 
    def reward_function(self, agent):

        reward = scenario_objects.User.bitrate_request() + scenario_objects.User.edge_computing_request() + scenario_objects.User.data_gathering()

        return reward
    # _____________________________________________________________________________________________________________________________________________


    def show_scores(self):
        
        # TO DO . . . --> FORSE --> ?????????????????????????????????????????????????

        pass

    def is_terminal_state(self, state):

        # TO DO . . . --> Decidi quando uno stato è terminale, ossia se è stato raggiunto un eventuale goal oppure se l'environment deve essere resettato --> !!!!!!!!!!!!!!!!!!!!!!!

        return False
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

    def render(self, agents_paths, where_to_save):

        plot.plt_map_views(obs_cells=self.obs_cells, cs_cells=self.cs_cells, enb_cells=self.eNB_cells, 
            points_status_matrix=self.points_status_matrix, cells_status_matrix=self.cells_status_matrix, 
            users=self.users, centroids=self.cluster_centroids, clusters_radiuses=self.clusters_radiuses, 
            area_height=AREA_HEIGHT, area_width=AREA_WIDTH, N_cells_row=CELLS_ROWS, N_cells_col=CELLS_COLS, 
            agents_paths=agents_paths, path_animation=True, where_to_save=where_to_save)

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
    # NOT USED
    def refresh(self):
        
        # Users random walk:
        users_walks = scenario_objects.User.k_random_walk(users, k)

        for user_id in range(self.n_users):
            self.users[user_id]._x_coord = users_walks[user_id[0]]
            self.users[user_id]._y_coord = users_walks[user_id[1]]
            self.users[user_id]._z_coord = users_walks[user_id[2]]

        # New clusters detection:
        self.clusterer = scenario_objects.User.compute_clusterer(self.users) # --> SE USI UN NUMERO FISSO DI CLUSTER, ALLORA VARIA QUEL NUMERO FISSO; SE USI UN NUMERO VARIABILE, ALLORA RITORERAI PIU' VALORI DA QUESTO METODO --> !!!!!!!!!!!!!!!!
        self.cluster_centroids = scenario_objects.User.actual_users_clusters_centroids(self.clusterer)

        # TO DO . . . --> ????????????????????????????????????

        pass