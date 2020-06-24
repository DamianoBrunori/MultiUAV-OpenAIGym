# UAV RL Thesis

### Table of Contents
You're sections headers will be used to reference location of destination.

- [Description](#description)
- [How To Use](#how-to-use)
- [References](#references)
- [License](#license)
- [Author Info](#author-info)

---

## Description

The aim of this project is to test some RL approaches in a 'new' multi-agent application, by comparing the different results obtained with different techniques and parameters setting. The application context is based on a multi-agent system made up by a varibale number of UAVs which are able to provide one or more (up to three) services to cluster(s) of users who request it. All the the Environment objects (obstacles, drones, grid-map, users, . . .) have been created from scratch in Python. The methods related to the trainining part are made by creating a custom environment with custom methods. All this is made so that my environment is consistent with the OpenAI Gym API. It is possibile to:

- create different environments of different xy (2D) and z (3D) dimensions;
- set objects of different heights and with a different distribution;
- use a desired resolution for the 'xy plane-grid' if you want to make the agents able to detect objects according to a resolution which is larger than the minimum one (the minimum resolution allows the agent to detect perfectly every obstacles and it is based on xy plane-squares of side 1);
- set a variable number of UAVs and charging stations (or even no charging station);
- create a base station (eNodeB) or no;
- set the number of users clusters and in case make users move according to a random walk;
- set the radius of the UAVs footprint and select a single-service or a multi-service system;
- select either a continue (infinite in time) or a discrete (variable in time) service request coming from users;
- select the the users priority (i.e either all the same or differentiated according to the user account, such as 'free user', 'premium user', . . .);
- training parameters like LR, EPSILON, DISCOUNT, . . .;
- . . .     

---

## How To Use

#### Programming Language

- Python (version 3.6)

#### Dependencies 

- gym
- Numpy
- decimal
- sklearn
- Matplotlib
- mpl_toolkits
- [pylayers](https://github.com/pylayers/pylayers/blob/master/INSTALL.md) (not for now)
- . . .



#### Notes

Before running the code you have to install my custom environment 'UAVEnv-v0', which consistents with OpenAI Gym. In order to do this, you have to navigate through your terminal/prompt to the folder containing 'setup.py', i.e. to 'UAV_RL_Thesis/custom_gym/', and type:

- pip install -e .

Every time you modify 'custom_uav_env.py', you have to perform this procedure before running the other files needed to run the simulation.

#### Run

The files must be ran in the following order:

- scenario_objects.py
- plotting.py
- env_wrapper.py


### Docker
Install Docker:
https://docs.docker.com/get-docker/

#### Ubuntu
Give permission to docker user on xhost:
```console
$ xhost +"local:docker@"
```
Build Docker Ubuntu based image named docker-uav-rl from Dockerfile:
```console
$ docker build -t docker-uav-rl:latest .
```
Run image :
```console
$ docker run --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" docker-uav-rl
```
#### Windows
...
#### MacOs
...

---

## Code structure
```
├── Cases
│   └── 3D_un_bat_inf_req_1UAVs_1clusters
│       └── QTables
│           ├── qtable-ep1.npy
│           └── UAV1
├── custom_gym
│   ├── agent.py
│   ├── Astar.py
│   ├── Cases
│   │   └── 3D_un_bat_inf_req_1UAVs_1clusters
│   │       ├── 1.png
│   │       ├── QoE1.png
│   │       ├── QoE2.png
│   │       ├── QTables
│   │       │   ├── qtable-ep1.npy
│   │       │   ├── UAV1
│   │       │   └── UAV1\qtable_graph-ep1.png
│   │       └── q_tables.pickle
│   ├── custom_uav_env.egg-info
│   │   ├── dependency_links.txt
│   │   ├── PKG-INFO
│   │   ├── requires.txt
│   │   ├── SOURCES.txt
│   │   └── top_level.txt
│   ├── envs
│   │   ├── custom_env_dir
│   │   │   ├── custom_uav_env.py
│   │   │   ├── __init__.py
│   │   │   └── __pycache__
│   │   │       ├── custom_uav_env.cpython-36.pyc
│   │   │       ├── custom_uav_env.cpython-37.pyc
│   │   │       ├── environment.cpython-36.pyc
│   │   │       ├── __init__.cpython-36.pyc
│   │   │       ├── __init__.cpython-37.pyc
│   │   │       ├── load_and_save_data.cpython-36.pyc
│   │   │       ├── my_utils.cpython-36.pyc
│   │   │       ├── prova.cpython-36.pyc
│   │   │       └── utils.cpython-36.pyc
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── __init__.cpython-36.pyc
│   │       └── __init__.cpython-37.pyc
│   ├── env_wrapper.py
│   ├── figures
│   ├── initial_users
│   │   ├── initial_centroids.npy
│   │   ├── initial_clusterer.npy
│   │   ├── initial_clusters_radiuses.npy
│   │   └── initial_users.npy
│   ├── load_and_save_data.py
│   ├── map_data
│   │   ├── cells_matrix.npy
│   │   ├── cs_cells.npy
│   │   ├── cs_points.npy
│   │   ├── enb_cells.npy
│   │   ├── eNB_point.npy
│   │   ├── obs_cells.npy
│   │   ├── obs_points.npy
│   │   └── points_matrix.npy
│   ├── map_status
│   │   ├── cells_status_matrix.npy
│   │   ├── perceived_status_matrix.npy
│   │   └── points_status_matrix.npy
│   ├── my_utils.py
│   ├── networking.py
│   ├── plotting.py
│   ├── __pycache__
│   │   ├── agent.cpython-36.pyc
│   │   ├── agent.cpython-37.pyc
│   │   ├── environment.cpython-36.pyc
│   │   ├── load_and_save_data.cpython-36.pyc
│   │   ├── load_and_save_data.cpython-37.pyc
│   │   ├── my_utils.cpython-36.pyc
│   │   ├── my_utils.cpython-37.pyc
│   │   ├── plotting.cpython-36.pyc
│   │   ├── plotting.cpython-37.pyc
│   │   ├── scenario_objects.cpython-36.pyc
│   │   └── scenario_objects.cpython-37.pyc
│   ├── scenario_objects.py
│   └── setup.py
├── Dockerfile
├── initial_users
│   ├── initial_centroids.npy
│   ├── initial_clusterer.npy
│   ├── initial_clusters_radiuses.npy
│   └── initial_users.npy
├── map_data
│   ├── cells_matrix.npy
│   ├── cs_cells.npy
│   ├── cs_points.npy
│   ├── enb_cells.npy
│   ├── eNB_point.npy
│   ├── obs_cells.npy
│   ├── obs_points.npy
│   └── points_matrix.npy
├── map_status
│   ├── cells_status_matrix.npy
│   ├── perceived_status_matrix.npy
│   └── points_status_matrix.npy
├── __pycache__
│   ├── environment.cpython-36.pyc
│   ├── grid.cpython-36.pyc
│   ├── load_and_save_data.cpython-36.pyc
│   ├── macro.cpython-36.pyc
│   ├── my_utils.cpython-36.pyc
│   ├── plotting.cpython-36.pyc
│   ├── prova4.cpython-36.pyc
│   └── utils.cpython-36.pyc
├── README.md
├── requirements.txt
└── run.sh

```

## References

. . .

---

## License

This project is intended for private use only. All rights are reserved and it is not Open Source or Free. You cannot modify or redistribute this code without explicit permission from the copyright holder. 

---

## Author Info

- Name: Damiano
- Surname: Brunori
- University: Sapienza University of Rome
- Master: [Artificial Intelligence and Robotics](https://corsidilaurea.uniroma1.it/it/corso/2019/30431/home)
- private e-mail: damiano.brunori@libero.it
- university e-mail: brunori.1583073@studenti.uniroma1.it 

[Back To The Top](#read-me-template)
