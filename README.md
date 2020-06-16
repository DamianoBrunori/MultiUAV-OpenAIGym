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

- pip install e .

Every time you modify 'custom_uav_env.py', you have to perform this procedure before running the other files needed to run the simulation.

#### Run

The files must be ran in the following order:

- scenario_objects.py
- plotting.py
- env_wrapper.py

---

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
