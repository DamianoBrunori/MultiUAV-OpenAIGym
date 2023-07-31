# MultiUAV-OpenAIGym

### Table of Contents

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

- Python (version >= 3.6)

#### Dependencies 

- gym (0.10.5)
- Numpy (1.15.2)
- scipy
- sklearn (0.19.1)
- Matplotlib (2.1.2)
- mpl_toolkits (the same as Matplotlib)
- ImageMagick
- [pylayers](https://github.com/pylayers/pylayers/blob/master/INSTALL.md) (not for now)

To install ImageMagick on UBUNTU, type the following commands from your terminal:
    
    - sudo apt update
    - sudo apt install imagemagick

To install ImageMagick on WINDOWS, simply download and install ImageMagick from [https://imagemagick.org/script/download.php](https://imagemagick.org/script/download.php).

### Setup

Before running the code you have to setup my custom environment 'UAVEnv-v0', which consistents with OpenAI Gym. In order to do this you can install all you nedd very quicly by using 1. Pip or 2. Docker.

#### 1. Pip

Navigate through your terminal/prompt to the folder containing 'setup.py', i.e. to 'UAV_RL_Thesis/custom_gym/', and type:

- pip install -e .

Now all needed dependencies are installed.

#### 2. Docker
Install Docker:
https://docs.docker.com/get-docker/

###### Using Jupyter
Build Docker Ubuntu based image named docker-uav-rl from Dockerfile:
```console
$ docker build -t docker-uav-rl:latest .
```
Run image opening port 8888 (change it if already in use):
```console
$ docker run -p 8888:8888 docker-uav-rl:latest 
```
Now that our image is running...
1. Open the link specified in terminal
2. In browser you should see the current file structure
3. Open the notebook file main.ipynb 
4.  **run all cells**

###### Ubuntu
**NO MORE AVAILABLE**

...
###### Windows
...
###### MacOs
...

---

#### Run

The files must be ran in the following order:

- scenario_objects.py
- plotting.py
- env_wrapper.py

Every time you want to modify the scenario or the training parameters, you can do it by simply changing the desired values in 'my_utils.py'. After doing this, you have to run again 'scenario_object.py' (to generate the new scenario according to the new selected parameters) and 'plotting.py' (to save relevant data and visualize your new scenario). Now you are ready to start the training by running 'env_wrapper.py' (if you want to run multiple times a training for the same scenario, you can just run this last file after launching once the other two files). 

## References

D. Brunori, S. Colonnese; F. Cuomo, L. Iocchi “A Reinforcement Learning Environment for Multi-Service UAV-enabled Wireless Systems”, PerVehicle 2021 3rd International Workshop on Pervasive Computing for Vehicular Systems colocated with PERCOM 2021, March 2021.

---

## MIT License

Copyright (c) 2020 Damiano Brunori

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE 

---

## Author Info

- Name: Damiano
- Surname: Brunori
- University: Sapienza University of Rome
- Master: [Artificial Intelligence and Robotics](https://corsidilaurea.uniroma1.it/it/corso/2019/30431/home)
- university e-mail: brunori@diag.uniroma1.it
- private e-mail damiano.brunori2@gmail.com 

[Back To The Top](#read-me-template)
