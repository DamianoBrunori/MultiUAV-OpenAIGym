#------------------------Transition_Model------------------------#


'''
/********************
 * GLOBAL VARIABLES *
 ********************/
'''
# Random generator seed
SEED = None

import argparse
from configparser import ConfigParser
import re
import numpy as np
import math
import ast
#-----------------------------------------FILE CONFIGS-----------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="UAVs flight generator")
    parser.add_argument("-c","--config", default="configs.ini", type=str,metavar="Path",
    help="path to the flight generator config file")
    return parser.parse_args()

args = parse_args()

file = "configs.ini"
config = ConfigParser()
config.read(file)
# To access : config[<section>][<element>]

#Configs.ini
scenario = config['UAV']['scenario']
id = float(config['UAV']['id'])                               #1
vol = config['UAV']['vol']                                    #Zu volume the U-Space
OperationCategory = config['UAV']['OperationCategory']
AirRisk = config['UAV']['AirRisk']                            #UC
GroundRisk = config['UAV']['GroundRisk']                      #Density areas
Op_Type = config['UAV']['Op_Type']
#Model
model = config['UAV']['model']
#Dimension & Mass
Dimension = float(config['UAV']['Dimension'])                 #Dimension in m
m = float(config['UAV']['mass'])                              #massa in kg
#Payload
payload = float(config['UAV']['payload_mass'])                #payload in kg
#Cruise speed
cruise_speed_ms = float(config['UAV']['cruise_speed'])       #Velocità di crociera
#Manoeuvrability
VRD = float(config['UAV']['VRD'])
VRC = float(config['UAV']['VRC'])
stationary = float(config['UAV']['stationary'])               #space in metri

Maxwind = float(config['UAV']['Maxwind'])
PayloadRisk = config['UAV']['PayloadRisk']
T_Type = config['UAV']['T_Type']
FlightMode = config['UAV']['FlightMode']
Monitoring = config['UAV']['Monitoring']
TrackingService = config['UAV']['TrackingService']
TacticalSeparation = config['UAV']['TacticalSeparation']

#Start
s1 = config['UAV']
start_xyz = ast.literal_eval(s1.get('start_xyz'))
#Destination
dest_xyz = ast.literal_eval(s1.get('dest_xyz'))
#waypoint-set
if(s1.get("add_waypoint")):
    add_waypoint = ast.literal_eval(s1.get('add_waypoint'))
else:
    add_waypoint = None


get = s1.get('user_waypoints')
#user_waypoints =  list( ast.literal_eval() ) if(get) else  []

#altitudine
altitude = ast.literal_eval(s1.get('altitude'))

#Distance
distance_goal = math.sqrt((dest_xyz[0] - start_xyz[0]) ** 2 + (dest_xyz[1] - start_xyz[1]) ** 2) # Distanza drone goal
distance_space_m = float(config['UAV']['distance'])
if distance_space_m == 0:  #Se nel file configs.ini non è impostata una distanza da percorrere me la calcolo
    distance_space_m = distance_goal
#Scenario Time or distance

# T = 180.55
scenario_Time = False
if scenario_Time == True:
    T = distance_space_m / (cruise_speed_ms)  # Time (seconds for waypoint - waypoint movement)
#------------------------------------------PLOT-RANGE-------------------------------------------
if start_xyz[0] > dest_xyz[0]:
    PLOTRANGE_X_POS = start_xyz[0]
    PLOTRANGE_X_NEG = dest_xyz[0]
else:
    PLOTRANGE_X_NEG = start_xyz[0]
    PLOTRANGE_X_POS = dest_xyz[0]

if start_xyz[1] > dest_xyz[1]:
    PLOTRANGE_Y_POS = start_xyz[1]
    PLOTRANGE_Y_NEG = dest_xyz[1]
else:
    PLOTRANGE_Y_NEG = start_xyz[1]
    PLOTRANGE_Y_POS = dest_xyz[1]

if start_xyz[0] == dest_xyz[0]:
    PLOTRANGE_X_POS = start_xyz[0]+4
    PLOTRANGE_X_NEG = dest_xyz[0]-4
if start_xyz[1] == dest_xyz[1]:
    PLOTRANGE_Y_POS = start_xyz[1]+4
    PLOTRANGE_Y_NEG = dest_xyz[1]-4

PLOTRANGE_Z_POS = altitude
PLOTRANGE_Z_NEG = 0


'''PLOTRANGE_X_POS = 800
PLOTRANGE_X_NEG = 0 
PLOTRANGE_Y_POS = 400
PLOTRANGE_Y_NEG = 0
PLOTRANGE_Z_POS = 2000
PLOTRANGE_Z_NEG = 0'''
#---------------------------------------------------------------------------------------------------
##############################################Time-Acc##############################################
s_km05 = 25.21
s_km1 = 12.4
km1 = 1000
s_km2_1000m = 15.2
km2 = 2000
s_km3_1000m = 10.6
km3 = 3000
s_km4_1000m = 9
km4 = 4000
s_km5_1000m = 7
km5 = 5000
s_km6_1000m = 6
km6 = 6000
km7 = 7000
km8 = 8000
km9 = 9000
km10 = 10000
s_km789_10_1000m = 18
km20 = 20000
s_km20_1000m = 31.5
km30 = 30000
s_km30_1000m = 23
km40 = 40000
s_km40_1000m = 17
km47 = 47000
s_km47_1000m = 9


def distance_AB_2D(start,end):
    return math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) # Distanza drone start-end

def distance_AB_3D(start,end):
    return math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2 + (end[2] - start[2])**2) # Distanza drone start-end
