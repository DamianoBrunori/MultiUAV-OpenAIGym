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
import numpy as np
import math
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
#Distance
space_m = float(config['UAV']['space'])
#T = space_m/(cruise_speed_ms*3.6)            #Time (seconds for waypoint - waypoint movement)
#---------------------------------------------------------------------------------------------------
PLOTRANGE_X_POS = 5
PLOTRANGE_X_NEG = -2000
PLOTRANGE_Y_POS = 500
PLOTRANGE_Y_NEG = -5
PLOTRANGE_Z_POS = 8
PLOTRANGE_Z_NEG = 0

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

