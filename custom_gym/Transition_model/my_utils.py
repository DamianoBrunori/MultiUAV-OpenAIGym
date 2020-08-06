#------------------------Transition_Model------------------------#


'''
/********************
 * GLOBAL VARIABLES *
 ********************/
'''
PLOTRANGE_X_POS = 3100
PLOTRANGE_X_NEG = -3100
PLOTRANGE_Y_POS = 3100
PLOTRANGE_Y_NEG = -3100
PLOTRANGE_Z_POS = 10
PLOTRANGE_Z_NEG = 0

# Random generator seed
SEED = None

import argparse
from configparser import ConfigParser
#-----------------------------------------FILE CONFIGS-----------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="UAVs flight generator")
    parser.add_argument("-c","--config", default="configs.yml", type=str,metavar="Path",
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
cruise_speed_kmh = float(config['UAV']['cruise_speed'])       #Velocità di crociera
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
T = space_m/cruise_speed_kmh            #Time (seconds for waypoint - waypoint movement)
#---------------------------------------------------------------------------------------------------