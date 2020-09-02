from math import cos, sin
from numpy.random import seed
from numpy.random import randint
from os import mkdir
from os.path import join, isdir
import numpy as np
import math
from Quadrotor import Quadrotor
from TrajectoryGenerator import TrajectoryGenerator
# sys.path.append(os.path.abspath("../custom_gym"))
from my_utils import *

import sys

from os import getenv

import datetime

LOG_DIRECTORY_NAME = "Logs_of_flights"

LOG_FILENAME = "log_"+str( datetime.datetime.now()).replace(" ","_") +".txt"

from datetime import datetime




LOG_DIRECTORY_NAME = "Logs_of_flights"
timeformatted = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
semiformatted = timeformatted.replace("-", "_")
almostformatted = semiformatted.replace(":", "_")
formatted = almostformatted.replace(".", "")
withspacegoaway = formatted.replace(" ", "--")
formattedstripped = withspacegoaway.strip()

LOG_FILENAME = "log_"+formattedstripped+".txt"


LOG_PATH = join(LOG_DIRECTORY_NAME, LOG_FILENAME)

# Class to redirect stdout to file logfile.log
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(LOG_PATH, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    



#----------------------------------------------------------------------------------------------------------------------------------#
info = []

info1 = "\n\n___________________________________________ENVIRONMENT AND DRONE INFO: ___________________________________________\n"
info.append(info1)
info24 = "\nWP3 SCENARIO: " + str(scenario)
info.append(info24)
info2 = "\nDRONE ID: " + str(id)
info.append(info2)
info3 = "\nVOL: " + str(vol)
info.append(info3)
info4 = "\nAIR RISK: " + str(AirRisk)
info.append(info4)
info5 = "\nGROUND RISK: " + str(GroundRisk)
info.append(info5)
info6 = "\nOP TYPE: " + str(Op_Type)
info.append(info6)
info7 = "\nTYPE OF DRONE: " + str(model)
info.append(info7)
info8 = "\nDIMENSION: " + str(Dimension) + " m"
info.append(info8)
info9 = "\nMASS: " + str(m) + " Kg"
info.append(info9)
info22 = "\nCRUISE SPEED: " + str(cruise_speed_ms) + " m/s"
info.append(info22)
info10 = "\nVDR: " + str(VRD) + " m/s"
info.append(info10)
info11 = "\nVRC: " + str(VRC) + " m/s"
info.append(info11)
info12 = "\nStationary Max: " + str(stationary) + " min"
info.append(info12)
info13 = "\nMAX WIND: " + str(Maxwind) + " m/s"
info.append(info13)
info14 = "\nPAYLOAD RISK: " + str(PayloadRisk)
info.append(info14)
info15 = "\nT. TYPE: " + str(T_Type)
info.append(info15)
info16 = "\nFLIGHT MODE: " + str(FlightMode)
info.append(info16)
info17 = "\nMONITORING: " + str(Monitoring)
info.append(info17)
info18 = "\nTRACKING SERVICE: " + str(TrackingService)
info.append(info18)
info19 = "\nTACTICAL SEPARATION: " + str(TacticalSeparation)
info.append(info19)
info20 = "\nDISTANCE TO REACH THE GOAL: " + str(space_m) + " m"
info.append(info20)
#info21 = "\nTIME TO REACH THE GOAL: " + str(T) + " s"
#info.append(info21)
info25 = "\nSTART COORDINATES: " + "X:" + str(start_xyz[0]) + " Y:" + str(start_xyz[1]) + " Z:" + str(start_xyz[2])
info.append(info25)
info26 = "\nDESTINATION COORDINATES: " + "X:" + str(dest_xyz[0]) + " Y:" + str(dest_xyz[1]) + " Z:" + str(dest_xyz[2])
info.append(info26)


info23 = "\n__________________________________________________________________________________________________________________\n\n"
info.append(info23)

if not isdir(LOG_DIRECTORY_NAME): mkdir(LOG_DIRECTORY_NAME)
file = open(LOG_PATH, "w")

for i in info:
    print(i)
    file.write(i)
file.close()
#----------------------------------------------------------------------------------------------------------------------------------#

show_animation = True

# Simulation parameters

g = 9.81                                #Gravity (m/s^-2)
#m = 0.2                                #Massa (Kg)
Ixx = 1
Iyy = 1
Izz = 1
#T = space_m/cruise_speed_kmh            #Time (seconds for waypoint - waypoint movement)
T = 120                     #Time (seconds for waypoint - waypoint movement)
cruise_speed_kmh = cruise_speed_ms/3.6   #Cruise speed km/h
T_s = T/5                                #Tempo fino a quando avviene un' accelerazione
T_s2 = T - T_s                           #Tempo dopo il quale avviene una decellerazione

# Proportional coefficients
Kp_x = 1
Kp_y = 1
Kp_z = 1
Kp_roll = 25
Kp_pitch = 25
Kp_yaw = 25

# Derivative coefficients
# Kd_x = 10
# Kd_y = 10
Kd_z = 1


print(start_xyz,"\n", dest_xyz)
waypoint1= [dest_xyz[0], dest_xyz[1], dest_xyz[2]]               #con file configs
start_pos = np.array([start_xyz[0], start_xyz[1], start_xyz[2]]) #con file configs
acc_max = 3
##################################Senza-file-configs################################
#waypoint1 = [1100, 0, 5]
#start_x = 0
#start_y = 0
#start_z = 5
#start_pos = np.array([start_x, start_y, start_z])
####################################################################################

#------------------------------------------------------Gestione-del-Tempo----------------------------------------------------------#
diff_metri = 0
incrementoT = 0

distance_start_goal = math.sqrt((waypoint1[0] - start_pos[0]) ** 2 + (waypoint1[1] - start_pos[1]) ** 2) # Distanza drone goal

def increment_flight_time(start_point,end_point):
    if (1) <= end_point < km1 / 2:
        T = 10.39
        diff_metri = start_point - 1
        incrementoT = diff_metri * (s_km05 / 500)  # su 500m
    elif (km1 / 2) <= end_point < km1:
        T = 35.6
        diff_metri = start_point - km1 / 2
        incrementoT = diff_metri * (s_km1 / 500)  # su 500m
    elif (km1 <= end_point < km2):
        T = 48
        diff_metri = start_point - km1
        incrementoT = diff_metri*(s_km2_1000m/1000)   #su 1000m 9s/1000=0.009s al metro -> aggiungo 0.9s ogni 100 metri
    elif (km2 <= end_point < km3):
        T = 63.2
        diff_metri = start_point - km2
        incrementoT = diff_metri*(s_km3_1000m/1000)   #su 1000m
    elif (km3 <= end_point < km4):
        T = 73.8
        diff_metri = start_point - km3
        incrementoT = diff_metri*(s_km4_1000m/1000)   #su 1000m
    elif (km4 <= end_point < km5):
        T = 82
        diff_metri = start_point - km4
        incrementoT = diff_metri * (s_km5_1000m / 1000)  # su 1000m
    elif (km5 <= end_point < km6):
        T = 89
        diff_metri = start_point - km5
        incrementoT = diff_metri * (s_km5_1000m / 1000)  # su 1000m
    elif (km6 <= end_point < km7):
        T = 95
        diff_metri = start_point - km6
        incrementoT = diff_metri * (s_km6_1000m / 1000)  # su 1000m
    elif (km7 <= end_point < km10):
        T = 101
        diff_metri = start_point - km7
        incrementoT = diff_metri * (s_km789_10_1000m / 4000)  # su 4000m
    elif (km10 <= end_point < km20):
        T = 114.5
        diff_metri = start_point - km10
        incrementoT = diff_metri * (s_km20_1000m / 10000)  # su 10000m
    elif (km20 <= end_point < km30):
        T = 146
        diff_metri = start_point - km20
        incrementoT = diff_metri * (s_km30_1000m / 10000)  # su 10000m
    elif (km30 <= end_point < km40):
        T = 169
        diff_metri = start_point - km30
        incrementoT = diff_metri * (s_km40_1000m / 10000)  # su 10000m
    elif (km40 <= end_point < km47):

        T = 186
        diff_metri = start_point - km40
        incrementoT = diff_metri * (s_km47_1000m / 7000)  # su 7000m
    T = T + incrementoT
    print("T:", T, "T_incremento:", incrementoT)
    return T



T = increment_flight_time(distance_start_goal,waypoint1[0])

#----------------------------------------------------------------------------------------------------------------------------------#
def quad_sim(x_c, y_c, z_c):
    
    """
    Calculates the necessary thrust and torques for the quadrotor to
    follow the trajectory described by the sets of coefficients
    x_c, y_c, and z_c. ##Spinta e Coppia
    """

    SALVO_PROPORZIONE = [0, 0]
    x_pos = start_xyz[0]
    y_pos = start_xyz[1]
    z_pos = start_xyz[2]
   
    x_vel = 0
    y_vel = 0
    z_vel = 0

    x_acc = 0
    y_acc = 0
    z_acc = 0

    roll = 0
    pitch = 0
    yaw = 0

    roll_vel = 0
    pitch_vel = 0
    yaw_vel = 0

    des_yaw = 0 #mantenere orientazione 0 quindi rimane fisso il  drone non gira su se stesso

    dt = 0.1
    t = 0
    t2 = 50
    t1 = 0
    UAV = Quadrotor(id = "uav", x=x_pos, y=y_pos, z=z_pos, roll=roll,
                  pitch=pitch, yaw=yaw, size=1, show_animation=show_animation)
    ciao = False
    fase_dec_x = False
    fase_dec_y = False
    Salvo_prop_acc = True
    Salvo_prop_dec = False
    i = 0
    n_run = 4 #Numero di Round (quanti waypoints vuoi vedere)
    irun = 0

    ok = True
    SALVO_TEMPO = 0
    t_dec = 0
    SALVO_DISTANZA_FATTA = 0
    salvo_acc_x = 0
    salvo_acc_y = 0
    salvo_acc_z = 0
    while True:
        while t <= T:
            #DIVIDO LA TRAIETTORIA IN PEZZETTI si fissano degli obiettivi molto vicini (interpolazione)
            # des_x_pos = calculate_position(x_c[i], t)
            # des_y_pos = calculate_position(y_c[i], t)
            des_z_pos = calculate_position(z_c[i], t)


            des_x_vel = calculate_velocity(x_c[i], t)       #destinazione
            des_y_vel = calculate_velocity(y_c[i], t)
            des_z_vel = calculate_velocity(z_c[i], t)


            des_x_acc = calculate_acceleration(x_c[i], t)
            des_y_acc = calculate_acceleration(y_c[i], t)
            des_z_acc = calculate_acceleration(z_c[i], t)


            PosizioneAttuale = np.array([x_pos, y_pos, z_pos])
            goal = np.array(waypoint1)
            dimension3D=np.array([30000.0,40000.0,50000.0])
            distanceAB_3D = distance(PosizioneAttuale, goal, dimension3D) #Distanza drone goal
            print(x_pos, y_pos, z_pos)

            distanceAB_2D = math.sqrt((waypoint1[0] - PosizioneAttuale[0]) ** 2 + (waypoint1[1] - PosizioneAttuale[1]) ** 2) # Distanza drone goal

            dist_percorsa3D = distance(start_pos, PosizioneAttuale, dimension3D) #Distanza percorsa

            dist_percorsa2D = math.sqrt((start_pos[0] - PosizioneAttuale[0]) ** 2 + (start_pos[1] - PosizioneAttuale[1]) ** 2)



            #print(des_x_acc," des_x_acc")
            #print(des_z_vel, "des_z_vel") sempre 0
            #print("time: ", t, i)
            # print([v for v in zip((x_c[i]),labels) ],t)
            # print(x_c[i], t)
            #print(y_c[i], t)

            acc_des_xyz = math.sqrt(des_x_acc ** 2 + des_y_acc ** 2 + des_z_acc ** 2)
            #print("Accelerazione desiderata: ", acc_des_xyz)
            

            vel_ms = math.sqrt(x_vel**2 + y_vel**2)
            acc_ms = math.sqrt(x_acc ** 2 + y_acc ** 2)


            
            vel_kmh =  vel_ms * 3.6
            acc_kmh =  acc_ms * 3.6

            #CONTROLLORI
            thrust = m * (g + des_z_acc + Kp_z * (des_z_pos -
                                                  z_pos) + Kd_z * (des_z_vel - z_vel))

            roll_torque = Kp_roll * \
                (((des_x_acc * sin(des_yaw) - des_y_acc * cos(des_yaw)) / g) - roll)
            pitch_torque = Kp_pitch * \
                (((des_x_acc * cos(des_yaw) - des_y_acc * sin(des_yaw)) / g) - pitch)
            yaw_torque = Kp_yaw * (des_yaw - yaw)

            #bang- coast- bang ACCELERAZIONE - VELOCITà COSTANTE - DECELLERAZIONE
            #Interpolation using splines per usare i polinomi e spezzetta la traiettoria


            roll_vel += roll_torque * dt / Ixx
            pitch_vel += pitch_torque * dt / Iyy
            yaw_vel += yaw_torque * dt / Izz

            #ANGOLI CHE DEFINISCONO L'ORIENTAZIONE DEL DRONE
            roll += roll_vel * dt       #Spostamento verso destra o sinistra PIEGARLO IN AVANTI O ALL'INSù
            pitch += pitch_vel * dt     #Spostamento verso avanti o dietro   RUOTARLO DI LATO
            yaw += yaw_vel * dt         #Rotazione sul proprio asse          RUOTARLO SU SE STESSO

            R = rotation_matrix(roll, pitch, yaw) #CAMBIA IL MODO PERO' UGUALE
            acc = (np.matmul(R, np.array(
                [0, 0, thrust.item()]).T) - np.array([0, 0, m * g]).T) / m
            #CON LA NUOVA ORIENTAZIONE R E LA TRUST, IN BASE A COME TI SEI ORIENTATO SPINGI(TRUST) E ARRIVI


            print("x:vel", "{:.2f}".format(x_vel),"(m/s)")
            print("y:vel", "{:.2f}".format(y_vel),"(m/s)")
            #print("Acc[0]", "{:.2f}".format(acc[0]),"(m/s^2")
            #print("des_y_acc", "{:.2f}



            # ------------------------------------------------------------------------------------------------------------------#

            if (vel_ms > cruise_speed_ms and distanceAB_2D > SALVO_DISTANZA_FATTA):
                if (ok == True):
                    SALVO_TEMPO = t - 0.1
                    SALVO_DISTANZA_FATTA = dist_percorsa2D                   #Salvo distanza percorsa (in quel momento)
                    if (SALVO_TEMPO > 0 and SALVO_DISTANZA_FATTA > 0):
                        t_dec = T - SALVO_TEMPO                              #Salvo t_dec (tempo di decellerazione) T - (Tempo impiegato a raggiungere la velocità di crociera)
                        SALVO_PROPORZIONE = [x_vel / vel_ms, y_vel / vel_ms] #SALVO_PROPORZIONE (delle velocità sugli assi in quel momento)
                        ok = False                                           #Non aggiorno più queste variabili
                t = -0.1  # non c'è accellerazione
                x_acc = 0
                y_acc = 0

            else:
                x_acc = acc[0]
                y_acc = acc[1]
            z_acc = acc[2]
                #SALVO_PROPORZIONE = [x_vel / 2, y_vel / 2]


            if (distanceAB_2D < SALVO_DISTANZA_FATTA):
                if (t < 1):
                    t = t_dec                                                                       #Imposto il tempo uguale a t_dec (inizia la fase di decellerazione)

            if (distanceAB_2D <= 6 and SALVO_PROPORZIONE[0] != 0 and SALVO_PROPORZIONE[1] != 0):    #Se la diztanza dal goal e minore di 6 e le proporzioni sono 0 (quindi non si è raggiunta una velocità max)
                if (x_vel > 0.2 or fase_dec_x == True):
                    x_acc = 0
                    x_vel = SALVO_PROPORZIONE[0]
                    fase_dec_x = True
                if (y_vel > 0.2 or fase_dec_y == True):
                    y_acc = 0
                    y_vel = SALVO_PROPORZIONE[1]
                    fase_dec_y = True
                t = t - 0.1

            if (waypoint1[0] < 0):
                if (waypoint1[0] < x_pos and distanceAB_2D < 2.5):
                    x_acc = 0
                    x_vel = -0.1
                    print("Sono xxxxxx neeeeeeeeeeeeeeeeeeeeeeeggggggggggggggggggggggg")
                if (waypoint1[0] >= x_pos):
                    x_acc = 0
                    x_vel = 0
            if (waypoint1[0] > 0):
                if (waypoint1[0] > x_pos and distanceAB_2D < 2.5):
                    x_acc = 0
                    x_vel = 0.1
                    print("Sono xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                if (waypoint1[0] <= x_pos):
                    x_acc = 0
                    x_vel = 0
            if (waypoint1[1] < 0):
                if (waypoint1[1] < y_pos and distanceAB_2D < 2.5):
                    y_acc = 0
                    y_vel = -0.1
                    print("Sono yyyyy neeeeeeeeeeeeeeeeeeeeeeeggggggggggggggggggggggg")
                if (waypoint1[1] >= y_pos):
                    y_acc = 0
                    y_vel = 0
            if (waypoint1[1] > 0):
                if (waypoint1[1] > y_pos and distanceAB_2D < 2.5):
                    y_acc = 0
                    y_vel = 0.1
                    print("Sono yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
                if (waypoint1[1] <= y_pos):
                    y_acc = 0
                    y_vel = 0
            if (waypoint1[0] == 0):
                y_acc = 0
                y_vel = 0
            if (waypoint1[1] == 0):
                y_acc = 0
                y_vel = 0

            if (distanceAB_2D < 2.5):
                t = t - 0.1

            '''if (distanceAB_2D <= 6 and SALVO_PROPORZIONE[0] == 0 and SALVO_PROPORZIONE[1] == 0 and t< T-0.2):
                t = t-0.1
                print("salvoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")'''



            if (distanceAB_2D < 0.1):
                t = T

            '''if(acc_ms > acc_max and Salvo_prop_acc == True): #Salvo istantanea delle proporzioni x_acc, y_acc quando l'accellerazione è magiore di (acc_max)
                salvo_acc_x = x_acc
                salvo_acc_y = y_acc
                Salvo_prop_acc = False'''

            if (acc_ms > acc_max and Salvo_prop_acc == True): #Mantengo l'acc sotto i 3 m/s facendo una proporzione tra x_acc e y_acc
                Kacc = acc_max / (x_acc + y_acc)
                salvo_acc_x = x_acc * Kacc
                salvo_acc_y = y_acc * Kacc
                Salvo_prop_acc = False


            if (Salvo_prop_acc == False and vel_ms < cruise_speed_ms and (distanceAB_2D > dist_percorsa2D)):
                x_acc = salvo_acc_x
                y_acc = salvo_acc_y
            if (z_pos == waypoint1[2]): #Mantengo il drone su una z fissa
                z_vel = 0
                z_acc = 0



            #print("thrust", thrust)
            print("acc_ms", acc_ms)



            print("SALVO_TEMPO:", SALVO_TEMPO, "SALVO_DISTANZA_FATTA:", SALVO_DISTANZA_FATTA, "t_dec:", t_dec, "dist_percorsa2D:", dist_percorsa2D)
            if (distanceAB_3D < 10):
                print("Goal a:", distanceAB_3D, "Dist percorsa3D:", dist_percorsa3D, "Vel_x:", x_vel, "Acc_x:", x_acc, "Time:", t, )



            print(vel_ms,"vel ms")
            print("X_vel:", "{:.2f}".format(x_vel), "\nX_acc:", "{:.2f}".format(x_acc))
            print("Y_vel:", "{:.2f}".format(y_vel), "\nY_acc:", "{:.2f}".format(y_acc))
            print("Z_vel:", "{:.2f}".format(z_vel), "\nZ_acc:", "{:.2f}".format(z_acc))
            print("X_dess:", des_x_acc, "\nY_dess:", des_y_acc)
            print("SALVO_PROPORZIONE", SALVO_PROPORZIONE[0],SALVO_PROPORZIONE[1])
            print("Distanza2D:", distanceAB_2D)
            print("dist_percorsa2D:", dist_percorsa2D)
            # ------------------------------------------------------------------------------------------------------------------#

            '''if (5 < t <25):
                x_acc = 0
                y_acc = 0
                if distanceAB > 2:
                    t=t-0.1
            else:
                x_acc = acc[0]
                y_acc = acc[1]'''


            '''if (t < 5 or t > 25):
                x_acc = acc[0]
                y_acc = acc[1]
            else:
                x_acc = 0
                y_acc = 0'''



            #a_max = 3.0
            #accc= cruise_speed_ms/a_max
            #dec = (x_acc-0)/(T-t)
            #print(dec)
            #t_a = x_vel/x_acc #tempo di arresto


            x_vel += x_acc * dt
            y_vel += y_acc * dt
            z_vel += z_acc * dt

            x_pos += x_vel * dt  # Velocità * Tempo
            y_pos += y_vel * dt
            z_pos += z_vel * dt

            #pos_ms = math.sqrt(x_pos ** 2 + y_pos ** 2 + z_pos ** 2)
            #vel2 = pos_ms * dt
            #print(vel2,"oooooooooooooooooooooo")


            UAV.update_pose(x_pos, y_pos, z_pos, roll, pitch, yaw)

            t += dt
            #print("Velocità x,y,z: ", x_vel, y_vel, z_vel, )
            #print("Acceleration: ", acc)
            #print("Spinta-thrust: ", thrust)
            #print("roll, pitch, yaw: ", roll, pitch, yaw )
            #print("Velocity:", vel_ms, "m/s, Vel X: ", x_vel, "m/s")
            #print("Velocity Km/h: ", vel_kmh)
            print("Acceleration:", "{:.2f}".format(acc_ms), " m/s^2, des_x_acc: ", des_x_acc, " m/s^2, Time: ", "{:.2f}".format(t),"s")
            #print("X_acc:", "{:.2f}".format(x_acc), "m/s^2,Y_acc:", "{:.2f}".format(y_acc),"m/s^2")
            #print("Acceleration km/h: ", acc_kmh)
            #print(des_y_vel, "sacsdcsdvsdvds")
            #print("des_vel", des_x_vel, des_y_vel, des_z_vel)
            #print(roll_torque, "roll_torque")
            #print(pitch_torque, "pitch_torque")
            #print(yaw_torque, "yaw_torque")
            #print(roll_vel, "roll_vel")
            #print(pitch_vel, "pitch_vel")
            #print(yaw_vel, "yaw_vel")
            #print("Time:", T, "s, Space:", space_m, "m, T_s:", T_s, "s, T_s2:", T_s2,"s")
            #print("goal distance: ", distanceAB_3D, "Dist percorsa:", dist_percorsa3D)

        print("[WAYPOINT TIME LIMIT REACHED]")
        if(distanceAB_2D != 0 ):
            print("[GOAL MISSED:","{:.2f}".format(distanceAB_2D), "m missing]")
        t = 0
        i = (i + 1) % 4
        irun += 1
        if irun >= n_run:
            break

    print("Done")



def distance(x0, x1, dimensions):
  delta = np.abs(x0 - x1)
  delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
  return np.sqrt((delta ** 2).sum(axis=-1))

def calculate_position(c, t):
    """
    Calculates a position given a set of quintic coefficients and a time.

    Args
        c: List of coefficients generated by a quintic polynomial
            trajectory generator.
        t: Time at which to calculate the position

    Returns
        Position
    """
#divido il perscorso in tanti piccoli punti

    return c[0] * t**5 + c[1] * t**4 + c[2] * t**3 + c[3] * t**2 + c[4] * t + c[5]


def calculate_velocity(c, t):
    """
    Calculates a velocity given a set of quintic coefficients and a time.

    Args
        c: List of coefficients generated by a quintic polynomial 
            trajectory generator.
        t: Time at which to calculate the velocity

    Returns
        Velocity
    """
    return 5 * c[0] * t**4 + 4 * c[1] * t**3 + 3 * c[2] * t**2 + 2 * c[3] * t + c[4]


def calculate_acceleration(c, t):
    """
    Calculates an acceleration given a set of quintic coefficients and a time.

    Args
        c: List of coefficients generated by a quintic polynomial 
            trajectory generator.
        t: Time at which to calculate the acceleration

    Returns
        Acceleration
    """
    return 20 * c[0] * t**3 + 12 * c[1] * t**2 + 6 * c[2] * t + 2 * c[3]

def rotation_matrix(roll, pitch, yaw):
    """
    Calculates the ZYX rotation matrix.

    Args
        Roll: Angular position about the x-axis in radians.
        Pitch: Angular position about the y-axis in radians.
        Yaw: Angular position about the z-axis in radians.
    Returns
        3x3 rotation matrix as NumPy array
    """
    return np.array(
        [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll)],
         [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) *
          sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll)],
         [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw)]
         ])



def main():

    sys.stdout = Logger()
    print("\n\n\n"+"".join( ["#"]*50) )

    print("User:",format(getenv("USER")))
    print("Date:",format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))

    if sys.platform.startswith('linux'):
        print("User:", format(getenv("USER")))      #For Linux
    if sys.platform.startswith('win32'):
        print("User:",format(getenv("USERNAME")))   #For Windows
    print("OS:", sys.platform)
    print("Date:",format(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]))

    print("".join( ["#"]*50)+"\n\n\n")
    """
    Calculates the x, y, z coefficients for the four segments 
    of the trajectory
    """
    x_coeffs = [[], [], [], []]
    y_coeffs = [[], [], [], []]
    z_coeffs = [[], [], [], []]

    if SEED!=None: seed(SEED)
    #values = randint(0, 8, 3)
    #values = space_m
    values = 15
    #waypoints = [[-values[0], -values[1], values[2]], [values[0], -values[1], values[2]], [values[0], values[1], values[2]], [-values[0], values[1], values[2]]]
    waypoints = [start_pos, waypoint1, [0, 15, 5], [-values, values, values]]
    print("Waypoints: ", waypoints)

    for i in range(4):
        traj = TrajectoryGenerator(waypoints[i], waypoints[(i + 1) % 4], T)
        traj.solve()
        x_coeffs[i] = traj.x_c
        y_coeffs[i] = traj.y_c
        z_coeffs[i] = traj.z_c

    quad_sim(x_coeffs, y_coeffs, z_coeffs)



if __name__ == "__main__":
    main()


