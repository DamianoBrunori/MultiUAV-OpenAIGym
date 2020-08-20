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
# Class to redirect stdout to file logfile.log
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "a")

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

info1 = "\n\n_______________________________________ENVIRONMENT AND DRONE INFO: _______________________________________\n"
info.append(info1)
info2 = "\nID: " + str(id)
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
info22 = "\nCRUISE SPEED: " + str(cruise_speed_kmh) + " Km/h"
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
info21 = "\nTIME TO REACH THE GOAL: " + str(T) + " s"
info.append(info21)
info23 = "\n__________________________________________________________________________________________________________________\n\n"
info.append(info23)

cases_directory = "Salvo"
if not isdir(cases_directory): mkdir(cases_directory)
file = open(join(cases_directory, "env_and_train_info.txt"), "w")

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
T = 60                                  #Time (seconds for waypoint - waypoint movement)
cruise_speed_ms = cruise_speed_kmh/3.6   #Cruise speed m/s
T_s = T/5                                #Tempo fino a quando avviene un' accelerazione
T_s2 = T - T_s                            #Tempo dopo il quale avviene una decellerazione

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

waypoint1= [200, 0, 5]
'''INITIAL_X_POS = 0
INITIAL_Y_POS = 0
INITIAL_Z_POS = 5
INITIAL_POS = [INITIAL_X_POS,INITIAL_Y_POS,INITIAL_Z_POS]'''

def quad_sim(x_c, y_c, z_c):
    
    """
    Calculates the necessary thrust and torques for the quadrotor to
    follow the trajectory described by the sets of coefficients
    x_c, y_c, and z_c. ##Spinta e Coppia
    """

    x_pos = 0
    y_pos = 0
    z_pos = 5
   
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
    i = 0
    n_run = 4 #Numero di Round (quanti waypoints vuoi vedere)
    irun = 0

    ok = 0
    SALVO_TEMPO = 0
    t_dec = 0
    SALVO_DISTANZA_FATTA = 0
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
            dimension3D=np.array([300.0,400.0,500.0])
            distanceAB = distance(PosizioneAttuale, goal, dimension3D) #Distanza drone goal
            print(x_pos, y_pos, z_pos)

#------------------------------------------------------------------------------------------------------------------#

            if (x_vel > 5 and distanceAB > SALVO_DISTANZA_FATTA):
                #SALVO_TEMPO = t
                #SALVO_DISTANZA_FATTA = waypoint1[0] - distanceAB
                ciao = True
                if (ciao == True and ok == 0):
                    SALVO_TEMPO = t - 0.1
                    SALVO_DISTANZA_FATTA = waypoint1[0] - distanceAB
                    if (SALVO_TEMPO > 0 and SALVO_DISTANZA_FATTA > 0):
                        t_dec = T - SALVO_TEMPO
                        ok = 1
                t = 0  # non c'è accellerazione


            if (distanceAB < SALVO_DISTANZA_FATTA):
                if (t < 1):
                    t = t_dec

            '''if (1 < distanceAB < 15):
                x_acc = 0
                t=t-0.1'''



            print("SALVO_TEMPO:", SALVO_TEMPO, "SALVO_DISTANZA_FATTA:", SALVO_DISTANZA_FATTA, "t_dec:", t_dec)
#------------------------------------------------------------------------------------------------------------------#



            '''if (x_vel>5 and distanceAB > 30):
                ok = t
                t_dec = T-ok
            elif (distanceAB < 30):
                ciao = True
                t = t_dec
            print (ok,"ok")
            print(t2, "t2")'''




            '''if(t<5 or t >10):
                des_x_acc = calculate_acceleration(x_c[i], t)
                des_y_acc = calculate_acceleration(y_c[i], t)
            else:
                des_x_acc = 0
                des_y_acc = 0
            des_z_acc = calculate_acceleration(z_c[i], t)'''

            #print(des_x_acc," des_x_acc")
            #print(des_z_vel, "des_z_vel") sempre 0
            #print("time: ", t, i)
            # print([v for v in zip((x_c[i]),labels) ],t)
            # print(x_c[i], t)
            #print(y_c[i], t)

            acc_des_xyz = math.sqrt(des_x_acc ** 2 + des_y_acc ** 2 + des_z_acc ** 2)
            #print("Accelerazione desiderata: ", acc_des_xyz)
            



            vel_ms = math.sqrt(x_vel**2 + y_vel**2 + z_vel**2)
            acc_ms = math.sqrt(x_acc ** 2 + y_acc ** 2 + z_acc ** 2)
            
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



            #des_vel_ms = math.sqrt(des_x_vel ** 2 + des_y_vel ** 2 + des_z_vel ** 2)
            #print("Des_vel_ms: ", des_vel_ms)
            Accc = t
            '''if (x_vel > +0.5):
                x_acc = 0
            else:'''
            #x_acc = acc[0]
            y_acc = acc[1]
            print("x:vel", "{:.2f}".format(x_vel),"(m/s)")
            print("y:vel", "{:.2f}".format(y_vel),"(m/s)")
            print("Acc[0]", "{:.2f}".format(acc[0]),"(m/s^2")
            #print("des_y_acc", "{:.2f}".format(des_y_acc))
            x_vel1 = x_vel
            x_acc1 = acc[0]

            dist_dec = waypoint1[2]-5
            dist_acc = 4

            '''if (dist_acc < distanceAB < 10 ):
                x_acc = 0
                y_acc = 0
            else:
                x_acc = acc[0]
                y_acc = acc[1]

            if distanceAB > 4:
                t = 14
            else:
                t = 29'''

            '''V_max= 5.5
            A_max = 2
            T_s= V_max/A_max
            #Time = 15
            L = 2 / 3 * T * V_max  # 2/3*time total * velocità massima->L = distanza tot
            Tot= (L*A_max + V_max**2)/(A_max*V_max)
            T_s2 = Tot - T_s
            print(T_s, "T_s")   #2.75
            print(Tot, "Tot")   #12.75
            print(T_s2, "T_s2") #10
            print(L, "L")       #55'''

            '''if (x_vel < 0.2):
                t1 = t1+0.1
                print(t1, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")


            if (x_vel > 0.2 and distanceAB > 5):
                x_acc = 0
                y_acc = 0
            else:
                x_acc = acc[0]
                y_acc = acc[1]'''

            x_acc = acc[0]
            y_acc = acc[1]

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

            z_acc = acc[2]

            #a_max = 3.0
            #accc= cruise_speed_ms/a_max
            #dec = (x_acc-0)/(T-t)
            #print(dec)
            #t_a = x_vel/x_acc #tempo di arresto

            '''if (t > 30):
                t=t-1

            if (distanceAB <= 0.9 and x_vel > 0.01):
                t=30
                x_acc = -0.01
            elif(distanceAB <= 0.9 and x_vel < 0.1):
                x_acc = 0

            if (distanceAB < 0.1):
                t=60    
            # NOTE: for each t in in [0.1,0] set to 60

            if (x_vel > 0.1 and distanceAB >= 0.9):
                x_acc = 0
            else:   # x_vel <= 0.1 or distanceAB < 0.9
                x_vel += x_acc * dt'''  # Accelerazione * Tempo




            '''if (y_vel > 0.1 and distanceAB >= 0.9):
                y_acc = 0
            else:
                x_vel += x_acc * dt'''  # Accelerazione * Tempo
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
            print("Velocity:", vel_ms, "m/s, Vel X: ", x_vel, "m/s")
            #print("Velocity Km/h: ", vel_kmh)
            print("Acceleration:", "{:.2f}".format(acc_ms), " m/s^2, des_x_acc: ", des_x_acc, " m/s^2, Time: ", "{:.2f}".format(t),"s")
            print("X_acc:", "{:.2f}".format(x_acc), "m/s^2,Y_acc:", "{:.2f}".format(y_acc),"m/s^2")
            #print("Acceleration km/h: ", acc_kmh)
            #print(des_y_vel, "sacsdcsdvsdvds")
            #print("des_vel", des_x_vel, des_y_vel, des_z_vel)
            #print(roll_torque, "roll_torque")
            #print(pitch_torque, "pitch_torque")
            #print(yaw_torque, "yaw_torque")
            #print(roll_vel, "roll_vel")
            #print(pitch_vel, "pitch_vel")
            #print(yaw_vel, "yaw_vel")
            print("Time:", T, "s, Space:", space_m, "m, T_s:", T_s, "s, T_s2:", T_s2,"s")
            print("goal distance: ", distanceAB)

        print("[WAYPOINT TIME LIMIT REACHED]")
        if(distanceAB != 0 ):
            print("[GOAL MISSED:","{:.2f}".format(distanceAB), "m missing]")
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
    print("\n\n\n"+ "".join( ["#"]*50))
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
    waypoints = [[0, 0, 5], waypoint1, [0, 15, 5], [-values, values, values]]
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
