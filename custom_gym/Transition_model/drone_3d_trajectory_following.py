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
T = 5                             #Time (seconds for waypoint - waypoint movement)
cruise_speed_ms = cruise_speed_kmh/3.6  #Cruise speed m/s

# Proportional coefficients
Kp_x = 1
Kp_y = 1
Kp_z = 1
Kp_roll = 25
Kp_pitch = 25
Kp_yaw = 25

# Derivative coefficients
Kd_x = 10
Kd_y = 10
Kd_z = 1


def quad_sim(x_c, y_c, z_c):
    
    """
    Calculates the necessary thrust and torques for the quadrotor to
    follow the trajectory described by the sets of coefficients
    x_c, y_c, and z_c. ##Spinta e Coppia
    """

    x_pos = -5
    y_pos = -5
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

    UAV = Quadrotor(id = "uav", x=x_pos, y=y_pos, z=z_pos, roll=roll,
                  pitch=pitch, yaw=yaw, size=1, show_animation=show_animation)

    i = 0
    n_run = 4 #Numero di Round (quanti waypoints vuoi vedere)
    irun = 0

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
            #print(des_x_acc," des_x_acc")
            #print(des_z_vel, "des_z_vel") sempre 0
            #print("time: ", t, i)
            print(x_c[i],t)
            #print(y_c[i], t)

            acc_des_xyz = math.sqrt(des_x_acc ** 2 + des_y_acc ** 2 + des_z_acc ** 2)
            #print("Accelerazione desiderata: ", acc_des_xyz)
            



            vel_ms = math.sqrt(x_vel**2 + y_vel**2 + z_vel**2)
            acc_ms = math.sqrt(x_acc ** 2 + y_acc ** 2 + z_acc ** 2)
            vel_kmh =  vel_ms*(60**2)/1000
            acc_kmh =  acc_ms*(60**2)/1000

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


            x_acc = acc[0]
            y_acc = acc[1]
            z_acc = acc[2]


            x_vel += x_acc * dt  # Accelerazione * Tempo
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
            #print("Velocity m/s: ", vel_ms)
            #print("Velocity Km/h: ", vel_kmh)
            print("Acceleration m/s: ", acc_ms)
            #print("Acceleration km/h: ", acc_kmh)
            #print(des_y_vel, "sacsdcsdvsdvds")
            #print("des_vel", des_x_vel, des_y_vel, des_z_vel)
            #print(roll_torque, "roll_torque")
            #print(pitch_torque, "pitch_torque")
            #print(yaw_torque, "yaw_torque")
            #print(roll_vel, "roll_vel")
            #print(pitch_vel, "pitch_vel")
            #print(yaw_vel, "yaw_vel")
       
        t = 0
        i = (i + 1) % 4
        irun += 1
        if irun >= n_run:
            break

    print("Done")


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
    """
    Calculates the x, y, z coefficients for the four segments 
    of the trajectory
    """
    x_coeffs = [[], [], [], []]
    y_coeffs = [[], [], [], []]
    z_coeffs = [[], [], [], []]

    if SEED!=None: seed(SEED)
    #values = randint(0, 8, 3)
    values = 5
    #waypoints = [[-values[0], -values[1], values[2]], [values[0], -values[1], values[2]], [values[0], values[1], values[2]], [-values[0], values[1], values[2]]]
    waypoints = [[-values, -values, 5], [values, -values, 5], [values, values, 5], [-values, values, 5]]
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
