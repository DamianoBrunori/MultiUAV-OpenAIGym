from math import cos, sin
from numpy.random import seed
from numpy.random import randint
from os import mkdir
from os.path import join, isdir
import numpy as np
import math
from Quadrotor import Quadrotor
from TrajectoryGenerator import TrajectoryGenerator
from my_utils import *



T = 0
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
info20 = "\nDISTANCE TO REACH THE GOAL: " + str(distance_space_m) + " m"
info.append(info20)
info21 = "\nTIME TO REACH THE GOAL: " + str(T) + " s"
info.append(info21)
info25 = "\nSTART COORDINATES: " + "X:" + str(start_pos[0]) + " Y:" + str(start_pos[1]) + " Z:" + str(start_pos[2])
info.append(info25)
info26 = "\nDESTINATION COORDINATES(1): " + "X:" + str(waypoint_dest[0]) + " Y:" + str(waypoint_dest[1]) + " Z:" + str(waypoint_dest[2])
info.append(info26)
if(add_waypoint):
    info27 = "\nDESTINATION COORDINATES(2): " + "X:" + str(add_waypoint[0]) + " Y:" + str(add_waypoint[1]) + " Z:" + str(add_waypoint[2])
    info.append(info27)
info28 = "\nAltitude: " + str(altitude) + " m"
info.append(info28)

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

g = 9.81        #Gravity (m/s^-2)
# m = 0.2        #Massa (Kg)
Ixx = 1
Iyy = 1
Izz = 1
cruise_speed_kmh = cruise_speed_ms/3.6   #Cruise speed km/h

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

salita = [start_pos[0], start_pos[1]+0.001, altitude]
discesa = [waypoint_dest[0], waypoint_dest[1]+0.001, altitude]

acc_max = 8

# waypoints = [start_pos, salita,add_waypoint, waypoint_dest, discesa]
# waypoints = [user_waypoints[0], salita] + \
#     [x for x in user_waypoints[1:-2] ] +[ discesa ,waypoint_dest]
# print(user_waypoints)
if(user_waypoints):
    waypoints = [start_pos,salita] + user_waypoints + [discesa,waypoint_dest]
else:
    waypoints = [start_pos,salita,discesa,waypoint_dest]

num_waypoints = len(waypoints)

##################################  Senza-file-configs  ################################
#waypoint1 = [1100, 0, 5]
#start_x = 0
#start_y = 0
#start_z = 5
#start_pos = np.array([start_x, start_y, start_z])
####################################################################################

#------------------------------------------------------Gestione-del-Tempo----------------------------------------------------------#
diff_metri = 0
incrementoT = 0
dist_X = 0
dist_Y = 0
dist_Z = 0



waypoints_distances = []
for i,d in enumerate(waypoints):
    if(i+1 < num_waypoints):
        waypoints_distances.append( distance_AB_3D(waypoints[i],waypoints[i+1]) )


def log_increment_flight_time(scalare_waypoints):
    if (1) <= scalare_waypoints < km5:
        base = np.exp(1)
        newT = 21.6508 * ( np.log(scalare_waypoints)/ np.log(base))  -98.7986
    if (km5) <= scalare_waypoints <= km10:
        base = np.exp(1)
        newT = 36.9049 * ( np.log(scalare_waypoints) / np.log(base)) -225.8359
    if (km10 < scalare_waypoints <= km47):
        base = np.exp(1)
        newT = 52.3241 * ( np.log(scalare_waypoints) / np.log(base)) -369.2763
    return newT

#-----------------------------------------------------------------------------------------------------------------------#
def quad_sim(x_c, y_c, z_c):

    """
    Calculates the necessary thrust and torques for the quadrotor to
    follow the trajectory described by the sets of coefficients
    x_c, y_c, and z_c. ##Spinta e Coppia
    """

    SALVO_PROPORZIONE = [0, 0]
    x_pos = start_pos[0]
    y_pos = start_pos[1]
    z_pos = start_pos[2]

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
    d = 0

    UAV = Quadrotor(id = "uav", x=x_pos, y=y_pos, z=z_pos, roll=roll,
                  pitch=pitch, yaw=yaw, size=1, show_animation=show_animation)

    fase_dec_x = False
    fase_dec_y = False
    Salvo_prop_acc = True
    #Salvo_prop_dec = False
    ciao = False

    X_limit = False
    Y_limit = False

    i = 0
    n_run = 3 #Numero di Round (quanti waypoints vuoi vedere)
    irun = 0

    ok = True
    SALVO_TEMPO = 0
    t_dec = 0
    SALVO_DISTANZA_FATTA = 0
    salvo_acc_x = 0
    salvo_acc_y = 0
    salvo_acc_z = 0

    decollo = True

    print("[waypoints_distances]",waypoints_distances)
        
    while True:

        scalare_waypoints = waypoints_distances[i % len(waypoints_distances)]
        
        T = log_increment_flight_time(scalare_waypoints)
        
        while t <= T:
            print()
            #DIVIDO LA TRAIETTORIA IN PEZZETTI si fissano degli obiettivi molto vicini (interpolazione)
            # des_x_pos = calculate_position(x_c[i], t)
            # des_y_pos = calculate_position(y_c[i], t)
            des_z_pos = calculate_position(z_c[i], t)


            des_x_vel = calculate_velocity(x_c[i], t)       #desiderata
            des_y_vel = calculate_velocity(y_c[i], t)
            des_z_vel = calculate_velocity(z_c[i], t)


            des_x_acc = calculate_acceleration(x_c[i], t)
            des_y_acc = calculate_acceleration(y_c[i], t)
            des_z_acc = calculate_acceleration(z_c[i], t)


            PosizioneAttuale = np.array([x_pos, y_pos, z_pos])
            goal = np.array(waypoints[(i+1)%num_waypoints])
            dimension3D=np.array([30000.0,40000.0,50000.0])
            distanceAB_3D = distance(PosizioneAttuale, goal, dimension3D) #Distanza drone goal
            print("POSITION (x,y,z)(m):","{:.2f}".format(x_pos), "{:.2f}".format(y_pos), "{:.2f}".format(z_pos))

            # distanceAB_2D = math.sqrt((waypoint1[0] - PosizioneAttuale[0]) ** 2 + (waypoint1[1] - PosizioneAttuale[1]) ** 2) # Distanza drone goal
            start = waypoints[i]
            next_goal = waypoints[(i+1)%num_waypoints]

            dist_X = math.sqrt((next_goal[0] - PosizioneAttuale[0]) ** 2)
            dist_Y = math.sqrt((next_goal[1] - PosizioneAttuale[1]) ** 2)
            dist_Z = math.sqrt((next_goal[2] - PosizioneAttuale[2]) ** 2)
            distanceAB_2D = distance_AB_2D(PosizioneAttuale,next_goal)
            dist_percorsa2D  = distance_AB_2D(start,PosizioneAttuale)
            dist_percorsa3D = distance(start, PosizioneAttuale, dimension3D) #Distanza percorsa


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



            
            if(decollo):
                z_acc = acc[2]
                y_acc = 0
                x_acc = 0
                if (dist_Z < 0.5):
                    decollo = False
                    t = T
            else: #(Not in decollo)
                if (scenario_Time ):
                    x_acc = acc[0]
                    y_acc = acc[1]
                    #z_acc = acc[2]
                    z_acc = 0


                    if (vel_ms < 1 and dist_percorsa2D > distanceAB_2D ):
                        x_acc = 0
                        y_acc = 0
                        t = t - 0.1
                        d = d + 0.1
                    if (distanceAB_2D < 20):
                        print("sono quiiiii")
                        if (next_goal[0] < 0):
                            if (next_goal[0] < x_pos and dist_X < 1):
                                x_acc = 0
                                x_vel = -0.1
                                print("------------------------------    x neg    ------------------------------")
                            if (next_goal[0] >= x_pos):
                                x_acc = 0
                                x_vel = 0
                                X_limit = True
                                print("X Goal - La X non cambia (neg)")
                        if (next_goal[0] > 0):
                            if (next_goal[0] > x_pos and dist_X < 1):
                                x_acc = 0
                                x_vel = 0.1
                                print("------------------------------   x (pos)   ------------------------------")
                            if (next_goal[0] <= x_pos):
                                x_acc = 0
                                x_vel = 0
                                X_limit = True
                                print("X Goal - La X non cambia (pos)")
                        if (next_goal[1] < 0):
                            if (next_goal[1] < y_pos and dist_Y < 1):
                                y_acc = 0
                                y_vel = -0.1
                                print("------------------------------     y neg   ------------------------------")
                            if (next_goal[1] >= y_pos):
                                y_acc = 0
                                y_vel = 0
                                Y_limit = True
                                print("Y Goal - La Y non cambia (neg)")
                        if (next_goal[1] > 0):
                            if (next_goal[1] > y_pos and dist_Y < 1):
                                y_acc = 0
                                y_vel = 0.1
                                print("------------------------------    y (pos)   ------------------------------")
                            if (next_goal[1] <= y_pos):
                                y_acc = 0
                                y_vel = 0
                                Y_limit = True
                                print("Y Goal - La Y non cambia (pos)")
                        if (next_goal[0] == 0):
                            if (next_goal[0] == x_pos and dist_X < 1):
                                x_acc = 0
                                x_vel = 0
                                X_limit = True
                                print("X = 0 La X non cambia")
                        if (next_goal[1] == 0):
                            if (next_goal[1] == y_pos and dist_Y < 1):
                                y_acc = 0
                                y_vel = 0
                                Y_limit = True
                                print("Y = 0 La X non cambia")
                        if (X_limit == True and Y_limit == True):
                            t = T
                            print("-----------------------WAYPOINT OK!-----------------------")

                        if (distanceAB_2D < 0.1):
                            t = T

                # 
                # 
                # 
                else: #( scenario_time is False)
                    if (vel_ms >= cruise_speed_ms and distanceAB_2D > SALVO_DISTANZA_FATTA ):
                        if (ok == True):
                            SALVO_TEMPO = t - 0.1
                            SALVO_DISTANZA_FATTA = dist_percorsa2D  #Salvo distanza percorsa (in quel momento)
                            if (SALVO_TEMPO > 0 and SALVO_DISTANZA_FATTA > 0):
                                t_dec = T - SALVO_TEMPO  # Salvo t_dec (tempo di decellerazione) T - (Tempo impiegato a raggiungere la velocità di crociera)
                                ok = False
                        x_acc = 0
                        y_acc = 0
                        t = -0.1
                    else:
                        x_acc = acc[0]
                        y_acc = acc[1]
                    z_acc = acc[2]

                    if (distanceAB_2D < SALVO_DISTANZA_FATTA):
                        if (t < 1):
                            t = t_dec

                    if (1<= distanceAB_2D <= 4 and t_dec != 0):  # Se la diztanza dal goal e minore di 4 e t_dec diverso da 0 (quindi non si è raggiunta una velocità max se t_dec = 0)
                        x_acc = 0
                        y_acc = 0
                        print("auuuuaiassjcdsjcindvdsvsdvdsvdsvdsvdsvdsvdsvdsvdds")
                        '''vel_max = 1
                        Kvel = vel_max / math.sqrt(x_vel**2 + y_vel**2) #Proporzione
                        x_vel = x_vel * Kvel
                        y_vel = y_vel * Kvel'''
                        t = 0

                    if (next_goal[0] < 0):
                        if (next_goal[0] < x_pos and dist_X < 1):
                            x_acc = 0
                            x_vel = -0.1
                            print("------------------------------    x neg    ------------------------------")
                        if (next_goal[0] >= x_pos):
                            x_acc = 0
                            x_vel = 0
                            X_limit = True
                            print("X Goal - La X non cambia (neg)")
                    if (next_goal[0] > 0):
                        if (next_goal[0] > x_pos and dist_X < 1):
                            x_acc = 0
                            x_vel = 0.1
                            print("------------------------------   x (pos)   ------------------------------")
                        if (next_goal[0] <= x_pos):
                            x_acc = 0
                            x_vel = 0
                            X_limit = True
                            print("X Goal - La X non cambia (pos)")
                    if (next_goal[1] < 0):
                        if (next_goal[1] < y_pos and dist_Y < 1):
                            y_acc = 0
                            y_vel = -0.1
                            print("------------------------------     y neg   ------------------------------")
                        if (next_goal[1] >= y_pos):
                            y_acc = 0
                            y_vel = 0
                            Y_limit = True
                            print("Y Goal - La Y non cambia (neg)")
                    if (next_goal[1] > 0):
                        if (next_goal[1] > y_pos and dist_Y < 1):
                            y_acc = 0
                            y_vel = 0.1
                            print("------------------------------    y (pos)   ------------------------------")
                        if (next_goal[1] <= y_pos):
                            y_acc = 0
                            y_vel = 0
                            Y_limit = True
                            print("Y Goal - La Y non cambia (pos)")
                    if (next_goal[0] == 0 ):
                        if (next_goal[0] == x_pos and dist_X < 1):
                            x_acc = 0
                            x_vel = 0
                            X_limit = True
                            print("X = 0 La X non cambia")
                    if (next_goal[1] == 0 ):
                        if (next_goal[1] == y_pos and dist_Y < 1):
                            y_acc = 0
                            y_vel = 0
                            Y_limit = True
                            print("Y = 0 La X non cambia")
                    if (X_limit == True and Y_limit == True):
                        t = T
                        print("-----------------------WAYPOINT OK!-----------------------")


                    if (acc_ms > acc_max and Salvo_prop_acc == True): #Mantengo l'acc sotto i 3 m/s facendo una proporzione tra x_acc e y_acc
                        Kacc = acc_max / math.sqrt(x_acc ** 2 + y_acc ** 2) #forse math.sqrt(x_acc ** 2 + y_acc ** 2)
                        salvo_acc_x = x_acc * Kacc #PUO'CREARE PROBLEMI PER FAR RITORNARE IL DRONE AL PUNTO DI PARTENZA
                        salvo_acc_y = y_acc * Kacc
                        Salvo_prop_acc = False


                    if (Salvo_prop_acc == False and vel_ms < cruise_speed_ms and (distanceAB_2D > dist_percorsa2D)):
                        x_acc = salvo_acc_x
                        y_acc = salvo_acc_y
                    if (z_pos == altitude): #Mantengo il drone su una z fissa (QUI VA GENERALIZZATO)
                        z_vel = 0
                        z_acc = 0


            #print("thrust", thrust)
            # print("acc_ms", acc_ms)

            print("Dist (x,y,z):","{:.2f}".format(dist_X), "{:.2f}".format(dist_Y), "{:.2f}".format(dist_Z) )

            print("Da 0 a", cruise_speed_ms,"m/s:", SALVO_TEMPO,"s", "SALVO_DISTANZA_FATTA:", SALVO_DISTANZA_FATTA, "t_dec:", t_dec)
            '''if (distanceAB_3D < 10):
                print("Goal a:", distanceAB_2D, "Dist percorsa3D:", dist_percorsa2D, "Vel_x:", x_vel, "Acc_x:", x_acc, "Time:", t, )'''

            print("scalare_waypoints", scalare_waypoints, "Time:", T, "incremento_T: ", incrementoT)

            print("Velocity (m/s):", "{:.2f}".format(vel_ms))
            print("X_vel (m/s):", "{:.2f}".format(x_vel), "\tX_acc (m/s^2):", "{:.2f}".format(x_acc))
            print("Y_vel (m/s):", "{:.2f}".format(y_vel), "\tY_acc (m/s^2):", "{:.2f}".format(y_acc))
            print("Z_vel (m/s):", "{:.2f}".format(z_vel), "\tZ_acc (m/s^2):", "{:.2f}".format(z_acc))
            # print("X_dess:", des_x_acc, "\tY_dess:", des_y_acc)
            #print("SALVO_PROPORZIONE", SALVO_PROPORZIONE[0],SALVO_PROPORZIONE[1])
            if (decollo == False):
                print("Distanza2D:", "{:.3f}".format(distanceAB_2D), "m")
                print("dist_percorsa2D:", "{:.3f}".format(dist_percorsa2D), "m")
            else:
                print("Dist_altitudine_desiderata:", "{:.2f}".format(dist_Z), "m")
                print("Altitudine:", "{:.2f}".format(z_pos), "m")
            # ------------------------------------------------------------------------------------------------------------------#


            x_vel += x_acc * dt
            y_vel += y_acc * dt
            z_vel += z_acc * dt

            x_pos += x_vel * dt  # Velocità * Tempo
            y_pos += y_vel * dt
            z_pos += z_vel * dt




            UAV.update_pose(x_pos, y_pos, z_pos, roll, pitch, yaw)

            t += dt
            #print("Velocità x,y,z: ", x_vel, y_vel, z_vel, )
            #print("Acceleration: ", acc)
            #print("Spinta-thrust: ", thrust)
            #print("roll, pitch, yaw: ", roll, pitch, yaw )
            #print("Velocity:", vel_ms, "m/s, Vel X: ", x_vel, "m/s")
            #print("Velocity Km/h: ", vel_kmh)
            print("Acceleration:", "{:.2f}".format(acc_ms), " m/s^2", "Time: ", "{:.2f}".format(t),"s")
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
            if (decollo):
                print("[GOAL MISSED:", "{:.2f}".format(dist_Z), "m missing]", "Time:", T)
            else:
                print("[GOAL MISSED:","{:.2f}".format(distanceAB_2D), "m missing]", "Time:", T)



        ok = True
        Salvo_prop_acc = True
        X_limit = False
        Y_limit = False
        salvo_acc_x = 0
        salvo_acc_y = 0
        SALVO_DISTANZA_FATTA = 0
        SALVO_TEMPO = 0
        t_dec = 0

        SALVO_PROPORZIONE = [0, 0]

        roll = 0
        pitch = 0
        yaw = 0

        x_acc = 0
        y_acc = 0
        z_acc = 0

        x_vel = 0
        y_vel = 0
        z_vel = 0

        roll_vel = 0
        pitch_vel = 0
        yaw_vel = 0

        des_yaw = 0  # mantenere orientazione 0 quindi rimane fisso il  drone non gira su se stesso

        dist_X = 0
        dist_Y = 0
        dist_Z = 0

        t = 0
        d = 0
        
        x_acc = 0
        y_acc = 0
        z_acc = 0

        x_vel = 0
        y_vel = 0
        z_vel = 0

        dist_X = 0
        dist_Y = 0
        dist_Z = 0

        
        i = (i + 1) % num_waypoints
        if (i==2): #((waypoints[i-1][2] != waypoints[i][2])):
            decollo = True

        irun += 1
        if irun >= n_run:
            break

    print(print("\n\n\n"+"".join( ["#"]*50) ))
    print("MISSION ACCOMPLISHED!")
    print(print("\n"+"".join(["#"] * 50)))
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
    
    x_coeffs = [[] for i in range(num_waypoints)]
    y_coeffs = [[] for i in range(num_waypoints)]
    z_coeffs = [[] for i in range(num_waypoints)]

    if SEED!=None: seed(SEED)
    #waypoints = [[-values[0], -values[1], values[2]], [values[0], -values[1], values[2]], [values[0], values[1], values[2]], [-values[0], values[1], values[2]]]
    print("Waypoints: ", waypoints)

    for i in range(num_waypoints):
        print(waypoints[i], waypoints[i - 1])
        scalare_waypoints = waypoints_distances[i % len(waypoints_distances)]

        T = log_increment_flight_time(scalare_waypoints)
        
        traj = TrajectoryGenerator(waypoints[i], waypoints[(i + 1) % num_waypoints], T)
        traj.solve()
        x_coeffs[i] = traj.x_c
        y_coeffs[i] = traj.y_c
        z_coeffs[i] = traj.z_c

    quad_sim(x_coeffs, y_coeffs, z_coeffs)



if __name__ == "__main__":
    main()


