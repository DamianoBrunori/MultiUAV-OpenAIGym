"""
Simulate a quadrotor following a 3D trajectory

Author: Daniel Ingram (daniel-s-ingram)
"""

from math import cos, sin
import numpy as np
from Quadrotor import Quadrotor
from TrajectoryGenerator import TrajectoryGenerator
from mpl_toolkits.mplot3d import Axes3D
import math
from os import mkdir
from os.path import join, isdir
from my_utils import *

show_animation = True



# Simulation parameters
g = 9.81
m = 0.2
Ixx = 1
Iyy = 1
Izz = 1




#T = 5           #<---------------------------------




acc_max_scalar = 3 #(m/s^2)
vel_max_scalar = 18 #(m/s)

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

waypoints = [[10, 0, 10], [10, 3000, 10], [10, 0, 10]]
num_waypoints = len(waypoints)

def quad_sim(x_c, y_c, z_c):
    """
    Calculates the necessary thrust and torques for the quadrotor to
    follow the trajectory described by the sets of coefficients
    x_c, y_c, and z_c.
    """
    x_pos = waypoints[0][0]
    y_pos = waypoints[0][1]
    z_pos = waypoints[0][2]
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

    des_yaw = 0

    dt = 0.1
    t = 0

    q = Quadrotor(x=x_pos, y=y_pos, z=z_pos, roll=roll,
                  pitch=pitch, yaw=yaw, size=1, show_animation=show_animation)

    i = 0
    n_run = 3 # Number of exploration of waypoints
    irun = 0

    
    while True:

        goal_x = waypoints[(i+1)%num_waypoints][0]
        goal_y = waypoints[(i + 1) % num_waypoints][1]
        L = distance_3D(waypoints[i],waypoints[(i+1)%num_waypoints])
        T = (L*acc_max_scalar + vel_max_scalar **2 ) /(acc_max_scalar * vel_max_scalar)
        T_s = vel_max_scalar / acc_max_scalar
 
        # acc_max_x, acc_max_y, acc_max_z = get_3D_components(waypoints[i],waypoints[(i+1)%num_waypoints],acc_max_scalar)
        # vel_max_x, vel_max_y, vel_max_z = get_3D_components(waypoints[i],waypoints[(i+1)%num_waypoints],vel_max_scalar)
 
        acc_max_x, acc_max_y = get_2D_components(waypoints[i],waypoints[(i+1)%num_waypoints],acc_max_scalar)
        vel_max_x, vel_max_y = get_2D_components(waypoints[i],waypoints[(i+1)%num_waypoints],vel_max_scalar)



        while t <= T:

            acc_ms = math.sqrt(x_acc ** 2 + y_acc ** 2)
            vel_ms = math.sqrt(x_vel ** 2 + y_vel ** 2)
            
            if( not is_bang_cost_available( L,acc_max_scalar,vel_max_scalar) ):
                raise Exception("Bang cost bang not feasible")

            print("running:","{:.2f}".format(t))
            # 3D TEST
            # des_x_pos, des_y_pos, des_z_pos = bang_position(x_pos,y_pos,z_pos,
            #     acc_max_x,acc_max_y,acc_max_z,
            #     vel_max_x,vel_max_y,vel_max_z,
            #     t,T,T_s)

            # des_x_vel, des_y_vel, des_z_vel = bang_velocity(acc_max_x,acc_max_y,acc_max_z,
            #     vel_max_x,vel_max_y,vel_max_z,
            #     t,T,T_s)

            # des_x_acc, des_y_acc, des_z_acc = bang_accelleration(acc_max_x,acc_max_y,acc_max_z,
            #     t,T,T_s)

            # 2D TEST
            des_x_pos, des_y_pos = bang_position_2D(x_pos,y_pos,
                acc_max_x,acc_max_y,
                t,T,T_s, goal_x, goal_y)

            des_x_vel, des_y_vel = bang_velocity_2D(acc_max_x,acc_max_y,
                t,T,T_s)

            des_x_acc, des_y_acc = bang_accelleration_2D(acc_max_x,acc_max_y,
                t,T,T_s)


            # ORIGINAL

            # # des_x_pos = calculate_position(x_c[i], t)
            # # des_y_pos = calculate_position(y_c[i], t)
            des_z_pos = calculate_position(z_c[i], t)
            # # des_x_vel = calculate_velocity(x_c[i], t)
            # # des_y_vel = calculate_velocity(y_c[i], t)
            des_z_vel = calculate_velocity(z_c[i], t)
            # des_x_acc = calculate_acceleration(x_c[i], t)
            # des_y_acc = calculate_acceleration(y_c[i], t)
            des_z_acc = calculate_acceleration(z_c[i], t)



            thrust = m * (g + des_z_acc + Kp_z * (des_z_pos -
                                                  z_pos) + Kd_z * (des_z_vel - z_vel))

            roll_torque = Kp_roll * \
                (((des_x_acc * sin(des_yaw) - des_y_acc * cos(des_yaw)) / g) - roll)
            pitch_torque = Kp_pitch * \
                (((des_x_acc * cos(des_yaw) - des_y_acc * sin(des_yaw)) / g) - pitch)
            yaw_torque = Kp_yaw * (des_yaw - yaw)

            roll_vel += roll_torque * dt / Ixx
            pitch_vel += pitch_torque * dt / Iyy
            yaw_vel += yaw_torque * dt / Izz

            roll += roll_vel * dt
            pitch += pitch_vel * dt
            yaw += yaw_vel * dt

            R = rotation_matrix(roll, pitch, yaw)
            acc = (np.matmul(R, np.array(
                [0, 0, thrust.item()]).T) - np.array([0, 0, m * g]).T) / m
            
            x_acc = acc[0]
            y_acc = acc[1]
            z_acc = 0

            x_vel += x_acc * dt
            y_vel += y_acc * dt
            z_vel += z_acc * dt

            x_pos += x_vel * dt
            y_pos += y_vel * dt
            z_pos += z_vel * dt

            if (T_s < t < T - T_s):
                y_vel = 18
                y_acc = 0

            q.update_pose(x_pos, y_pos, z_pos, roll, pitch, yaw)
            
            # # # # # # # 
            
            print("X_pos:","{:.2f}".format(x_pos) ,"\tX_vel (m/s):", "{:.2f}".format(x_vel), "\tX_acc (m/s^2):", "{:.2f}".format(x_acc))
            print("Y_pos:","{:.2f}".format(y_pos) ,"\tY_vel (m/s):", "{:.2f}".format(y_vel), "\tY_acc (m/s^2):", "{:.2f}".format(y_acc))
            print("Z_pos:","{:.2f}".format(z_pos) ,"\tZ_vel (m/s):", "{:.2f}".format(z_vel), "\tZ_acc (m/s^2):", "{:.2f}".format(z_acc))
            print("Acceleration:", "{:.2f}".format(acc_ms))
            print("Velocity (m/s):", "{:.2f}".format(vel_ms))

            # # # # # # # 
            
            t += dt

        print("-"*20,"[Missing",distance_3D([x_pos,y_pos,z_pos],waypoints[(i+1)%num_waypoints]), "m]","-"*20)
        
        t = 0
        i = (i + 1) % 4
        irun += 1
        if irun >= n_run:
            break
        print("-"*20,"[PASSING TO NEXT WAYPOINT]","-"*20)

    print("-"*20,"MISSION COMPLETE","-"*20)


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


def is_bang_cost_available(L,acc_max,vel_max):
    return L > vel_max ** 2 / acc_max


def bang_position(x_pos,y_pos,z_pos,
    acc_max_x,acc_max_y,acc_max_z,
    vel_max_x,vel_max_y,vel_max_z,
    t,T,T_s):
    # acc_max_x, acc_max_y, acc_max_z = get_acc_max()
    # vel_max_x, vel_max_y, vel_max_z = get_vel_max()
    
    if t < T_s:
        x_des_pos = x_pos + 0.5 * acc_max_x * t**2
        y_des_pos = y_pos + 0.5 * acc_max_y * t**2
        z_des_pos = z_pos + 0.5 * acc_max_z * t**2
    elif T_s < t <= T-T_s:
        x_des_pos = x_pos + vel_max_x * t
        y_des_pos = y_pos + vel_max_y * t
        z_des_pos = z_pos + vel_max_z * t
    elif t > T-T_s :
        x_des_pos = x_pos + 0.5 * (-acc_max_x) * t**2
        y_des_pos = y_pos + 0.5 * (-acc_max_y) * t**2
        z_des_pos = z_pos + 0.5 * (-acc_max_z) * t**2
    
    return x_des_pos, y_des_pos, z_des_pos

def bang_velocity(acc_max_x,acc_max_y,acc_max_z,
    vel_max_x,vel_max_y,vel_max_z,
    t,T,T_s):
    # acc_max_x, acc_max_y, acc_max_z = get_acc_max()
    # vel_max_x, vel_max_y, vel_max_z = get_vel_max()
    
    if t < T_s:
        x_des_vel = acc_max_x * t
        y_des_vel = acc_max_y * t
        z_des_vel = acc_max_z * t
    elif T_s < t <= T-T_s:
        x_des_vel = vel_max_x 
        y_des_vel = vel_max_y 
        z_des_vel = vel_max_z 
    elif t > T-T_s :
        x_des_vel = -acc_max_x * t
        y_des_vel = -acc_max_y * t
        z_des_vel = -acc_max_z * t

    
    return x_des_vel, y_des_vel, z_des_vel

def bang_accelleration(acc_max_x,acc_max_y,acc_max_z,
    t,T,T_s):
    if t < T_s:
        x_des_acc = acc_max_x
        y_des_acc = acc_max_y
        z_des_acc = acc_max_z
    elif T_s < t <= T-T_s:
        x_des_acc = 0 
        y_des_acc = 0 
        z_des_acc = 0 
    elif t > T-T_s :
        x_des_acc = -acc_max_x 
        y_des_acc = -acc_max_y 
        z_des_acc = -acc_max_z 

    return x_des_acc, y_des_acc, z_des_acc


def bang_position_2D(x_pos,y_pos,
    acc_max_x,acc_max_y,t,T,T_s, goal_x, goal_y):
    
    if t <= T_s:
        x_des_pos = x_pos + 0.5 * acc_max_x * t**2
        y_des_pos = y_pos + 0.5 * acc_max_y * t**2
    elif T_s < t <= T-T_s:
        x_des_pos = x_pos + 0.5 * acc_max_x * T_s * (t - 0.5 * T_s)
        y_des_pos = y_pos + 0.5 * acc_max_y * T_s * (t - 0.5 * T_s)
    elif t > T-T_s:
        x_des_pos = goal_x + 0.5 * (-acc_max_x) * (T-t)**2
        y_des_pos = goal_y + 0.5 * (-acc_max_y) * (T-t)**2
    
    return x_des_pos, y_des_pos

def bang_velocity_2D(acc_max_x,acc_max_y,t,T,T_s):
    # acc_max_x, acc_max_y, acc_max_z = get_acc_max()
    # vel_max_x, vel_max_y, vel_max_z = get_vel_max()
    
    if t < T_s:
        x_des_vel = acc_max_x * t
        y_des_vel = acc_max_y * t
    elif T_s < t <= T-T_s:
        x_des_vel = T_s * acc_max_x
        y_des_vel = T_s * acc_max_y
    elif t > T-T_s:
        x_des_vel = acc_max_x * (T - t)
        y_des_vel = acc_max_y * (T - t)
    
    return x_des_vel, y_des_vel

def bang_accelleration_2D(acc_max_x,acc_max_y,
    t,T,T_s):
    if t < T_s:
        x_des_acc = acc_max_x
        y_des_acc = acc_max_y
    elif T_s < t <= T-T_s:
        x_des_acc = 0 
        y_des_acc = 0 
    elif t > T-T_s :
        x_des_acc = -acc_max_x 
        y_des_acc = -acc_max_y 

    return x_des_acc, y_des_acc




def get_3D_components(start3D,end3D,scalar):
    d_x = end3D[0] - start3D[0] 
    d_y = end3D[1] - start3D[1] 
    d_z = end3D[2] - start3D[2] 
    d = distance_3D(start3D, end3D)

    alpha = math.acos(d_x/d)
    beta = math.acos(d_y/d)
    gamma = math.acos(d_z/d)

    x_component = math.cos(alpha) * scalar
    y_component = math.cos(beta) * scalar
    z_component = math.cos(gamma) * scalar

    return x_component, y_component, z_component
    

def get_2D_components(start2D,end2D,scalar):
    d_x = end2D[0] - start2D[0] 
    d_y = end2D[1] - start2D[1] 
    d = distance_2D(start2D, end2D)
    
    angle = math.acos(d_x/d)
    
    x_component = math.cos(angle) * scalar
    y_component = math.sin(angle) * scalar

    return x_component, y_component
    

def distance_2D(start,end):
    return math.sqrt( (end[1] - start[1] ) **2 + (end[0] - start[0] )**2 )    

def distance_3D(start,end):
    return math.sqrt( (end[2] - start[2] )**2 + (end[1] - start[1] ) **2 + (end[0] - start[0] )**2 )    


def main():
    sys.stdout = Logger()
    print("\n\n\n" + "".join(["#"] * 50))
    if sys.platform.startswith('linux'):
        print("User:", format(getenv("USER")))  # For Linux
    if sys.platform.startswith('win32'):
        print("User:", format(getenv("USERNAME")))  # For Windows
    print("OS:", sys.platform)
    print("Date:", format(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]))
    print("".join(["#"] * 50) + "\n\n\n")
    """
    Calculates the x, y, z coefficients for the four segments 
    of the trajectory
    """
    x_coeffs = [[], [], [], []]
    y_coeffs = [[], [], [], []]
    z_coeffs = [[], [], [], []]
    
    for i in range(num_waypoints):
        L = distance_3D(waypoints[i],waypoints[(i+1)%num_waypoints])
        T = (L*acc_max_scalar + vel_max_scalar **2 ) /(acc_max_scalar * vel_max_scalar)
        print("L:",L,", T:",T)

        traj = TrajectoryGenerator(waypoints[i], waypoints[(i + 1) % num_waypoints], T)
        traj.solve()
        x_coeffs[i] = traj.x_c
        y_coeffs[i] = traj.y_c
        z_coeffs[i] = traj.z_c


    quad_sim(x_coeffs, y_coeffs, z_coeffs)


if __name__ == "__main__":
    main()
