import sys
import os
from random import uniform

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines
from matplotlib import collections  as mc
import numpy as np

import time
import math
from custom_gym.my_utils import *
from my_utils import *
# Simulation parameters
g = 9.81
m = 0.2
Ixx = 1
Iyy = 1
Izz = 1
T = 5

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


def function(x, A, B):
    return math.exp(A*x) * math.sin(B*x)

# sys.path.append(os.path.abspath("../custom_gym"))
# from agent import Agent

# TODO: create a config for global variables 
STD_UAV_VELOCITY = 27 #standard uav velocity (m/s)
## Class representing UAV drones
class UAV():
    def __init__(self,id, start_pos = (0,0,0),dest_pos=(0,0,0),battery_level=100,velocity=STD_UAV_VELOCITY, start_acc=[0,0,0], des_acc=[0,0,0]):
        self.id = id # Unique (should be) identifier of drones
        self.start_pos = start_pos
        self.dest_pos = dest_pos
        self.trajectory = dict( x = [start_pos[0],dest_pos[0] ],
                                y = [start_pos[1],dest_pos[1] ],
                                z = [start_pos[2],dest_pos[2] ] ) # Dictionary of coord-points of trajectory (initially just [start,dest]) 
        #Accelerazione------------------
        self.start_x_acc = start_acc[0]
        self.start_y_acc = start_acc[1]
        self.start_z_acc = start_acc[2]
        self.des_x_acc = des_acc[0]
        self.des_y_acc = des_acc[1]
        self.des_z_acc = des_acc[2]
        #-------------------------------
        self.velocity = velocity   #
        self._battery_level = battery_level
        self._coming_home = False
        self._crashed = False

    ## Preatty print trajectory of current drone
    def print_trajectory(self):
        coo=["x","y","z"]
        for i,k in enumerate(self.trajectory):
            print("A.",coo[i], ":",self.start_pos[i],",B.",coo[i],": ",self.dest_pos[i])
            print(coo[i]+"s:",self.trajectory[k])

class FlightSimulator():
    def __init__(self,uavs=None):
        self.uavs = uavs # List of uavs in the simulator
        self.num_uavs = len(uavs)
        self.started = False
        self.timesample = 0.5 # NOTE: it's the sampling interval time for a trajectory
        ## Parameters of gaussian
        self.mean = 0
        self.std = 1

    ## Generate 3D point (x,y,z) inside bounded uniform distribution
    def __gen_uniform_position(self,xmin,xmax,ymin,ymax,zmin,zmax):
        return ( uniform(xmin,xmax), uniform(ymin,ymax), uniform(zmin,zmax) )

    # Start the simulation of flights
    def start(self, num_uavs,xmin,xmax,ymin,ymax,zmin,zmax,mean=0,std=0.3):
            
        for i in range( num_uavs ):
            # Choose casual start and dest pos in specified bounds
            
            start_pos = self.__gen_uniform_position(xmin,xmax,ymin,ymax,zmin,zmax)

            dest_pos =  self.__gen_uniform_position(xmin,xmax,ymin,ymax,zmin,zmax)
            
            # Create uav object
            uav = UAV(id = "uav"+ str(i),start_pos = start_pos,dest_pos = dest_pos )
            self.uavs.append( uav )

            # Simulate Trajectory for this uav
            uav.trajectory["x"],uav.trajectory["y"],uav.trajectory["z"]= self.__create_sample_points(start_pos,dest_pos,
                                                                            self.timesample,uav.velocity)
            # Print computed trajectory
            uav.print_trajectory()


            
            

        self.started = True
        self.num_uavs = num_uavs
        
        self.mean = mean
        self.std = std

        self.__save()
        pass


    # TODO Save result (trajectories, other infos...) of simulation in a file or other
    def __save(self): 
        pass

    def get_all_paths(self):
        return [ [uav.start_pos, uav.dest_pos ] for uav in self.uavs ]
    
    def get_all_2D_paths(self):
        return [ [uav.start_pos[:-1], uav.dest_pos[:-1] ] for uav in self.uavs ]

    # From a,b points generate [a,...,c,....,b] points where c are points inside interval on segment connecting a-b
    # TODO remove it (DEPRECATED)  
    def __interpolate(self,a,b,precision):
        points=[]
        for i in range(0,precision+1):
            res = a + i * (b-a)/(precision+1) 
            points.append(res)
        points.append(b)
        return points
    
    # Generate points inside a-b segment [a,...,b] using step as a time based sampler
    def __create_points_single_axis(self,a,b,step):
        # print("**",a,b)
        if(step==0):
            return [a],1
        points = np.arange( min(a,b), max(a,b),step) 
        points = np.append(points,max(a,b))
        # swap if necessary
        if(a != min(a,b)):
            points=points[::-1]
        return  points , len(points)
    


    # Return sampled array of points with Gaussian Noise based on velocity and timestamp
    def __create_sample_points(self,a,b,timesample,velocity):
        # TODO 3 velocities x y z
        
        xi, xf = a[0], b[0]
        yi, yf = a[1], b[1]
        zi, zf = a[2], b[2]
        
        dx= abs(xf-xi) 
        dy= abs(yf-yi)        
        dz= abs(zf-zi)        
        
        d = math.sqrt( dx*dx + dy*dy + dz*dz)
        vx = dx/d * velocity
        vy = dy/d * velocity
        vz = dz/d * velocity

        step_x = vx * timesample
        step_y = vy * timesample
        step_z = vz * timesample
        
        traj_x, len_x = self.__create_points_single_axis(xi,xf,step_x)
        traj_y, len_y = self.__create_points_single_axis(yi,yf,step_y)
        traj_z, len_z = self.__create_points_single_axis(zi,zf,step_z)
        res = [ traj_x, traj_y, traj_z ]
        lens = [ len_x,len_y,len_z ]
        num_points=max(lens)
        
        # Extend singleton point arrays
        for i,ll in enumerate(lens) :
            if ll==1:
                res[i] = np.array( list(res[i]) * num_points )
            else:
                # print(ll)
                noise = np.random.normal(self.mean,self.std, ll-2 )  
                res[i] +=  np.append(np.append(0,noise),0)
   
        # print("res",res)
        return res 

        

    def plot_2D(self):
        if (not self.started):
            raise Exception("You need to start the simulator first")
        # colors = np.array([(1,0,0,1), (0, 1, 0, 1), (0, 0, 1, 1)])
        # paths = self.get_all_2D_paths()  
        paths,starts_xs,starts_ys,dest_xs,dest_ys = ( [] for _ in range(5) )
        

        fig, ax = plt.subplots()

        for uav in self.uavs:
            paths.append( [uav.start_pos[:-1],uav.dest_pos[:-1]] )
            starts_xs.append(uav.start_pos[0])
            starts_ys.append(uav.start_pos[1])
            dest_xs.append(uav.dest_pos[0])
            dest_ys.append(uav.dest_pos[1])


            ax.plot(uav.trajectory["x"],uav.trajectory["y"])
            xx=uav.start_pos[0],uav.dest_pos[0]

        lc = mc.LineCollection(paths, colors="black", linewidths=1, linestyle="--")
        print(lc)
        ax.scatter(x=starts_xs, y=starts_ys, c='g', s=40, label="Start")
        ax.scatter(x=dest_xs, y=dest_ys, c='black', s=40, label="Dest")

        ax.legend()
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        
        
        fig.canvas.set_window_title("Trajectories")
        ax.set_title("2D Trajectories")
        plt.show()  

        # points = 100 #Number of points in a line
        # xmin, xmax = -1, 5
        # xlist = list(map(lambda x: float(xmax - xmin)*x/points, range(points+1)) )
        # ylist = list ( map(lambda y: function(y, -1, 5), xlist) )
        # plt.plot(xlist, ylist)
        # plt.show()





    def plot_3D(self):

        cmap=plt.get_cmap('copper')
        colors=[cmap(float(ii)/(max(self.num_uavs-1,1))) for ii in range(self.num_uavs)]

        #plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        for idx,uav in enumerate(self.uavs):
            # segii= [ list(uav.start_pos), list(uav.dest_pos) ]
            xs = [uav.start_pos[0], uav.dest_pos[0] ] 
            ys = [uav.start_pos[1], uav.dest_pos[1] ]
            zs = [uav.start_pos[2], uav.dest_pos[2] ]
            lii,=ax.plot(xs, ys, zs, color=colors[idx],linewidth=2,linestyle="--")
            ax.plot(uav.trajectory["x"],uav.trajectory["y"],uav.trajectory["z"])

            ax.scatter(xs=uav.start_pos[0],ys=uav.start_pos[1],zs=uav.start_pos[2],
                c='g', s=40, label="Start")
            ax.scatter(xs=uav.dest_pos[0],ys=uav.dest_pos[1],zs=uav.dest_pos[2],
                c='black', s=40, label="Dest")

            #lii.set_dash_joinstyle('round')
            #lii.set_solid_joinstyle('round')
            lii.set_solid_capstyle('round')

        ax.autoscale()
        ax.legend
        fig.canvas.set_window_title("Trajectories")
        ax.set_title("3D Trajectories")
        plt.show()
        #save plot
        # plt.savefig('3D_Line.png', dpi=600, facecolor='w', edgecolor='w',
        #             orientation='portrait')
                

if (__name__ == "__main__"):
    uavs = []   

    
    fs = FlightSimulator(uavs)
    fs.start(num_uavs=5,xmin=-10,xmax=10,ymin=-10,ymax=10,zmin=4,zmax=4,mean=0.2,std=10)
    fs.plot_2D()
    fs.plot_3D()
    [u.print_trajectory() for u in fs.uavs]

    