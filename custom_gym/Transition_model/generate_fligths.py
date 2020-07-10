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
from my_utils import *

def function(x, A, B):
    return math.exp(A*x) * math.sin(B*x)

# sys.path.append(os.path.abspath("../custom_gym"))
# from agent import Agent


PRECISION = 10 # TODO move it in constructor

class UAV():
    def __init__(self,id, start_pos = (0,0,0),dest_pos=(0,0,0),battery_level=100):
        self._uav_ID = id
        self.start_pos = start_pos
        self.dest_pos = dest_pos
        self.trajectory = dict( x = [start_pos[0],dest_pos[0] ],
                                y = [start_pos[1],dest_pos[1] ],
                                z = [start_pos[2],dest_pos[2] ] )
        self.velocity = 1        
        self._battery_level = battery_level
        self._coming_home = False
        self._crashed = False

class FlightSimulator():
    def __init__(self,uavs=None):
        self.uavs = uavs
        self.num_uavs = len(uavs)
        self.started = False

        self.mean = 0
        self.std = 1
        
    # Start the simulation of flights
    def start(self, num_uavs,xmin,ymin,xmax,ymax,zmin,zmax,mean=0,std=0.3):
            
        for i in range( num_uavs ):
            start_pos = ( uniform(xmin,xmax), uniform(ymin,ymax),  uniform(zmin,zmax) )
            dest_pos = ( uniform(xmin,xmax), uniform(ymin,ymax), uniform(zmin,zmax) ) 
            uav = UAV(id = "uav"+ str(i),start_pos = start_pos,dest_pos = dest_pos )
            self.uavs.append( uav )

            uav.trajectory["x"] = np.array( self.__interpolate(uav.start_pos[0],uav.dest_pos[0],PRECISION) )
            noise = np.random.normal(self.mean,self.std,PRECISION) # NOTE
            uav.trajectory["x"] += np.append(np.append(0,noise),0)

            uav.trajectory["y"] = np.array( self.__interpolate(uav.start_pos[1],uav.dest_pos[1],PRECISION) )
            noise = np.random.normal(self.mean,self.std,PRECISION) # NOTE
            uav.trajectory["y"] +=  np.append(np.append(0,noise),0)

            uav.trajectory["z"] = np.array( self.__interpolate(uav.start_pos[2],uav.dest_pos[2],PRECISION) )
            if(zmin != zmax):
                noise = np.random.normal(self.mean,self.std,PRECISION) # NOTE
                uav.trajectory["z"] +=  np.append(np.append(0,noise),0)

        self.started = True
        self.num_uavs = num_uavs
        
        self.mean = mean
        self.std = std

        self.__save()
        pass

    # Save result of simulation
    def __save(self): #TODO
        pass

    def get_all_paths(self):
        return [ [uav.start_pos, uav.dest_pos ] for uav in self.uavs ]
    
    def get_all_2D_paths(self):
        return [ [uav.start_pos[:-1], uav.dest_pos[:-1] ] for uav in self.uavs ]

    def __interpolate(self,a,b,precision):
        points=[]
        for i in range(0,precision+1):
            res = a + i * (b-a)/(precision+1) 
            points.append(res)
        points.append(b)
        return points
        
    def plot_2D(self):
        if (not self.started):
            raise Exception("You need to start the simulator first")
        # colors = np.array([(1,0,0,1), (0, 1, 0, 1), (0, 0, 1, 1)])
        # paths = self.get_all_2D_paths()  
        paths,starts_xs,starts_ys,dest_xs,dest_ys = ( [] for _ in range(5) )
        num_points = 100 #Number of points in a line
        precision = 10  # num of points per line
        

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
        colors=[cmap(float(ii)/(self.num_uavs-1)) for ii in range(self.num_uavs)]

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
    
    fs.start(num_uavs=5,xmin=-10,ymin=10,xmax=-20,ymax=20,zmin=4,zmax=4,mean=0,std=0.3)
    fs.plot_2D()
    fs.plot_3D()


    