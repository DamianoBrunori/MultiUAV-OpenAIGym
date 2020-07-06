import sys
import os
from random import uniform

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines
from matplotlib import collections  as mc
import numpy as np

import time

# sys.path.append(os.path.abspath("../custom_gym"))
# from agent import Agent

UAVS=[]
class UAV():
    def __init__(self,id, start_pos = (0,0,0),dest_pos=(0,0,0),battery_level=100):
        self._uav_ID = id
        self.start_pos = start_pos
        self.dest_pos = dest_pos
        self._battery_level = battery_level
        self._coming_home = False
        self._crashed = False
        

class FlightSimulator():
    def __init__(self,uavs=None):
        self.uavs = uavs
        self.num_uavs = len(uavs)
        self.started = False
        
    # Start the simulation of flights
    def start(self, num_uavs,xmin,ymin,xmax,ymax,zmin,zmax):
            
        for i in range( num_uavs ):
            start_pos = ( uniform(xmin,xmax), uniform(ymin,ymax),  uniform(zmin,zmax) )
            dest_pos = ( uniform(xmin,xmax), uniform(ymin,ymax), uniform(zmin,zmax) ) 
            uav = UAV(id = "uav"+ str(i),start_pos = start_pos,dest_pos = dest_pos )
            self.uavs.append( uav )
            
        self.started = True
        self.num_uavs = num_uavs
        self.__save()
        pass

    # Save result of simulation
    def __save(self): #TODO
        pass

    def get_all_paths(self):
        return [ [uav.start_pos, uav.dest_pos ] for uav in self.uavs ]
    
    def get_all_2D_paths(self):
        return [ [uav.start_pos[:-1], uav.dest_pos[:-1] ] for uav in self.uavs ]


    def plot_2D(self):
        if (not self.started):
            raise Exception("You need to start the simulator first")
        # colors = np.array([(1,0,0,1), (0, 1, 0, 1), (0, 0, 1, 1)])
        # paths = self.get_all_2D_paths()  
        paths,starts_xs,starts_ys,dest_xs,dest_ys = ( [] for _ in range(5) )
        for uav in self.uavs:
            paths.append( [uav.start_pos[:-1],uav.dest_pos[:-1]] )
            starts_xs.append(uav.start_pos[0])
            starts_ys.append(uav.start_pos[1])
            dest_xs.append(uav.dest_pos[0])
            dest_ys.append(uav.dest_pos[1])


        lc = mc.LineCollection(paths, colors=None, linewidths=2, linestyle="--")
        fig, ax = plt.subplots()
        ax.scatter(x=starts_xs, y=starts_ys, c='g', s=40, label="Start")
        ax.scatter(x=dest_xs, y=dest_ys, c='b', s=40, label="Dest")

        ax.legend()
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        
        
        fig.canvas.set_window_title("Trajectories")
        ax.set_title("2D Trajectories")
        plt.show()        





    def __new_line2(self,ax,p1, p2):
        
        xmin, xmax = ax.get_xbound()

        if(p2[0] == p1[0]):
            xmin = xmax = p1[0]
            ymin, ymax = ax.get_ybound()
        else:
            ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
            ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

        l = mlines.Line2D([xmin,xmax], [ymin,ymax])
        return l


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
            ax.scatter(xs=uav.start_pos[0],ys=uav.start_pos[1],zs=uav.start_pos[2],
                c='g', s=40, label="Start")
            ax.scatter(xs=uav.dest_pos[0],ys=uav.dest_pos[1],zs=uav.dest_pos[2],
                c='black', s=40, label="Dest")

            #lii.set_dash_joinstyle('round')
            #lii.set_solid_joinstyle('round')
            lii.set_solid_capstyle('round')

        ax.autoscale()
        plt.title('3D-Figure')

        plt.show()
        #save plot
        # plt.savefig('3D_Line.png', dpi=600, facecolor='w', edgecolor='w',
        #             orientation='portrait')
                

if (__name__ == "__main__"):
    uavs = []   

  
    fs = FlightSimulator(uavs)
    fs.start(5,-10,10,-20,20,4,4)
    fs.plot_2D()
    fs.plot_3D()


    '''
    3D PLOT
    '''

    