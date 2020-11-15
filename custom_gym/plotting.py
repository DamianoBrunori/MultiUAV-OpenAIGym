# MAIN CLASS AND METHODS TO VISUALIZE 2D AND 3D MAP VIEW (WITH POINTS AND CELLS) EITHER OF THE WHOLE ENVIRONMENT OR OF A PART OF IT; IT IS USED ALSO TO SAVE THE STATUS MATRICES.

from os import mkdir
from os.path import join, isdir
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap, BoundaryNorm
from decimal import Decimal
from my_utils import *
from load_and_save_data import *
from scenario_objects import Point, Cell, User
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from statistics import stdev

# If EnodeB has not been created in 'scenario_objets', then if you try to plot it, it will obviously raise an Error.

u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)

env_directory = "Environment_Views"
if not isdir(env_directory): mkdir(env_directory)

MAX_CLUSTERS_COLORS = 20 # --> It is used to generate 20 different colors for 20 different clusters.
PLOT_EACH_N_EPOCH = 200
CONSTANT_FOR_LABELS = 10
LABELS_EACH_N_EPOCH = PLOT_EACH_N_EPOCH*CONSTANT_FOR_LABELS
RANGE_EPOCHS_TO_PLOT = range(0, EPISODES+1, PLOT_EACH_N_EPOCH)
RANGE_EPOCHS_TO_VISUALIZE = range(0, EPISODES+1, LABELS_EACH_N_EPOCH)
RANGE_X_TICKS = range(0, EPISODES+1, LABELS_EACH_N_EPOCH)

if MIDDLE_CELL_ASSUMPTION==True:
    incr_assumed_coord = 0.5
else:
    incr_assumed_coord = 0.0

class Plot:
    '''
    |-------------------------------------------------------------|
    |Define a class containings method aimed to plot or to compute|
    |elements used to plot:                                       |
    |-------------------------------------------------------------|
    '''

    def __init__(self):
        clusters_colors, num_color_range = self.RGBA_01_random_colors(MAX_CLUSTERS_COLORS)
        self.clusters_colors = clusters_colors
        self.num_color_range = num_color_range
        pass

    def update_animation_3D(self, num, dataLines, lines, circles, n_circles_range, ax):
        # # # # # # # # # # # # #  
        # 3D animation updating #
        # # # # # # # # # # # # #

        for line, data, circle_idx in zip(lines, dataLines, n_circles_range):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_xdata(data[1, :num])
            line.set_ydata(data[0, :num])
            line.set_3d_properties(data[2, :num])
            line.set_marker("4")
            line.set_markersize(16)
            
            circles[circle_idx].remove()
            x =  (data[1, num]) + ACTUAL_UAV_FOOTPRINT * np.outer(np.cos(u), np.sin(v))
            y = (data[0, num]) + ACTUAL_UAV_FOOTPRINT * np.outer(np.sin(u), np.sin(v))
            z = 0 * np.outer(np.ones(np.size(u)), np.cos(v))
            
            surf = ax.plot_surface(x, y, z, color=UAVS_COLORS[circle_idx], alpha=0.18, linewidth=0)
            circles[circle_idx] = surf

        return tuple(lines) + tuple(circles)

    def update_animation_2D(self, num, dataLines, lines, circles):
        # # # # # # # # # # # # #  
        # 2D animation updating #
        # # # # # # # # # # # # #

        for line, data, circle in zip(lines, dataLines, circles):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, :num])
            line.set_marker("4")
            line.set_markersize(16)
            
            circle.center = (data[0][num], data[1][num])
        
        return tuple(lines) + tuple(circles)

    def compute_status_matrix(self, matrix_area, area_height, area_width):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Returns a matrix made by the elements representing their states;  #
        # the states are extracted from 'matrix_area'.                      #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        status_matrix = [[matrix_area[i][j]._status for j in range(area_width)] for i in range(area_height)]

        return status_matrix

    def compute_perceived_status_matrix(self, cells_matrix, area_height, area_width, reduced_height, reduced_width):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Returns a matrix made by the elements representing the 'perceived' states; if a larger resolution w.r.t.      #
        # the minimum one is used, then there will be cells which results to be occupied in full even if the actual     #
        # obstacles inside them only occupies a part of them; the 'perceived' states are extracted from 'cells_matrix'. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        perceived_status_matrix = np.zeros((area_height, area_width))

        for r in range(reduced_height):
            for c in range(reduced_width):
                current_cell = cells_matrix[r][c]
                current_cell_status = current_cell._status

                if (current_cell_status == OBS_IN):
                    value_to_assign = OBS_IN
                elif (current_cell_status == CS_IN):
                    value_to_assign = CS_IN
                elif (current_cell_status == ENB_IN):
                    value_to_assign = ENB_IN
                else:
                    value_to_assign = FREE

                for point in current_cell._points:
                    perceived_status_matrix[point._y_coord][point._x_coord] = value_to_assign

        return perceived_status_matrix

    def extract_coord_from_xyz(self, coordinates):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Returns in separated lists the coordinates (x,y,z) which are contained in 'coordinates'.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        N_coordinates = len(coordinates)
        x_extracted_coords = [coordinates[coords_idx]._x_coord for coords_idx in range(N_coordinates)]
        y_extracted_coords = [coordinates[coords_idx]._y_coord for coords_idx in range(N_coordinates)]
        z_extracted_coords = [coordinates[coords_idx]._z_coord for coords_idx in range(N_coordinates)]

        return x_extracted_coords, y_extracted_coords, z_extracted_coords

    def RGBA_01_random_colors(self, num_colors):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Assign as many random color as it is 'num_color'  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # #

        num_color_range = range(num_colors)
        colors = [None for color in num_color_range]
        
        for color_idx in num_color_range:
            colors[color_idx] = (random.randint(0, 255)/255, random.randint(0, 255)/255, random.randint(0, 255)/255, 1.0)

        return colors, num_color_range

    def plt_map_views(self, obs_points=None, cs_points=None, enb_point=None,
                      obs_cells=None, cs_cells=None, enb_cells=None, points_status_matrix=None,
                      cells_status_matrix=None, perceived_status_matrix=None, users=None, centroids=None,
                      clusters_radiuses=None, area_height=None, area_width=None, N_cells_row=None,
                      N_cells_col=None, agents_paths=None, path_animation=False, where_to_save=None, episode=None):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Create 3 figures which contains respectively:             #
        #   -   2D and 3D Points-Map;                               #
        #   -   2D and 3D Cells-Map;                                #
        #   -   2D and 3D Mixed-map (i.e., with Points and Cells);  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  

        self.num_color_range = range(len(centroids))

        # Define colors to use for the plots:
        WHITE = "#ffffff"
        DARK_RED = "#800000" # 0.5, 0, 0
        LIGHT_RED = "#ff0000"
        DARK_BLUE = "#000099"
        LIGHT_BLUE = "#66ffff"
        DARK_GREEN = "#006600"
        LIGHT_GREEN = "#66ff99"
        GOLD = '#FFD700'
        # UAVS colors:
        VIOLET = '#EE82EE'
        ORANGE = '#FFA500'
        GREY = '#808080'
        BROWN = '#A52A2A'
        UAVS_COLORS = [VIOLET, ORANGE, GREY, BROWN]

        # Define colored canvas for the legend:
        DARK_RED_square = mlines.Line2D([], [], color=DARK_RED, marker='s', markersize=15, label="'Point' eNodeB")
        LIGHT_RED_square = mlines.Line2D([], [], color=LIGHT_RED, marker='s', markersize=15, label="'Cell' eNodeB")
        DARK_BLUE_square = mlines.Line2D([], [], color=DARK_BLUE, marker='s', markersize=15, label="'Point' Obstacles")
        LIGHT_BLUE_square = mlines.Line2D([], [], color=LIGHT_BLUE, marker='s', markersize=15, label="'Cell' Obstacles")
        DARK_GREEN_square = mlines.Line2D([], [], color=DARK_GREEN, marker='s', markersize=15, label="'Point' Charging Stations")
        LIGHT_GREEN_square = mlines.Line2D([], [], color=LIGHT_GREEN, marker='s', markersize=15, label="'Cell' Charging Stations")
        GOLD_circle = mlines.Line2D([], [], color=GOLD, marker='o', markersize=15, label="Users")

        # The following 'magic' number represent the RGBA values for charging stations and obstacles: 
        cs_cells_colors = [(0.4, 1, 0.59, 0.3) for i in range(N_CS)]
        obs_cells_colors = [(0.4, 1, 1, 0.3) for i in range(len(obs_cells))]

        bottom = 0
        width = 1
        depth = 1

        # ______________________________________________________________ FIGURE for the animation: ______________________________________________________________

        if path_animation == True: 
            
            if (CREATE_ENODEB == True):
                colors = [WHITE, LIGHT_BLUE, LIGHT_GREEN, LIGHT_RED]
            else:
                if (DIMENSION_2D == False):
                    colors = [WHITE, LIGHT_BLUE, LIGHT_GREEN]
                else:
                    colors = [WHITE, LIGHT_GREEN]
            cmap = ListedColormap(colors)

            fig = plt.figure('Cells')

            if (DIMENSION_2D == False):
                ax = fig.add_subplot(111, projection='3d')
                #ax = p3.Axes3D(fig)
                ax.view_init(elev=60, azim=40)
                if (UNLIMITED_BATTERY == True):
                    cells_status_matrix_un_bat = [[FREE if cells_status_matrix[r][c]==CS_IN else cells_status_matrix[r][c] for c in range(N_cells_col)] for r in range(N_cells_row)]
                    cells_status_matrix = cells_status_matrix_un_bat
            else:
                ax = fig.add_subplot(111)
                if (UNLIMITED_BATTERY == True):
                    cells_status_matrix_2D = [[FREE if cells_status_matrix[r][c]==OBS_IN or cells_status_matrix[r][c]==CS_IN else cells_status_matrix[r][c] for c in range(N_cells_col)] for r in range(N_cells_row)]
                    cells_status_matrix = cells_status_matrix_2D
                else:
                    cells_status_matrix_2D = [[FREE if cells_status_matrix[r][c]==OBS_IN else cells_status_matrix[r][c] for c in range(N_cells_col)] for r in range(N_cells_row)]
                    cells_status_matrix = cells_status_matrix_2D

            users_x, users_y, users_z = self.extract_coord_from_xyz(users)
            users_x_for_2DplotCells, users_y_for_2DplotCells = [float(x)-0.5 for x in users_x], [float(y)-0.5 for y in users_y]
            users_x_for_3DplotCells, users_y_for_3DplotCells, users_z_for_3DplotCells = [x for x in users_x], [y for y in users_y], users_z
            num_clusters = len(centroids)
            
            x_obs_cells, y_obs_cells, z_obs_cells = self.extract_coord_from_xyz(obs_cells)
            x_cs_cells, y_cs_cells, z_cs_cells = self.extract_coord_from_xyz(cs_cells)

            if (CREATE_ENODEB == True):
                x_eNB_cells, y_eNB_cells, z_eNB_cells = self.extract_coord_from_xyz(enb_cells)

            if (DIMENSION_2D == False):
                ax.scatter(users_y_for_3DplotCells, users_x_for_3DplotCells, users_z_for_3DplotCells, s=10, c=GOLD)
                for cluster_idx in self.num_color_range:
                    patch = plt.Circle([centroids[cluster_idx][1]/CELL_RESOLUTION_PER_COL+0.25, centroids[cluster_idx][0]/CELL_RESOLUTION_PER_ROW+0.25, centroids[cluster_idx][2]], (float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_ROW)) + float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_COL)))/2, color=self.clusters_colors[cluster_idx], fill=False)
                    ax.add_patch(patch)
                    art3d.pathpatch_2d_to_3d(patch)
                    pass

                ax.bar3d(y_obs_cells, x_obs_cells, bottom, width, depth, z_obs_cells, shade=True, color=obs_cells_colors, edgecolor="none")
                if (UNLIMITED_BATTERY == False):
                    ax.bar3d(y_cs_cells, x_cs_cells, bottom, width, depth, z_cs_cells, shade=True, color=cs_cells_colors, edgecolor="none")
                if (CREATE_ENODEB == True):
                    ax.bar3d(y_eNB_cells, x_eNB_cells, bottom, width, depth, z_eNB_cells, shade=True, color=(0.5, 0, 0, 0.3), edgecolor="none")

                ax.set_xlim(xmin=0, xmax=CELLS_COLS)
                ax.set_ylim(ymin=CELLS_ROWS, ymax=0) # --> I want to set 0 in the bottom part of the 2D plane-grid.
                ax.set_zlim(zmin=0)
                ax.set_title('3D Animation')
            
            else:

                ax.imshow(cells_status_matrix, cmap=cmap)

                ax.set_xticks(np.arange(0, N_cells_col+1, 1)-0.5)
                ax.set_yticks(np.arange(0, N_cells_row+1, 1)-0.5)

                ax.set_xticklabels(np.arange(0, area_width+1, 1))
                ax.set_yticklabels(np.arange(0, area_height+1, 1))

                ax.grid(which='major')
                ax.scatter(users_x_for_2DplotCells, users_y_for_2DplotCells, s=10, c=GOLD)
                
                # A Graphical approximation is needed in order to get a cluster in 'cells view' which is as closest as possible to the one in 'points view' (The approximation is only graphical):
                for cluster_idx in self.num_color_range:
                    [ax.add_artist(plt.Circle([centroids[cluster_idx][0]/CELL_RESOLUTION_PER_ROW-0.25, centroids[cluster_idx][1]/CELL_RESOLUTION_PER_COL-0.25, centroids[cluster_idx][2]], (float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_ROW)) + float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_COL)))/2, color=self.clusters_colors[cluster_idx], fill=False)) for cluster_idx in self.num_color_range]

                ax.set_xlim(xmin=0-0.5, xmax=CELLS_COLS+0.5)
                ax.set_ylim(ymin=CELLS_ROWS+0.5, ymax=0-0.5)
                ax.set_title('2D Animation')


            if (CREATE_ENODEB == True):
                fig.legend(handles=[LIGHT_BLUE_square, LIGHT_GREEN_square, LIGHT_RED_square, GOLD_circle])
            else:
                if (DIMENSION_2D == False):
                    ax.set_xlim(xmin=0, xmax=CELLS_COLS)
                    ax.set_ylim(ymin=0, ymax=CELLS_ROWS)
                    if (UNLIMITED_BATTERY == True):
                        fig.legend(handles=[LIGHT_BLUE_square, GOLD_circle])
                    else:
                        fig.legend(handles=[LIGHT_BLUE_square, LIGHT_GREEN_square, GOLD_circle])
                else:
                    if (UNLIMITED_BATTERY == True):
                        fig.legend(handles=[GOLD_circle])    
                    else:
                        fig.legend(handles=[LIGHT_GREEN_square, GOLD_circle])

            data_path = []
            
            for path in agents_paths:
                if (DIMENSION_2D == False):
                    path_x, path_y, path_z = [np.array(coords[0]) for coords in path], [np.array(coords[1]) for coords in path], [np.array(coords[2]) for coords in path]
                    data_path.append([path_x, path_y, path_z])
                else:
                    path_x, path_y = [np.array(coords[0]) for coords in path], [np.array(coords[1]) for coords in path]
                    data_path.append([path_x, path_y])

            data_path = np.array(data_path)  

            lines = []
            circles = []
            uav_color_count = 0

            if (DIMENSION_2D == False):
                x = ACTUAL_UAV_FOOTPRINT * np.outer(np.cos(u), np.sin(v))
                y = ACTUAL_UAV_FOOTPRINT * np.outer(np.sin(u), np.sin(v))
                z = 0 * np.outer(np.ones(np.size(u)), np.cos(v))
                for path in data_path:
                    lines.append(ax.plot(path[1, 0:1], path[0, 0:1], path[2, 0:1], color=UAVS_COLORS[uav_color_count])[0])
                    circles.append(ax.plot_surface(x, y, z, color=UAVS_COLORS[uav_color_count], linewidth=0))
                    uav_color_count += 1
            else:
                for path in data_path:
                    lines.append(ax.plot(path[0, 0:1], path[1, 0:1], color=UAVS_COLORS[uav_color_count])[0])
                    circles.append(plt.Circle(xy=(path[0, 0:1], path[1, 0:1]), radius=ACTUAL_UAV_FOOTPRINT, color=UAVS_COLORS[uav_color_count], fill=True, alpha=0.18))
                    uav_color_count += 1
                for patch in circles:
                    ax.add_patch(patch)

            if (DIMENSION_2D == False):
                n_circles_range = range(len(circles))
                ani = animation.FuncAnimation(fig, self.update_animation_3D, frames=ITERATIONS_PER_EPISODE-1, fargs=(data_path, lines, circles, n_circles_range, ax), interval=100, blit=True, repeat=True) # fargs=(data_path, lines, circles)
            else:
                ani = animation.FuncAnimation(fig, self.update_animation_2D, frames=ITERATIONS_PER_EPISODE-1, fargs=(data_path, lines, circles), interval=100, blit=True, repeat=True)

            if (DIMENSION_2D==False):
                ax.set_zlim(zmin=0)
            ani.save(join(where_to_save, "animation_ep" + str(episode) + ".gif"), writer='imagemagick')

            plt.close(fig)

        # ______________________________________________________________ FIGURES FOR STATIC ENVIRNONMENT VISUALIZATION (WITHOUT UAVS): ______________________________________________________________ 

        else:

            num_clusters = len(centroids)

            x_obs_points, y_obs_points, z_obs_points = self.extract_coord_from_xyz(obs_points)
            x_cs_points, y_cs_points, z_cs_points = self.extract_coord_from_xyz(cs_points)

            if (CREATE_ENODEB == True):
                x_enb_point, y_enb_point, z_enb_point = self.extract_coord_from_xyz(enb_point)
            x_obs_cells, y_obs_cells, z_obs_cells = self.extract_coord_from_xyz(obs_cells)
            
            x_cs_cells, y_cs_cells, z_cs_cells = self.extract_coord_from_xyz(cs_cells)
            if (CREATE_ENODEB == True):
                x_eNB_cells, y_eNB_cells, z_eNB_cells = self.extract_coord_from_xyz(enb_cells)

            users_x, users_y, users_z = self.extract_coord_from_xyz(users)
            users_x_for_2Dplot, users_y_for_2Dplot = [float(x)-0.5 for x in users_x], [float(y)-0.5 for y in users_y]
            users_x_for_3Dplot, users_y_for_3Dplot, users_z_for_3Dplot = users_x, users_y, users_z
            users_x_for_2DplotCells, users_y_for_2DplotCells = [float(x)/CELL_RESOLUTION_PER_ROW-0.5 for x in users_x], [float(y)/CELL_RESOLUTION_PER_COL-0.5 for y in users_y]
            users_x_for_3DplotCells, users_y_for_3DplotCells, users_z_for_3DplotCells = [float(x)/CELL_RESOLUTION_PER_ROW for x in users_x], [float(y)/CELL_RESOLUTION_PER_COL for y in users_y], users_z

            # Redefine cells in such a way to have the right plot visualization:
            x_obs_cells_for_2Dplot = [elem*CELL_RESOLUTION_PER_COL for elem in x_obs_cells]
            y_obs_cells_for_2Dplot = [elem*CELL_RESOLUTION_PER_ROW for elem in y_obs_cells]
            x_cs_cells_for_2Dplot = [elem*CELL_RESOLUTION_PER_COL for elem in x_cs_cells]
            y_cs_cells_for_2Dplot = [elem*CELL_RESOLUTION_PER_ROW for elem in y_cs_cells]
            if (CREATE_ENODEB == True):
                x_eNB_cells_for_2Dplot = [elem*CELL_RESOLUTION_PER_COL for elem in x_eNB_cells]
                y_eNB_cells_for_2Dplot = [elem*CELL_RESOLUTION_PER_ROW for elem in y_eNB_cells]

            # FIGURE 1 (Points 'point of view'):
            fig1 = plt.figure('Points')

            if (DIMENSION_2D == False):
                ax1 = fig1.add_subplot(121)
                ax2 = fig1.add_subplot(122, projection='3d')
                if (UNLIMITED_BATTERY == True):
                    points_status_matrix_un_bat = [[FREE if points_status_matrix[r][c]==CS_IN else points_status_matrix[r][c] for c in range(area_width)] for r in range(area_height)]
                    cells_status_matrix_un_bat = [[FREE if cells_status_matrix[r][c]==CS_IN else cells_status_matrix[r][c] for c in range(N_cells_col)] for r in range(N_cells_row)]
                    perceived_status_matrix_un_bat = [[FREE if perceived_status_matrix[r][c]==CS_IN else perceived_status_matrix[r][c] for c in range(area_width)] for r in range(area_height)]
                    points_status_matrix = points_status_matrix_un_bat
                    cells_status_matrix = cells_status_matrix_un_bat
                    perceived_status_matrix = perceived_status_matrix_un_bat
            else:
                ax1 = fig1.add_subplot(111)
                if (UNLIMITED_BATTERY == True):
                    points_status_matrix_un_bat = [[FREE if points_status_matrix[r][c]==OBS_IN or points_status_matrix[r][c]==CS_IN else points_status_matrix[r][c] for c in range(area_width)] for r in range(area_height)]
                    cells_status_matrix_un_bat = [[FREE if cells_status_matrix[r][c]==OBS_IN or cells_status_matrix[r][c]==CS_IN else cells_status_matrix[r][c] for c in range(N_cells_col)] for r in range(N_cells_row)]
                    perceived_status_matrix_un_bat = [[FREE if perceived_status_matrix[r][c]==OBS_IN or perceived_status_matrix[r][c]==CS_IN else perceived_status_matrix[r][c] for c in range(area_width)] for r in range(area_height)]
                    points_status_matrix = points_status_matrix_un_bat
                    cells_status_matrix = cells_status_matrix_un_bat
                    perceived_status_matrix = perceived_status_matrix_un_bat
                else:
                    points_status_matrix_2D = [[FREE if points_status_matrix[r][c]==OBS_IN else points_status_matrix[r][c] for c in range(area_width)] for r in range(area_height)]
                    cells_status_matrix_2D = [[FREE if cells_status_matrix[r][c]==OBS_IN else cells_status_matrix[r][c] for c in range(N_cells_col)] for r in range(N_cells_row)]
                    perceived_status_matrix_2D = [[FREE if perceived_status_matrix[r][c]==OBS_IN else perceived_status_matrix[r][c] for c in range(area_width)] for r in range(area_height)]
                    points_status_matrix = points_status_matrix_2D
                    cells_status_matrix = cells_status_matrix_2D
                    perceived_status_matrix = perceived_status_matrix_2D

            if (CREATE_ENODEB == True):
                colors1 = [WHITE, DARK_BLUE, DARK_GREEN, DARK_RED]
            else:
                if (DIMENSION_2D == False):
                    if (UNLIMITED_BATTERY == True):
                        colors1 = [WHITE, DARK_BLUE]
                    else:
                        colors1 = [WHITE, DARK_BLUE, DARK_GREEN]
                else:
                    if (UNLIMITED_BATTERY == True):
                        colors1 = [WHITE]
                    else:
                        colors1 = [WHITE, DARK_GREEN]
            cmap1 = ListedColormap(colors1)

            ax1.imshow(points_status_matrix, cmap=cmap1) # Here the transpose is used because the first argument of 'imshow' take (M,N) where 'M' are the rows and 'N' are the columns (while we store them in the form (x,y) where 'x' are the columns and 'y' are the rows)
            ax1.set_xticks(np.arange(0, area_width+1, 1)-0.5)
            ax1.set_yticks(np.arange(0, area_height+1, 1)-0.5)
            ax1.set_xticklabels(np.arange(0, area_width+1, 1))
            ax1.set_yticklabels(np.arange(0, area_height+1, 1))
            ax1.grid(which='both')
            ax1.scatter(users_x_for_2Dplot, users_y_for_2Dplot, s=10, c=GOLD)
            
            for cluster_idx in self.num_color_range:
                [ax1.add_artist(plt.Circle([centroids[cluster_idx][0], centroids[cluster_idx][1]], float(clusters_radiuses[cluster_idx]), color=self.clusters_colors[cluster_idx], fill=False)) for cluster_idx in self.num_color_range]
            ax1.set_title('2D Points-Map')

            if (DIMENSION_2D == False):
                ax2.scatter(users_y_for_3Dplot, users_x_for_3Dplot, users_z_for_3Dplot, s=10, c=GOLD)
                
                for cluster_idx in self.num_color_range:
                    patch = plt.Circle([centroids[cluster_idx][1]+incr_assumed_coord, centroids[cluster_idx][0]+incr_assumed_coord, centroids[cluster_idx][2]], float(clusters_radiuses[cluster_idx]), color=self.clusters_colors[cluster_idx], fill=False)
                    ax2.add_patch(patch)
                    art3d.pathpatch_2d_to_3d(patch)        
                
                ax2.bar3d(y_obs_points, x_obs_points, bottom, width, depth, z_obs_points, shade=True, color=(0, 0, 0.6), edgecolor="none")
                
                if (UNLIMITED_BATTERY == False):
                    cs_colors = [(0, 0.4, 0) for cs in range(N_CS)]
                    ax2.bar3d(y_cs_points, x_cs_points, bottom, width, depth, z_cs_points, shade=True, color=cs_colors, edgecolor="none")
                if (CREATE_ENODEB == True):
                    ax2.bar3d(y_enb_point, x_enb_point, bottom, width, depth, z_enb_point, shade=True, color=(0.5, 0, 0), edgecolor="none")
                
                ax2.set_xlim(xmin=0, xmax=CELLS_COLS)
                ax2.set_ylim(ymin=0, ymax=CELLS_ROWS)
                ax2.set_zlim(zmin=0)
                ax2.set_title('3D Points-Map')

            if (CREATE_ENODEB == True):
                fig1.legend(handles=[DARK_BLUE_square, DARK_GREEN_square, DARK_RED_square, GOLD_circle])
            else:
                if (DIMENSION_2D == False):
                    if (UNLIMITED_BATTERY == True):
                        fig1.legend(handles=[DARK_BLUE_square, GOLD_circle])
                    else:
                        fig1.legend(handles=[DARK_BLUE_square, DARK_GREEN_square, GOLD_circle])
                else:
                    if (UNLIMITED_BATTERY == True):
                        fig1.legend(handles=[GOLD_circle])                        
                    else:
                        fig1.legend(handles=[DARK_GREEN_square, GOLD_circle])

            plt.savefig(join(env_directory, "Minimum_Resolution.png"))

            # FIGURE 2 (Cells 'point of view'):
            fig2 = plt.figure('Cells')

            if (DIMENSION_2D == False):
                ax3 = fig2.add_subplot(121)
                ax4 = fig2.add_subplot(122, projection='3d')
            else:
                ax3 = fig2.add_subplot(111)

            if (CREATE_ENODEB == True):
                colors2 = [WHITE, LIGHT_BLUE, LIGHT_GREEN, LIGHT_RED]
            else:
                if (DIMENSION_2D == False):
                    if (UNLIMITED_BATTERY == True):
                        colors2 = [WHITE, LIGHT_BLUE]
                    else:    
                        colors2 = [WHITE, LIGHT_BLUE, LIGHT_GREEN]
                else:
                    if (UNLIMITED_BATTERY == True):
                        colors2 = [WHITE, LIGHT_BLUE]
                    else:
                        colors2 = [WHITE, LIGHT_BLUE, LIGHT_GREEN]
            cmap2 = ListedColormap(colors2)

            ax3.imshow(cells_status_matrix, cmap=cmap2)
            ax3.set_xticks(np.arange(0, N_cells_col+1, 1)-0.5)
            ax3.set_yticks(np.arange(0, N_cells_row+1, 1)-0.5)
            ax3.set_xticklabels(np.arange(0, area_width+1, 1))
            ax3.set_yticklabels(np.arange(0, area_height+1, 1))
            ax3.grid(which='major')
            ax3.scatter(users_x_for_2DplotCells, users_y_for_2DplotCells, s=10, c=GOLD)
            
            # A Graphical approximation is needed in order to get a cluster in 'cells view' which is as closest as possible to the one in 'points view' (The approximation is only graphical):
            for cluster_idx in self.num_color_range:
                [ax3.add_artist(plt.Circle([centroids[cluster_idx][0]/CELL_RESOLUTION_PER_ROW-0.25, centroids[cluster_idx][1]/CELL_RESOLUTION_PER_COL-0.25, centroids[cluster_idx][2]], (float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_ROW)) + float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_COL)))/2, color=self.clusters_colors[cluster_idx], fill=False)) for cluster_idx in self.num_color_range]
            ax3.set_title('2D Cells-Map')

            if (DIMENSION_2D == False):
                ax4.scatter(users_y_for_3DplotCells, users_x_for_3DplotCells, users_z_for_3DplotCells, s=10, c=GOLD)
                for cluster_idx in self.num_color_range:
                    patch = plt.Circle([centroids[cluster_idx][1]/CELL_RESOLUTION_PER_COL+0.25, centroids[cluster_idx][0]/CELL_RESOLUTION_PER_ROW+0.25, centroids[cluster_idx][2]], (float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_ROW)) + float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_COL)))/2, color=self.clusters_colors[cluster_idx], fill=False)
                    ax4.add_patch(patch)
                    art3d.pathpatch_2d_to_3d(patch)
                    pass
                
                ax4.bar3d(y_obs_cells, x_obs_cells, bottom, width, depth, z_obs_cells, shade=True, color=(0.4, 1, 1), edgecolor="none")
                
                if (UNLIMITED_BATTERY == False):
                    cs_cells_colors = [(0.4, 1, 0.59) for cs in range(N_CS)]
                    ax4.bar3d(y_cs_cells, x_cs_cells, bottom, width, depth, z_cs_cells, shade=True, color=cs_cells_colors, edgecolor="none")
                if (CREATE_ENODEB == True):
                    ax4.bar3d(y_eNB_cells, x_eNB_cells, bottom, width, depth, z_eNB_cells, shade=True, color=(0.5, 0, 0), edgecolor="none")
                
                ax4.set_xlim(xmin=0, xmax=CELLS_COLS)
                ax4.set_ylim(ymin=0, ymax=CELLS_ROWS)
                ax4.set_zlim(zmin=0)
                ax4.set_title('3D Cells-Map')

            if (CREATE_ENODEB == True):
                fig2.legend(handles=[LIGHT_BLUE_square, LIGHT_GREEN_square, LIGHT_RED_square, GOLD_circle])
            else:
                if (DIMENSION_2D == False):
                    if (UNLIMITED_BATTERY == True):
                        fig2.legend(handles=[LIGHT_BLUE_square, GOLD_circle])
                    else:
                        fig2.legend(handles=[LIGHT_BLUE_square, LIGHT_GREEN_square, GOLD_circle])
                else:
                    if (UNLIMITED_BATTERY == True):
                        fig2.legend(handles=[GOLD_circle])
                    else:
                        fig2.legend(handles=[LIGHT_GREEN_square, GOLD_circle])

            plt.savefig(join(env_directory, "Desired_Resolution.png"))

            # FIGURE 3 (Mixed 'point of view'):
            if ( (area_height != N_cells_row) and (area_width != N_cells_col) ):

                fig3 = plt.figure('Points and Cells')

                if (DIMENSION_2D == False):
                    ax5 = fig3.add_subplot(121)
                    ax6 = fig3.add_subplot(122, projection='3d')
                else:
                    ax5 = fig3.add_subplot(111)
                
                ax5.imshow(points_status_matrix, cmap=cmap1)
                ax5.imshow(perceived_status_matrix, cmap=cmap2, alpha=0.5)
                ax5.set_xticks(np.arange(0, area_width+1, CELL_RESOLUTION_PER_COL)-0.5)
                ax5.set_xticks(np.arange(0, area_width+1, 1)-0.5, minor=True)
                ax5.set_yticks(np.arange(0, area_height+1, CELL_RESOLUTION_PER_ROW)-0.5)
                ax5.set_yticks(np.arange(0, area_height+1, 1)-0.5, minor=True)
                ax5.set_xticklabels(np.arange(0, area_width+1, CELL_RESOLUTION_PER_COL))
                ax5.set_yticklabels(np.arange(0, area_width+1, CELL_RESOLUTION_PER_ROW))
                ax5.grid(which='minor', alpha=0.2)
                ax5.grid(which='major', alpha=0.5)
                ax5.scatter(users_x_for_2Dplot, users_y_for_2Dplot, s=10, c=GOLD)
                
                for cluster_idx in self.num_color_range:
                    [ax5.add_artist(plt.Circle([centroids[cluster_idx][0], centroids[cluster_idx][1]], float(clusters_radiuses[cluster_idx]), color=self.clusters_colors[cluster_idx], fill=False)) for cluster_idx in self.num_color_range]
                ax5.set_title('2D Points/Cells-Map')

                if (DIMENSION_2D == False):
                    ax6.scatter(users_y_for_3Dplot, users_x_for_3Dplot, users_z_for_3Dplot, s=10, c=GOLD)
                    for cluster_idx in self.num_color_range:
                        patch = plt.Circle([centroids[cluster_idx][1]+incr_assumed_coord, centroids[cluster_idx][0]+incr_assumed_coord, centroids[cluster_idx][2]], float(clusters_radiuses[cluster_idx]), color=self.clusters_colors[cluster_idx], fill=False)
                        ax6.add_patch(patch)
                        art3d.pathpatch_2d_to_3d(patch)
                    ax6.bar3d(y_obs_points, x_obs_points, bottom, width, depth, z_obs_points, shade=True, color=(0, 0, 0.6), edgecolor="none")
                    if (UNLIMITED_BATTERY == False):
                        ax6.bar3d(y_cs_points, x_cs_points, bottom, width, depth, z_cs_points, shade=True, color=(0, 0.4, 0), edgecolor="none")
                    if (CREATE_ENODEB == True):
                        ax6.bar3d(y_enb_point, x_enb_point, bottom, width, depth, z_enb_point, shade=True, color=(1, 0, 0), edgecolor="none")
                    ax6.bar3d(y_obs_cells_for_2Dplot, x_obs_cells_for_2Dplot, bottom, CELL_RESOLUTION_PER_COL, CELL_RESOLUTION_PER_ROW, z_obs_cells, shade=True, color=obs_cells_colors, edgecolor="none")
                    if (UNLIMITED_BATTERY == False):
                        ax6.bar3d(y_cs_cells_for_2Dplot, x_cs_cells_for_2Dplot, bottom, CELL_RESOLUTION_PER_COL, CELL_RESOLUTION_PER_ROW, z_cs_cells, shade=True, color=cs_cells_colors, edgecolor="none")
                    if (CREATE_ENODEB == True):
                        ax6.bar3d(y_eNB_cells_for_2Dplot, x_eNB_cells_for_2Dplot, bottom, CELL_RESOLUTION_PER_COL, CELL_RESOLUTION_PER_ROW, z_eNB_cells, shade=True, color=(1, 0, 0, 0.3), edgecolor="none")
                    ax6.set_xlim(xmin=0, xmax=CELLS_COLS)
                    ax6.set_ylim(ymin=0, ymax=CELLS_ROWS)
                    ax6.set_zlim(zmin=0)
                    ax6.set_title('3D Points/Cells-Map')

                if (CREATE_ENODEB == True):
                    fig3.legend(handles=[DARK_BLUE_square, LIGHT_BLUE_square, DARK_GREEN_square, LIGHT_GREEN_square, DARK_RED_square, LIGHT_RED_square, GOLD_circle])
                else:
                    if (DIMENSION_2D == False):
                        if (UNLIMITED_BATTERY == True):
                            fig3.legend(handles=[DARK_BLUE_square, LIGHT_BLUE_square, GOLD_circle])
                        else:
                            fig3.legend(handles=[DARK_BLUE_square, LIGHT_BLUE_square, DARK_GREEN_square, LIGHT_GREEN_square, GOLD_circle])
                    else:
                        if (UNLIMITED_BATTERY == True):
                            fig3.legend(handles=[GOLD_circle])
                        else:
                            fig3.legend(handles=[DARK_GREEN_square, LIGHT_GREEN_square, GOLD_circle])

            if ( (area_height!=N_cells_row) and (area_width!=N_cells_col) ):
                plt.savefig(join(env_directory, "Mixed_Resolution.png"))

            plt.show()

    def QoE_plot(self, parameter_values, epochs, where_to_save, param_name):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Save the image representing QoEs parameters in 'parameters_values' in 'where_to_save'.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        fig = plt.figure(param_name)
        ax = fig.add_subplot(111)
        plt.xlabel('Epochs')
        plt.ylabel(param_name + ' Trend')
        plt.title(param_name)
        legend_labels = []

        parameter_samples = []
        end = PLOT_EACH_N_EPOCH
        for start in range(0, EPISODES, PLOT_EACH_N_EPOCH):
            current_sample = parameter_values[start:end]
            parameter_samples.append(current_sample)
            start = end
            end = start+PLOT_EACH_N_EPOCH

        # Standard Deviation:
        stds = [stdev(params) for params in parameter_samples]
        stds.insert(0, 0.0)

        last_value = parameter_values[-1]
        parameter_values = parameter_values[::PLOT_EACH_N_EPOCH]
        parameter_values.append(last_value)
        plt.errorbar(RANGE_EPOCHS_TO_PLOT, parameter_values, yerr=stds)
        
        if (STATIC_REQUEST==False):
            legend_labels.append('Starting Epoch for Users moving')
        for user_epoch_move in range(MOVE_USERS_EACH_N_EPOCHS, epochs+1, MOVE_USERS_EACH_N_EPOCHS):
            if (STATIC_REQUEST==False):
                plt.axvline(x=user_epoch_move, color='green') #, label='Starting Epoch for Users moving')
        
        legend_labels.append('QoE Parameter and related std')
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
        ax.set_xticks(RANGE_X_TICKS)
        ax.set_xticklabels(RANGE_EPOCHS_TO_VISUALIZE)
        plt.legend(legend_labels)
        plt.savefig(where_to_save + '.png')

    def users_covered_percentage_per_service(self, services_values, epochs, where_to_save):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
        # Save the image representing the percentage of covered percentage for each service in 'service_values' in 'where_to_save'. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        fig = plt.figure("Services")
        ax = fig.add_subplot(111)
        plt.xlabel('Epochs')
        plt.ylabel('Users Served Percentage')
        plt.title("Services Provision")

        legend_labels = []
        for service_idx in range(N_SERVICES):
            if (service_idx==0):
                service_label = "Throughput Service"
            elif (service_idx==1):
                service_label = "Edge Computing"
            elif (service_idx==2):
                service_label = "Data Gathering"

            last_values = services_values[-1]
            all_values = services_values[::PLOT_EACH_N_EPOCH]
            all_values.append(last_values)
            values_to_plot = [value[service_idx] for value in all_values]

            plt.plot(RANGE_EPOCHS_TO_PLOT, values_to_plot)
            legend_labels.append(service_label)

        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
        ax.set_xticks(RANGE_X_TICKS)
        ax.set_xticklabels(RANGE_EPOCHS_TO_VISUALIZE)
        plt.legend(legend_labels)
        plt.savefig(where_to_save + '.png')

    def UAVS_reward_plot(self, epochs, UAVs_rewards, directory_name, q_values=False):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
        # Save the image representing the percentage of covered percentage for each service in service_values in 'directory_name'.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        UAV_ID = 0
        if (q_values==False):
            fig = plt.figure('UAVs rewards')
            plt.ylabel ('UAVs Rewards')
            plt.title('Rewards')
        else:
            fig = plt.figure('UAVs Q-values')
            plt.ylabel('UAVs Q-values')
            plt.title('Q-values')
        plt.xlabel('Epochs')

        ax = fig.add_subplot(111)

        legend_labels = []
        for UAV in UAVs_rewards:
            rewards = [reward for reward in UAV]
            last_value = rewards[-1]
            rewards_to_plot = rewards[::PLOT_EACH_N_EPOCH]
            rewards_to_plot.append(last_value)
            plt.plot(RANGE_EPOCHS_TO_PLOT, rewards_to_plot, color=UAVS_COLORS[UAV_ID])
            UAV_ID += 1
            legend_labels.append('UAV' + str(UAV_ID))
        
        ax.set_xlim(xmin=1)
        ax.set_ylim(ymin=0)
        ax.set_xticks(RANGE_X_TICKS)
        ax.set_xticklabels(RANGE_EPOCHS_TO_VISUALIZE)
        plt.legend(legend_labels)
        if (q_values==False):
            plt.savefig(join(directory_name, "Rewards_per_epoch_UAVs.png"))
        else:
            plt.savefig(join(directory_name, "Q-values_per_epoch_UAVs.png"))

    def epsilon(self, epsilon_history, epochs, directory_name):
        # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Save the image representing 'epslilon_history' in 'dericetory_name'.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # 

        epochs_to_plot = range(1, epochs+1)

        fig = plt.figure()

        ax = fig.add_subplot(111)
        
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Epsilon Value")

        ax.plot(epochs_to_plot, epsilon_history)
        
        ax.set_xlim(xmin=1)
        ax.set_ylim(ymin=0)
        plt.savefig(join(directory_name, "Epsilon_per_epoch.png"))

    def actions_min_max_per_epoch(self, q_table, directory_name, episode, which_uav):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Save the image representing the MIN-MAX values per action (per epoch) in 'directory_name'.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        fig = plt.figure()
        ax = fig.add_subplot(111)

        actual_q_table = q_table[which_uav-1]

        state_count = 0
        for state in actual_q_table:
            action_values = actual_q_table[state]
            max_action_value = max(action_values)
            min_action_value = min(action_values)
            ax.scatter(state_count, DICT_ACTION_SPACE[action_values.index(max_action_value)], c="green", marker="o", alpha=0.4)
            ax.scatter(state_count, DICT_ACTION_SPACE[action_values.index(min_action_value)], c="red", marker="o", alpha=0.4)
            state_count += 1
        
        ax.set_xlabel("States")
        ax.set_ylabel("Actions")
        
        plt.savefig(directory_name + f"\qtable_graph-ep{episode}.png")

    def users_wait_times(self, n_users, users, directory_name, episode):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Save the image representing the waiting time of the users related to the service provision in 'directory_name'. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        fig = plt.figure()
        ax = fig.add_subplot(111)

        wait_times = [user._info[3] for user in users]

        ax.bar(np.arange(n_users), wait_times, align='center')
        ax.set_ylabel('Elapsed time between service request and service')
        ax.set_xlabel('Users Number')
        ax.set_title('QoE2 histogram distribution among users')

        plt.savefig(join(directory_name, "UsersWaitingTimes" + str(episode) + ".png"))

    def bandwidth_for_each_epoch(self, epochs, directory_name, UAVs_used_bandwidth, users_bandwidth_request_per_UAVfootprint):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Save the image representing the bandwidth usage in 'directory_name'.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   

        fig = plt.figure()

        ax = fig.add_subplot(111)
        
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Bandwidth Usage")

        range_n_uavs = range(N_UAVS)
        legend_labels = []
        for uav_idx in range_n_uavs:
            bandwidths = UAVs_used_bandwidth[uav_idx][::PLOT_EACH_N_EPOCH]
            last_value = UAVs_used_bandwidth[uav_idx][-1]
            bandwidths.append(last_value)
            ax.plot(RANGE_EPOCHS_TO_PLOT, bandwidths)
            last_value = users_bandwidth_request_per_UAVfootprint[uav_idx][-1]
            bandwidth_requests = users_bandwidth_request_per_UAVfootprint[uav_idx][::PLOT_EACH_N_EPOCH]
            bandwidth_requests.append(last_value)
            ax.scatter(RANGE_EPOCHS_TO_PLOT, bandwidth_requests)
            legend_labels.append('UAV' + str(uav_idx+1))
        
        for uav_idx in range_n_uavs:
            legend_labels.append('Actual request for UAV' + str(uav_idx+1))

        ax.set_xlim(xmin=1)
        ax.set_ylim(ymin=0)
        ax.set_xticks(RANGE_X_TICKS)
        ax.set_xticklabels(RANGE_EPOCHS_TO_VISUALIZE)
        plt.legend(legend_labels)
        plt.savefig(join(directory_name, "UAVsBandiwidth_per_epoch.png"))

    def battery_when_start_to_charge(self, battery_history, directory_name):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
        # Save the image representing the battery level of a UAV when it starts to recharge in 'directory_name'.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        n_recharges_per_uav = [range(1, len(battery_history[uav])+1) for uav in range(N_UAVS)]
        max_recharges_per_uav = max([len(battery_history[uav]) for uav in range(N_UAVS)])
        UAV_ID = 0

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.set_xlabel("N-th Start of Charging (every " + str(SHOW_BATTERY_LEVEL_FOR_CHARGING_INSTANT) + " charges)")
        ax.set_ylabel("Battery Level")
        plt.title('Battery level when start to charge')

        for UAV in n_recharges_per_uav:
            ax.bar(n_recharges_per_uav[UAV_ID], [self.battery_percentage(battery_levels) for battery_levels in battery_history[UAV_ID]], color=UAVS_COLORS[UAV_ID], align='center')
            UAV_ID += 1

            ax.set_xlim(xmin=0)
            ax.set_ylim(ymin=0)
            ax.set(xticks=range(0, max_recharges_per_uav+1), xticklabels=range(0, max_recharges_per_uav*SHOW_BATTERY_LEVEL_FOR_CHARGING_INSTANT+1, SHOW_BATTERY_LEVEL_FOR_CHARGING_INSTANT))
            
            plt.xticks(range(0, max_recharges_per_uav+1))
            plt.tick_params(labelbottom=False)
            plt.yticks(range(0, self.battery_percentage(CRITICAL_BATTERY_LEVEL), 5))
            
            plt.savefig(join(directory_name[UAV_ID-1], "Battery_level_when_start_charge_UAV" + str(UAV_ID)) + ".png")

    def UAVS_crashes(self, epochs, UAVs_crashes, directory_name):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Save the image representing the number of UAVs crashes in 'directory_name'. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        epochs_to_plot = range(1, epochs+1)
        fig = plt.figure('UAVs crashes')
        ax = fig.add_subplot(111)

        plt.xlabel('Epochs')
        plt.ylabel('UAV ID')
        plt.title('UAVs Crashes')

        legend_labels = []
        n_labels = len(legend_labels)
        UAV_ID = 0
        
        # The following commented condition can be useful in case I want to plot before the end of the training (just to check something):
        #if (UAVs_crashes[episode-1]==0):
            #break

        for uav_idx in range(N_UAVS):
            current_uav_crashes = [ep_idx for ep_idx in epochs_to_plot if UAVs_crashes[ep_idx-1][uav_idx]==True]
            UAV_ID = uav_idx+1
            ax.scatter(current_uav_crashes, [UAV_ID for elem in range(len(current_uav_crashes))], color=UAVS_COLORS[uav_idx], marker="x")
            legend_labels.append('UAV' + str(UAV_ID))

        ax.set_xlim(xmin=1, xmax=epochs+1)
        ax.set_yticks(np.arange(1, N_UAVS+1))
        ax.set_yticklabels(np.arange(1, N_UAVS+1))
        
        plt.legend(legend_labels)
        plt.savefig(join(directory_name, "UAVs_crashes.png"))

    def battery_percentage(self, battery_level):
        # # # # # # # # # # # # # # # # # # # # # #
        # Return the battery level in percentage. #
        # # # # # # # # # # # # # # # # # # # # # #

        percentage_battery_level = round((battery_level*100)/FULL_BATTERY_LEVEL)

        return percentage_battery_level

if __name__ == '__main__':
    
    plot = Plot()

    # ___________________________________________ Loading: ___________________________________________

    load = Loader()
    load.maps_data()
    load.users_clusters()

    obs_points = load.obs_points
    points_matrix = load._points_matrix
    cs_points = load.cs_points
    eNB_point = load.enb_point

    cells_matrix = load.cells_matrix
    obs_cells = load.obs_cells
    cs_cells = load.cs_cells
    eNB_cells = load.enb_cells

    initial_users = load.initial_users
    initial_centroids = load.initial_centroids
    initial_clusters_radiuses = load.initial_clusters_radiuses

    # _________________________________________________________________________________________________

    # ___________________________________________ Status Matrices Creation: ___________________________________________

    points_status_matrix = plot.compute_status_matrix(points_matrix, AREA_HEIGHT, AREA_WIDTH)
    cells_status_matrix = plot.compute_status_matrix(cells_matrix, CELLS_ROWS, CELLS_COLS)
    perceived_status_matrix = plot.compute_perceived_status_matrix(cells_matrix, AREA_HEIGHT, AREA_WIDTH, CELLS_ROWS, CELLS_COLS)

    # _________________________________________________________________________________________________________________

    # ___________________________________________ Directory Creation and Saving: ___________________________________________
    
    # Create directory 'MAP_STATUS_DIR' to save data:
    directory = Directories()
    directory.create_map_status_dir()
    
    # Saving:
    save = Saver()
    save.maps_status(points_status_matrix, cells_status_matrix, perceived_status_matrix)

    # _______________________________________________________________________________________________________________________

    # ___________________________________________ Plotting (with no animation): ___________________________________________

    agents_paths = [[(0,0,1), (1,0,1), (1,1,2), (1,1,3), (2,1,2)], [(0,0,1), (0,1,1), (1,1,0), (1,1,2), (1,2,3)]]
    plot.plt_map_views(obs_points, cs_points, eNB_point, obs_cells, cs_cells, eNB_cells, points_status_matrix, cells_status_matrix, perceived_status_matrix, initial_users, initial_centroids, initial_clusters_radiuses, AREA_HEIGHT, AREA_WIDTH, CELLS_ROWS, CELLS_COLS, agents_paths=None, path_animation=False)

    # _____________________________________________________________________________________________________________________
