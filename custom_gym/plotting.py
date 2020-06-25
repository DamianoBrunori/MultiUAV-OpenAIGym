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
from utils import *
from load_and_save_data import *
from scenario_objects import Point, Cell, User
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# If EnodeB has not been created in 'scenario_objets', then if you try to plot it, it will obviously raise an Error.

u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)

env_directory = "Environment_Views"
if not isdir(env_directory): mkdir(env_directory)

class Plot:
    '''
    |-------------------------------------------------------------|
    |Define a class containings method aimed to plot or to compute|
    |elements used to plot:                                       |
    |-------------------------------------------------------------|
    '''

    def __init__(self):
        pass

    def update_animation(self, num, dataLines, lines, circles, n_circles_range, ax): #, ax
        for line, data, circle_idx in zip(lines, dataLines, n_circles_range): #for line, data, circle in zip(lines, dataLines, circles):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_xdata(data[1, :num])
            line.set_ydata(data[0, :num])
            #line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2, :num])
            line.set_marker("4")
            line.set_markersize(16)
            #print("CIRCLES:", circles)
            #print("CIRCLES INDEX:", circle_idx)
            circles[circle_idx].remove()
            x =  data[1, num] + ACTUAL_UAV_FOOTPRINT * np.outer(np.cos(u), np.sin(v))
            y = data[0, num] + ACTUAL_UAV_FOOTPRINT * np.outer(np.sin(u), np.sin(v))
            z = 0 * np.outer(np.ones(np.size(u)), np.cos(v))
            surf = ax.plot_surface(x, y, z, color=UAVS_COLORS[circle_idx], alpha=0.18, linewidth=0)
            #print(type(surf))
            #surf._facecolors2d=surf._facecolors3d
            #surf._edgecolors2d=surf._edgecolors3d
            #ax.set_facecolor(UAVS_COLORS_RGB_PERCENTAGE[circle_idx] + (0.18,))
            #surf._edgecolors2d=surf._edgecolors3d
            circles[circle_idx] = surf
            #circle._segment3d[:num] = (data[0:2, :num])
            #print("CIRCLE SEGMENT:", circle._segment3d) 
            #print("LINEEEE", line)
            #print("CIRLCEE", circle)
            #circle._offsets3d(data[0:2, :num])
            #circle.set_3d_properties(data[2, :num])
            #print("UNOOOOOO", data[0:2, :num])
            #print("DUEEEEEE", data[2, :num])
            #print("PINOLOOOOOOO", (data[0][num], data[1][num]))
            #print("BEFORE", circle.center)
            #circle.center = (data[0][:num], data[1][:num])
            #print(circle.center)
            #circle.center = (data[0][num], data[1][num])
            #path = circle.get_path()
            #trans = circle.get_patch_transform()
            #mpath = trans.transform_path(path)
            #circle.set_3d_properties(mpath)
            #circle.set_3d_properties(data[0][:num], data[1][:num])
            #print("AFTER", circle.center)
        return tuple(lines) + tuple(circles)

    '''
    def update_animation(self, num, dataLines, lines):
        for line, data in zip(lines, dataLines):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2, :num])
            line.set_marker("o")
        return lines
    '''

    def update_animation_2D(self, num, dataLines, lines, circles):
        for line, data, circle in zip(lines, dataLines, circles):
            # NOTE: there is no .set_data() for 3 dim data...
            #print(line)
            line.set_data(data[0:2, :num])
            #line.set_3d_properties(data[2, :num])
            line.set_marker("4")
            line.set_markersize(16)
            #print("PINOLOOOOOO", num, (data[0][num], data[1][num]))
            #print(circle)
            circle.center = (data[0][num], data[1][num])
        return tuple(lines) + tuple(circles)

    '''
    def update_animation_2D(self, num, dataLines, lines):
        for line, data in zip(lines, dataLines):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, :num])
            #line.set_3d_properties(data[2, :num])
            line.set_marker("o")
        return lines
    '''

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

                print("CELL")
                print((current_cell._x_coord, current_cell._y_coord))
                for point in current_cell._points:
                    print("POINTS")
                    print((point._y_coord, point._x_coord))
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

        num_color_range = range(num_colors)
        colors = [None for color in num_color_range]
        
        for color_idx in num_color_range:
            colors[color_idx] = (random.randint(0, 255)/255, random.randint(0, 255)/255, random.randint(0, 255)/255, 1.0)

        return colors, num_color_range

    def plt_map_views(self, obs_points=None, cs_points=None, enb_point=None, obs_cells=None, cs_cells=None, enb_cells=None, points_status_matrix=None, cells_status_matrix=None, perceived_status_matrix=None, users=None, centroids=None, clusters_radiuses=None, area_height=None, area_width=None, N_cells_row=None, N_cells_col=None, agents_paths=None, path_animation=False, where_to_save=None, episode=None):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Create 3 figures which contains respectively:             #
        #   -   2D and 3D Points-Map;                               #
        #   -   2D and 3D Cells-Map;                                #
        #   -   2D and 3D Mixed-map (i.e., with Points and Cells);  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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

        bottom = 0
        width = 1
        depth = 1

        if path_animation == True: 
            
            if (CREATE_ENODEB == True):
                colors = [WHITE, LIGHT_BLUE, LIGHT_GREEN, LIGHT_RED]
            else:
                if (DIMENSION_2D == False):
                    colors = [WHITE, LIGHT_BLUE, LIGHT_GREEN]
                else:
                    colors = [WHITE, LIGHT_GREEN]
            cmap = ListedColormap(colors)

            # FIGURE for the animation:
            fig = plt.figure('Cells')
            #canvas = FigureCanvas(fig)

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
            clusters_colors, num_color_range = self.RGBA_01_random_colors(num_clusters)
            x_obs_cells, y_obs_cells, z_obs_cells = self.extract_coord_from_xyz(obs_cells)
            x_cs_cells, y_cs_cells, z_cs_cells = self.extract_coord_from_xyz(cs_cells)
            if (CREATE_ENODEB == True):
                x_eNB_cells, y_eNB_cells, z_eNB_cells = self.extract_coord_from_xyz(enb_cells)

            if (DIMENSION_2D == False):
                ax.scatter(users_y_for_3DplotCells, users_x_for_3DplotCells, users_z_for_3DplotCells, s=10, c=GOLD)
                for cluster_idx in num_color_range:
                    #patch = plt.Circle([centroids[cluster_idx][1]+0.5, centroids[cluster_idx][0]+0.5, centroids[cluster_idx][2]], float(clusters_radiuses[cluster_idx]), color=clusters_colors[cluster_idx], fill=False)
                    patch = plt.Circle([centroids[cluster_idx][1]/CELL_RESOLUTION_PER_COL+0.25, centroids[cluster_idx][0]/CELL_RESOLUTION_PER_ROW+0.25, centroids[cluster_idx][2]], (float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_ROW)) + float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_COL)))/2, color=clusters_colors[cluster_idx], fill=False)
                    ax.add_patch(patch)
                    art3d.pathpatch_2d_to_3d(patch)
                    pass
                ax.bar3d(y_obs_cells, x_obs_cells, bottom, width, depth, z_obs_cells, shade=True, color=(0.4, 1, 1, 0.3), edgecolor="none")
                if (UNLIMITED_BATTERY == False):
                    ax.bar3d(y_cs_cells, x_cs_cells, bottom, width, depth, z_cs_cells, shade=True, color=(0.4, 1, 0.59, 0.3), edgecolor="none")
                if (CREATE_ENODEB == True):
                    ax.bar3d(y_eNB_cells, x_eNB_cells, bottom, width, depth, z_eNB_cells, shade=True, color=(0.5, 0, 0, 0.3), edgecolor="none")

                ax.set_xlim(xmin=0, xmax=CELLS_COLS)
                ax.set_ylim(ymin=CELLS_ROWS, ymax=0) # --> I want to set 0 in the bottom part of the 2D plane-grid.
                ax.set_zlim(zmin=0)
                ax.set_title('3D Animation')
            
            else:

                ax.imshow(cells_status_matrix, cmap=cmap)
                #print(cells_status_matrix)

                ax.set_xticks(np.arange(0, N_cells_col+1, 1)-0.5)
                ax.set_yticks(np.arange(0, N_cells_row+1, 1)-0.5)

                ax.set_xticklabels(np.arange(0, area_width+1, 1))
                ax.set_yticklabels(np.arange(0, area_height+1, 1))

                ax.grid(which='major')
                ax.scatter(users_x_for_2DplotCells, users_y_for_2DplotCells, s=10, c=GOLD)
                # A Graphical approximation is needed in order to get a cluster in 'cells view' which is as closest as possible to the one in 'points view' (The approximation is only graphical):
                for cluster_idx in num_color_range:
                    [ax.add_artist(plt.Circle([centroids[cluster_idx][0]/CELL_RESOLUTION_PER_ROW-0.25, centroids[cluster_idx][1]/CELL_RESOLUTION_PER_COL-0.25, centroids[cluster_idx][2]], (float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_ROW)) + float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_COL)))/2, color=clusters_colors[cluster_idx], fill=False)) for cluster_idx in num_color_range]

                #ax.set_xlim(xmin=0-0.5, xmax=CELLS_COLS+0.5)
                #ax.set_ylim(ymin=CELLS_ROWS+0.5, ymax=0-0.5)
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
            #print(agents_paths)
            for path in agents_paths:
                if (DIMENSION_2D == False):
                    path_x, path_y, path_z = [np.array(coords[0]) for coords in path], [np.array(coords[1]) for coords in path], [np.array(coords[2]) for coords in path]
                    data_path.append([path_x, path_y, path_z])
                else:
                    path_x, path_y = [np.array(coords[0]-0.5) for coords in path], [np.array(coords[1]-0.5) for coords in path]
                    data_path.append([path_x, path_y])

            #print(data_path)
            data_path = np.array(data_path)  

            #print("AOOOOH:\n", data_path)

            lines = []
            circles = []
            uav_color_count = 0

            if (DIMENSION_2D == False):
                x = ACTUAL_UAV_FOOTPRINT * np.outer(np.cos(u), np.sin(v))
                y = ACTUAL_UAV_FOOTPRINT * np.outer(np.sin(u), np.sin(v))
                z = 0 * np.outer(np.ones(np.size(u)), np.cos(v))
                for path in data_path:
                    #print(path[0, 0:1], path[1, 0:1])
                    lines.append(ax.plot(path[1, 0:1], path[0, 0:1], path[2, 0:1], color=UAVS_COLORS[uav_color_count])[0])
                    circles.append(ax.plot_surface(x, y, z, color=UAVS_COLORS[uav_color_count], linewidth=0))
                    #circles.append(plt.Circle(xy=(path[1, 0:1], path[0, 0:1]), radius=ACTUAL_UAV_FOOTPRINT, color=UAVS_COLORS[uav_color_count], fill=True, alpha=0.18))
                    #circles.append(ax.scatter(path[0, 0:1], path[1, 0:1], 0, s=2000, color=UAVS_COLORS[uav_color_count], alpha=0.18))
                    uav_color_count += 1
                '''                
                for patch in circles:
                    ax.add_patch(patch)
                    art3d.pathpatch_2d_to_3d(patch)
                '''
            else:
                for path in data_path:
                    lines.append(ax.plot(path[0, 0:1], path[1, 0:1], color=UAVS_COLORS[uav_color_count])[0])
                    circles.append(plt.Circle(xy=(path[0, 0:1], path[1, 0:1]), radius=ACTUAL_UAV_FOOTPRINT, color=UAVS_COLORS[uav_color_count], fill=True, alpha=0.18))
                    uav_color_count += 1
                for patch in circles:
                    ax.add_patch(patch)

            #line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], color='orange')
            #print("AOHHHHH", lines)

            if (DIMENSION_2D == False):
                n_circles_range = range(len(circles))
                ani = animation.FuncAnimation(fig, self.update_animation, frames=ITERATIONS_PER_EPISODE-1, fargs=(data_path, lines, circles, n_circles_range, ax), interval=100, blit=True, repeat=True) # fargs=(data_path, lines, circles)
            else:
                ani = animation.FuncAnimation(fig, self.update_animation_2D, frames=ITERATIONS_PER_EPISODE-1, fargs=(data_path, lines, circles), interval=100, blit=True, repeat=True)

            #ax.set_xlim(xmin=0-0.5, xmax=CELLS_COLS+0.5)
            #ax.set_ylim(ymin=0-0.5, ymax=CELLS_ROWS+0.5)
            if (DIMENSION_2D==False):
                ax.set_zlim(zmin=0)
            ani.save(join(where_to_save, "animation_ep" + str(episode) + ".gif"), writer='imagemagick')

            plt.close(fig)
            
            #canvas.print_figure('test')
            #plt.show()

            #Writer = animation.writers['ffmpeg']

        else:

            #print("GUARDA QUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", centroids[0], clusters_radiuses[0], centroids[1], clusters_radiuses[1])
            num_clusters = len(centroids)
            clusters_colors, num_color_range = self.RGBA_01_random_colors(num_clusters)

            x_obs_points, y_obs_points, z_obs_points = self.extract_coord_from_xyz(obs_points)
            '''
            print("POINTS OBSTACLES")
            for obs in obs_points:
                print((obs._x_coord, obs._y_coord))
            '''
            x_cs_points, y_cs_points, z_cs_points = self.extract_coord_from_xyz(cs_points)
            if (CREATE_ENODEB == True):
                x_enb_point, y_enb_point, z_enb_point = self.extract_coord_from_xyz(enb_point)
            x_obs_cells, y_obs_cells, z_obs_cells = self.extract_coord_from_xyz(obs_cells)
            '''
            print("CELLS OBSTACLES")
            for obs in obs_cells:
                print((obs._x_coord, obs._y_coord))
            '''
            
            x_cs_cells, y_cs_cells, z_cs_cells = self.extract_coord_from_xyz(cs_cells)
            if (CREATE_ENODEB == True):
                x_eNB_cells, y_eNB_cells, z_eNB_cells = self.extract_coord_from_xyz(enb_cells)

            users_x, users_y, users_z = self.extract_coord_from_xyz(users)
            print("USERS")
            for us in users:
                print("Point user coords:")
                print( (us._x_coord, us._y_coord, us._z_coord) )
                print("Cell user coords:")
                print( (us._x_coord/CELL_RESOLUTION_PER_COL, us._y_coord/CELL_RESOLUTION_PER_ROW, us._z_coord) )
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
            print(points_status_matrix)

            #ax1.imshow(cs_points, cmap="Greens")

            ax1.set_xticks(np.arange(0, area_width+1, 1)-0.5)
            ax1.set_yticks(np.arange(0, area_height+1, 1)-0.5)

            ax1.set_xticklabels(np.arange(0, area_width+1, 1))
            ax1.set_yticklabels(np.arange(0, area_height+1, 1))

            #line1 = line2 = line3 = [1, 2, 3]
            #ax1.legend((line1, line2, line3), ('label1', 'label2', 'label3'))
            #red_patch = mpatches.Patch(color='red', label='The red data')
            #plt.legend(handles=[red_patch])
            #ax1.scatter([1.3, 2.5, 3.1, 4.8], [1.3, 2.5, 3.1, 4.8], s=10, c=GOLD)
            ax1.grid(which='both')
            ax1.scatter(users_x_for_2Dplot, users_y_for_2Dplot, s=10, c=GOLD)
            for cluster_idx in num_color_range:
                [ax1.add_artist(plt.Circle([centroids[cluster_idx][0], centroids[cluster_idx][1]], float(clusters_radiuses[cluster_idx]), color=clusters_colors[cluster_idx], fill=False)) for cluster_idx in num_color_range]
            ax1.set_title('2D Points-Map')

            #ax2.scatter([1.3, 2.5, 3.1, 4.8], [1.3, 2.5, 3.1, 4.8], [1, 2, 3, 4], s=10, c=GOLD)
            if (DIMENSION_2D == False):
                ax2.scatter(users_y_for_3Dplot, users_x_for_3Dplot, users_z_for_3Dplot, s=10, c=GOLD)
                '''
                p = plt.Circle((5, 5), 3)
                ax2.add_patch(p)
                art3d.pathpatch_2d_to_3d(p)
                '''
                for cluster_idx in num_color_range:
                    patch = plt.Circle([centroids[cluster_idx][1]+0.5, centroids[cluster_idx][0]+0.5, centroids[cluster_idx][2]], float(clusters_radiuses[cluster_idx]), color=clusters_colors[cluster_idx], fill=False)
                    ax2.add_patch(patch)
                    art3d.pathpatch_2d_to_3d(patch)        
                #[ax2.add_artist(plt.Circle(centroids[cluster_idx], float(clusters_radiuses[cluster_idx]), color=clusters_colors[cluster_idx], fill=False)) for cluster_idx in num_color_range]
                ax2.bar3d(y_obs_points, x_obs_points, bottom, width, depth, z_obs_points, shade=True, color=(0, 0, 0.6), edgecolor="none")
                if (UNLIMITED_BATTERY == False):
                    ax2.bar3d(y_cs_points, x_cs_points, bottom, width, depth, z_cs_points, shade=True, color=(0, 0.4, 0), edgecolor="none")
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

            #line, = ax4.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], color='orange')

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
            print(cells_status_matrix)

            ax3.set_xticks(np.arange(0, N_cells_col+1, 1)-0.5)
            ax3.set_yticks(np.arange(0, N_cells_row+1, 1)-0.5)

            ax3.set_xticklabels(np.arange(0, area_width+1, 1))
            ax3.set_yticklabels(np.arange(0, area_height+1, 1))

            ax3.grid(which='major')
            ax3.scatter(users_x_for_2DplotCells, users_y_for_2DplotCells, s=10, c=GOLD)
            # A Graphical approximation is needed in order to get a cluster in 'cells view' which is as closest as possible to the one in 'points view' (The approximation is only graphical):
            for cluster_idx in num_color_range:
                #[ax3.add_artist(plt.Circle([centroids[cluster_idx][0]/CELL_RESOLUTION_PER_ROW, centroids[cluster_idx][1]/CELL_RESOLUTION_PER_COL, centroids[cluster_idx][2]], (float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_ROW)) + float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_COL)))/2, color=clusters_colors[cluster_idx], fill=False)) for cluster_idx in num_color_range]
                [ax3.add_artist(plt.Circle([centroids[cluster_idx][0]/CELL_RESOLUTION_PER_ROW-0.25, centroids[cluster_idx][1]/CELL_RESOLUTION_PER_COL-0.25, centroids[cluster_idx][2]], (float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_ROW)) + float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_COL)))/2, color=clusters_colors[cluster_idx], fill=False)) for cluster_idx in num_color_range]
            ax3.set_title('2D Cells-Map')

            if (DIMENSION_2D == False):
                ax4.scatter(users_y_for_3DplotCells, users_x_for_3DplotCells, users_z_for_3DplotCells, s=10, c=GOLD)
                for cluster_idx in num_color_range:
                    #patch = plt.Circle([centroids[cluster_idx][1]+0.5, centroids[cluster_idx][0]+0.5, centroids[cluster_idx][2]], float(clusters_radiuses[cluster_idx]), color=clusters_colors[cluster_idx], fill=False)
                    patch = plt.Circle([centroids[cluster_idx][1]/CELL_RESOLUTION_PER_COL+0.25, centroids[cluster_idx][0]/CELL_RESOLUTION_PER_ROW+0.25, centroids[cluster_idx][2]], (float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_ROW)) + float(clusters_radiuses[cluster_idx]/(CELL_RESOLUTION_PER_COL)))/2, color=clusters_colors[cluster_idx], fill=False)
                    ax4.add_patch(patch)
                    art3d.pathpatch_2d_to_3d(patch)
                    pass
                ax4.bar3d(y_obs_cells, x_obs_cells, bottom, width, depth, z_obs_cells, shade=True, color=(0.4, 1, 1), edgecolor="none")
                if (UNLIMITED_BATTERY == False):
                    ax4.bar3d(y_cs_cells, x_cs_cells, bottom, width, depth, z_cs_cells, shade=True, color=(0.4, 1, 0.59), edgecolor="none")
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

                print("QUAAAA", perceived_status_matrix)
                
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
                for cluster_idx in num_color_range:
                    [ax5.add_artist(plt.Circle([centroids[cluster_idx][0], centroids[cluster_idx][1]], float(clusters_radiuses[cluster_idx]), color=clusters_colors[cluster_idx], fill=False)) for cluster_idx in num_color_range]
                ax5.set_title('2D Points/Cells-Map')

                cs_cells_colors = [(0.4, 1, 0.59, 0.3) for i in range(N_CS)]
                obs_cells_colors = [(0.4, 1, 1, 0.3) for i in range(len(obs_cells))]

                if (DIMENSION_2D == False):
                    ax6.scatter(users_y_for_3Dplot, users_x_for_3Dplot, users_z_for_3Dplot, s=10, c=GOLD)
                    for cluster_idx in num_color_range:
                        patch = plt.Circle([centroids[cluster_idx][1]+0.5, centroids[cluster_idx][0]+0.5, centroids[cluster_idx][2]], float(clusters_radiuses[cluster_idx]), color=clusters_colors[cluster_idx], fill=False)
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
            #fig3.legend(handles=[DARK_BLUE_square, red_square])

            #ani = animation.FuncAnimation(fig2, self.update, N, fargs=(data, line), blit=False)

            plt.show()

    def plt_daily_users_distribution(self, daily_users_traffic_per_cluster):

        # Define colored canvas for the legend:
        light_red_line = mlines.Line2D([], [], color=LIGHT_RED, marker='_', markersize=15, label="Users 1° Timeslot")
        light_blue_line = mlines.Line2D([], [], color=LIGHT_BLUE, marker='_', markersize=15, label="Users 2° Timeslot")
        light_green_line = mlines.Line2D([], [], color=LIGHT_GREEN, marker='_', markersize=15, label="Users 3° Timeslot")
        purple_line = mlines.Line2D([], [], color=PURPLE, marker='_', markersize=15, label="Users 4° Timeslot")
        orange_line = mlines.Line2D([], [], color=ORANGE, marker='_', markersize=15, label="Users 5° Timeslot")
        brown_line = mlines.Line2D([], [], color=BROWN, marker='_', markersize=15, label="Users 6° Timeslot")

        #daily_users_traffic = User.users_per_timeslot(15, 30)

        #print(daily_users_traffic)
        hours = [hour for hour in range(1, HOURS_PER_CONSIDERED_TIME+1)]
        daily_users_traffic = [sum(cluster_traffic) for cluster_traffic in daily_users_traffic_per_cluster]
        barlist = plt.bar(hours, daily_users_traffic, align='center')
        slot_divisor_line_colors = [LIGHT_RED, LIGHT_BLUE, LIGHT_GREEN, PURPLE, ORANGE, BROWN]
        
        color_idx = 0
        for slot in range(STARTING_TIMESLOT, 25, MICRO_SLOTS_PER_MACRO_TIMESLOT):
            first_bar_idx = abs(STARTING_TIMESLOT - slot)
            last_bar_idx = first_bar_idx + STARTING_TIMESLOT
            #print((first_bar_idx, last_bar_idx))

            for bar_idx in range(first_bar_idx, last_bar_idx):
                barlist[bar_idx].set_color(slot_divisor_line_colors[color_idx])

            #print(slot_divisor_line_colors[color_idx])
            color_idx += 1
        #plt.plot(daily_users_traffic)

        plt.xlabel("Hours")
        plt.xticks(hours)
        plt.ylabel("Users")
        plt.title("Users per day")
        plt.legend(handles=[light_red_line, light_blue_line, light_green_line, purple_line, orange_line, brown_line])
        #plt.legend(handles=slot_divisor_line_colors, labels=['1° Timeslot', '2° Timeslot', '3° Timeslot', '4° Timeslot', '5° Timeslot', '6° Timeslot'])
        plt.plot(hours, daily_users_traffic)
        #plt.xlim(left=1, right=26)
        #plt.xlim(left=1, right=24)
        plt.show()

    def QoE_plot(self, parameter_values, epochs, where_to_save, param_name, uav_idx, legend_labels):

        epochs_to_plot = range(1, epochs+1)

        fig = plt.figure(param_name)
        ax = fig.add_subplot(111)
        plt.xlabel('Epochs')
        plt.ylabel(param_name + ' Trend')
        plt.title(param_name)
        plt.plot(epochs_to_plot, parameter_values)
        #legend_labels.append('UAV' + str(uav_idx+1) )
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
        plt.legend(legend_labels)
        plt.savefig(where_to_save + '.png')

        #plt.show()

    def UAVS_reward_plot(self, epochs, UAVs_rewards, directory_name, q_values=False):

        epochs_to_plot = range(1, epochs+1)
        UAV_ID = 0
        #print("BRIIIIIII")
        #print(UAVs_rewards)
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
        #plt.set_xticklabels(epochs_to_plot)
        #plt.set_yticklabels(range(0, 1))

        legend_labels = []
        for UAV in UAVs_rewards:
            #print(UAV)
            #print()
            plt.plot(epochs_to_plot, [reward for reward in UAV], color=UAVS_COLORS[UAV_ID])
            UAV_ID += 1
            legend_labels.append('UAV' + str(UAV_ID))
            #plt.savefig(join(directory_name, str(UAV_ID)) + ".png")
        
        ax.set_xlim(xmin=1)
        ax.set_ylim(ymin=0)
        plt.legend(legend_labels)
        #plt.savefig(join(directory_name, "Rewards_per_epoch_UAV" + str(UAV_ID)) + ".png")
        if (q_values==False):
            plt.savefig(join(directory_name, "Rewards_per_epoch_UAVs.png"))
        else:
            plt.savefig(join(directory_name, "Q-values_per_epoch_UAVs.png"))

        #plt.show()

    def epsilon(self, epsilon_history, epochs, directory_name):

        epochs_to_plot = range(1, epochs+1)

        fig = plt.figure()

        ax = fig.add_subplot(111)
        
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Epsilon Value")

        ax.plot(epochs_to_plot, epsilon_history)
        
        ax.set_xlim(xmin=1)
        ax.set_ylim(ymin=0)
        plt.savefig(join(directory_name, "Epsilon_per_epoch.png"))

        #plt.show()

    def actions_min_max_per_epoch(self, q_table, directory_name, episode, which_uav):

        #ax1 = fig.add_subplot(311)
        #ax2 = fig.add_subplot(312)
        #ax3 = fig.add_subplot(313)
        fig = plt.figure()

        #axes = []
        #for action in range(1, n_actions+1):
        #    axes.append(fig.add_subplot(n_actions,1,action))
        ax = fig.add_subplot(111)

        #q_table = np.load(f"qtables/{i}-qtable.npy")
        actual_q_table = q_table[which_uav-1]

        #n_actions_range = range(n_actions)
        #for action_id in n_actions_range:
        state_count = 0
        for state in actual_q_table:
            #print("Actions Values:", actual_q_table[state])
            #print("Current Action:", actual_q_table[state][action_id])
            action_values = actual_q_table[state]
            #print("AOOOOOOOOOOOOOOOOOOOOOOOOOOOH", action_values)
            #print("AEEEEEEEEEEEEEEEEEEEEEEEEEEEH", len(action_values))
            #print("Current Action vaulues:", action_values)
            max_action_value = max(action_values)
            min_action_value = min(action_values)
            ax.scatter(state_count, DICT_ACTION_SPACE[action_values.index(max_action_value)], c="green", marker="o", alpha=0.4)
            ax.scatter(state_count, DICT_ACTION_SPACE[action_values.index(min_action_value)], c="red", marker="o", alpha=0.4)
            state_count += 1
        ax.set_xlabel("States")
        ax.set_ylabel("Actions")
        #print("MAX Action:", actual_q_table[state][action_id])
        '''
        state_count = 0
        for state in actual_q_table:
            for action in action_values:
                #print("Action ID:", action_id)
                #print("Action:", action)
                current_state = actual_q_table[state]
                if (action == max_action_value):
                    ax.scatter(state_count, action_id, c="green", marker="o", alpha=1.0)
                elif (action == min_action_value):
                    ax.scatter(state_count, action_id, c="red", marker="o", alpha=1.0)
        '''
        '''
        if action == max_action_value:
            ax.scatter(action_id, action, c="green", marker="o", alpha=1.0)
            #axes[action_id].scatter(action_id, action, c="green", marker="o", alpha=1.0)
        else:
            ax.scatter(action_id, action, c="red", marker="o", alpha=0.3)
            #axes[action_id].scatter(action_id, action, c="red", marker="o", alpha=0.3)
        '''
        '''
        for x, x_vals in enumerate(actual_q_table):
            print("X:", x, x_vals)
            print("Q-TABLE actions in X:", actual_q_table[x_vals])
            action_values = []
            for action_val in actual_q_table[x_vals]:
                action_values.append(action_val)
            max_action_value = max(action_values)

            for act in action_values:
                if act == max_action_value:
                    axes[x].scatter(x, act, c="green", marker="o", alpha=1.0)
                else:
                    axes[x].scatter(x, act, c="red", marker="o", alpha=0.3)
        '''
            
            #for y, y_vals in enumerate(x_vals):
                #print("Y:", y, y_vals)
                #for action in range(n_actions):
                    #print("LUNNNNNNNNNN", len(y_vals))
                    #axes[action].scatter(x, y, c=self.q_color_value(y_vals, x_vals)[0], marker="o", alpha=self.q_color_value(y_vals, x_vals)[1])

                    #ax[row, col].scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
                    #ax1.scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
                    #ax2.scatter(x, y, c=get_q_color(y_vals[1], y_vals)[0], marker="o", alpha=get_q_color(y_vals[1], y_vals)[1])
                    #ax3.scatter(x, y, c=get_q_color(y_vals[2], y_vals)[0], marker="o", alpha=get_q_color(y_vals[2], y_vals)[1])
                    #axes[action].set_ylabel("Action " + str(action))
                    #ax1.set_ylabel("Action 0")
                    #ax2.set_ylabel("Action 1")
                    #ax3.set_ylabel("Action 2")

        #print("AOOOOH", directory_name + f"qtable_graph-ep{episode}.png")
        plt.savefig(directory_name + f"\qtable_graph-ep{episode}.png")


    def battery_when_start_to_charge(self, battery_history, directory_name):

        n_recharges_per_uav = [range(1, len(battery_history[uav])+1) for uav in range(N_UAVS)]
        print(len(battery_history[0]))
        max_recharges_per_uav = max([len(battery_history[uav]) for uav in range(N_UAVS)])
        #n_recharges_per_uav.append(range(1, len(battery_history[0])+1))
        #copy = [elem+1 for elem in battery_history[0]]
        #battery_history.append(copy)
        UAV_ID = 0

        fig = plt.figure()

        ax = fig.add_subplot(111)
        
        ax.set_xlabel("N-th Start of Charging (every " + str(SHOW_BATTERY_LEVEL_FOR_CHARGING_INSTANT) + " charges)")
        ax.set_ylabel("Battery Level")
        plt.title('Battery level when start to charge')

        #legend_labels = []
        for UAV in n_recharges_per_uav:
            #print(UAV)
            #print()
            #print("UAV IDDDDDDDDDDDDDDDDDDDDDDD", UAV_ID)
            ax.bar(n_recharges_per_uav[UAV_ID], [self.battery_percentage(battery_levels) for battery_levels in battery_history[UAV_ID]], color=UAVS_COLORS[UAV_ID], align='center')
            #plt.plot(n_recharges_per_uav[UAV_ID], [self.battery_percentage(battery_levels) for battery_levels in battery_history[UAV_ID]], color=UAVS_COLORS[UAV_ID])
            #print(len(n_recharges_per_uav[UAV_ID]))
            UAV_ID += 1
            #legend_labels.append('UAV' + str(UAV_ID))
            ax.set_xlim(xmin=0)
            ax.set_ylim(ymin=0)
            print(range(0, max_recharges_per_uav*SHOW_BATTERY_LEVEL_FOR_CHARGING_INSTANT, SHOW_BATTERY_LEVEL_FOR_CHARGING_INSTANT))
            ax.set(xticks=range(0, max_recharges_per_uav+1), xticklabels=range(0, max_recharges_per_uav*SHOW_BATTERY_LEVEL_FOR_CHARGING_INSTANT+1, SHOW_BATTERY_LEVEL_FOR_CHARGING_INSTANT))
            plt.xticks(range(0, max_recharges_per_uav+1))
            plt.tick_params(labelbottom=False)
            plt.yticks(range(0, self.battery_percentage(CRITICAL_BATTERY_LEVEL), 5))
            #plt.legend(['UAV' + str(UAV_ID)])
            plt.savefig(join(directory_name, "Battery_level_when_start_charge_UAV" + str(UAV_ID)) + ".png")
        
        #plt.savefig(join(directory_name, "Battery_level_when_start_charge" + str(UAV_ID)) + ".png")

        #plt.show()

    def UAVS_crashes(self, epochs, UAVs_crashes, directory_name):

        epochs_to_plot = range(1, epochs+1)
        #print("BRIIIIIII")
        #print(UAVs_rewards)
        fig = plt.figure('UAVs crashes')
        ax = fig.add_subplot(111)
        #plt.set_xticklabels(epochs_to_plot)
        #plt.set_yticklabels(range(0, 1))

        plt.xlabel('Epochs')
        plt.ylabel('UAV ID')
        plt.title('UAVs Crashes')

        #print("CRASHEEEEEEEE", UAVs_crashes)

        legend_labels = []
        for episode in epochs_to_plot:
            n_labels = len(legend_labels)
            UAV_ID = 1
            # Questa condizione mi serve solo nel caso in cui voglio plottare prima della fine (per vedere se tutto funziona e fare dei tests) --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if (UAVs_crashes[episode-1]==0):
                break
            for UAV_crash in UAVs_crashes[episode-1]:
                str_uav_id = str(UAV_ID)
                #print(UAV)
                #print()
                #print(UAV_ID)
                if (UAV_crash==True):
                    ax.scatter(episode, str_uav_id, color=UAVS_COLORS[UAV_ID-1], marker="x")
                #plt.plot(epochs_to_plot, [crash for crash in UAVS_chrashes], color=UAVS_COLORS[UAV_ID])
                if (n_labels<N_UAVS):
                    legend_labels.append('UAV' + str(UAV_ID))
                UAV_ID += 1
            #plt.savefig(join(directory_name, str(UAV_ID)) + ".png")
        
        #ax.set_xlim(xmin=1)
        #ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=1, xmax=epochs+1)
        plt.legend(legend_labels)
        #plt.savefig(join(directory_name, "Rewards_per_epoch_UAV" + str(UAV_ID)) + ".png")
        plt.savefig(join(directory_name, "UAVs_crashes.png"))

        #plt.show()

    def battery_percentage(self, battery_level):

        percentage_battery_level = round((battery_level*100)/FULL_BATTERY_LEVEL)

        return percentage_battery_level

if __name__ == '__main__':
    
    plot = Plot()

    # Loading:
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

    points_status_matrix = plot.compute_status_matrix(points_matrix, AREA_HEIGHT, AREA_WIDTH)
    #print(points_status_matrix)
    cells_status_matrix = plot.compute_status_matrix(cells_matrix, CELLS_ROWS, CELLS_COLS)
    perceived_status_matrix = plot.compute_perceived_status_matrix(cells_matrix, AREA_HEIGHT, AREA_WIDTH, CELLS_ROWS, CELLS_COLS)
    #print(perceived_status_matrix)

    '''
    print("Points Representation: 1 elem = 1 cell\n")
    for row in range(AREA_HEIGHT):
        for col in range(AREA_WIDTH):
            print(points_matrix[row][col]._status, end= " ")
        print()

    print("\n")

    print("Cells Representation: 1 elem =", CELL_RESOLUTION_PER_ROW*CELL_RESOLUTION_PER_COL, "cell\n")
    for row in range(CELLS_ROWS):
        for col in range(CELLS_COLS):
            print(cells_matrix[row][col]._status, end= " ")
        print()

    print("\n")
    '''

    # Create directory 'MAP_STATUS_DIR' to save data:
    directory = Directories()
    directory.create_map_status_dir()
    
    # Saving:
    save = Saver()
    save.maps_status(points_status_matrix, cells_status_matrix, perceived_status_matrix)

    #print(obs_cells[0]._vector)

    # Plotting:
    agents_paths = [[(0,0,1), (1,0,1), (1,1,2), (1,1,3), (2,1,2)], [(0,0,1), (0,1,1), (1,1,0), (1,1,2), (1,2,3)]]
    plot.plt_map_views(obs_points, cs_points, eNB_point, obs_cells, cs_cells, eNB_cells, points_status_matrix, cells_status_matrix, perceived_status_matrix, initial_users, initial_centroids, initial_clusters_radiuses, AREA_HEIGHT, AREA_WIDTH, CELLS_ROWS, CELLS_COLS, agents_paths=None, path_animation=False)
    
    '''
    # CONTROLLA IL CASO CON PIU' CLUSTERS DI UTENTI --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    daily_users_traffic = User.users_per_cluster_per_timeslot(MIN_USERS_PER_DAY, MAX_USERS_PER_DAY, FIXED_CLUSTERS_NUM) # --> Questo è qui solo per fare una prova (dovrai salvare il traffico ad ogni nuovo giorno) --> !!!!!!!!!!!!!!!!!!!!!!!!
    print("Users per timeslot:", daily_users_traffic)
    plot.plt_daily_users_distribution(daily_users_traffic)
    '''
