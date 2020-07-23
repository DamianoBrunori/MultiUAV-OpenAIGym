from itertools import product
from scenario_objects import Point, Cell
from my_utils import LOWER_BOUNDS, AREA_WIDTH, AREA_HEIGHT, CELLS_ROWS, CELLS_COLS, MINIMUM_AREA_HEIGHT, MAXIMUM_AREA_HEIGHT, OBS_IN, CS_IN, DIMENSION_2D, UAV_Z_STEP
from load_and_save_data import Loader
import math

coords_moves = [-1, 0, 1]

'''
def moves(cell):

    all_moves = list(product(coords_moves, coords_moves, coords_moves))
    return moves
'''

def moves2D(cell):

    all_moves = list(product(coords_moves, coords_moves))
    moves2D=all_moves
    print ('Moves 2D')
    return moves2D

def moves3D(cell):

    all_moves = list(product(coords_moves, coords_moves, coords_moves))
    moves3D=all_moves
    print('Moves 3D')
    return moves3D

def allowed_neighbours_2D(moves, node, env_matrix, resolution='min'):
    
    if (resolution == 'min'):
        x_upper_bound = AREA_WIDTH
        y_upper_bound = AREA_HEIGHT
    elif (resolution == 'des'):
        x_upper_bound = CELLS_ROWS
        y_upper_bound = CELLS_COLS

    neighbours_list = []

    node_x = node.position[0]
    node_y = node.position[1]

    for move in moves:
        move_x = move[0]
        move_y = move[1]

        new_node_x = node_x + move_x
        new_node_y = node_y + move_y

        try:
            # if 'move_x' and 'move_y' are out of 'env_matrix', then keep searching for other moves (see continue inside 'except'):
            y = int(new_node_y-0.5)
            x = int(new_node_x-0.5)
            if ( (x<0) or (y<0) ):
                continue
            else:
                current_cell = env_matrix[y][x]
        except:
            continue
        
        if (current_cell._status != OBS_IN):
            neighbours_list.append((new_node_x, new_node_y))
        else:
            continue
        #if (env_matrix[new_node_y][new_node_x]._z_coord < new_node_z):
        #    neighbours_list.append((new_node_x, new_node_y, new_node_z))

    return neighbours_list

def allowed_neighbours_3D(moves, node, env_matrix, resolution='min'):
    
    if (resolution == 'min'):
        x_upper_bound = AREA_WIDTH
        y_upper_bound = AREA_HEIGHT
    elif (resolution == 'des'):
        x_upper_bound = CELLS_ROWS
        y_upper_bound = CELLS_COLS

    neighbours_list = []

    node_x = node.position[0]
    node_y = node.position[1]
    node_z = node.position[2]

    for move in moves:
        move_x = move[0]
        move_y = move[1]
        move_z = move[2] if DIMENSION_2D==False else 0

        new_node_x = node_x + move_x
        new_node_y = node_y + move_y
        new_node_z = node_z + move_z if DIMENSION_2D==False else 0
        
        # If 'new_node_z' is not inside (lower_bounds, z_upper_bound), then keep searching for other moves:
        if ( (new_node_z < MINIMUM_AREA_HEIGHT) or (new_node_z >= MAXIMUM_AREA_HEIGHT) ):
            continue

        try:
            # if 'move_x' and 'move_y' are out of 'env_matrix', then keep searching for other moves (see continue inside 'except'):
            y = int(new_node_y-0.5)
            x = int(new_node_x-0.5)
            if ( (x<0) or (y<0) ):
                continue
            else:
                current_cell = env_matrix[y][x]
        except:
            continue
        
        if ( (current_cell._status != OBS_IN) or ((current_cell._status == OBS_IN) and (current_cell._z_coord < new_node_z)) ):
            neighbours_list.append((new_node_x, new_node_y, new_node_z))
        else:
            continue
        #if (env_matrix[new_node_y][new_node_x]._z_coord < new_node_z):
        #    neighbours_list.append((new_node_x, new_node_y, new_node_z))

    return neighbours_list

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position if DIMENSION_2D == False else position + (0,)

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(env_matrix, start, goal):
    """Returns a list of tuples as a path from the given start to the given goal in the given 'env_matrix'"""

    # Create start and goal node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    goal_node = Node(None, goal)
    goal_node.g = goal_node.h = goal_node.f = 0

    # Initialize both open and closed list
    open_set = []
    closed_set = []

    # Add the start node
    open_set.append(start_node)

    if (DIMENSION_2D == True):
        moves = moves2D
        allowed_neighbours = allowed_neighbours_2D
    else:
        moves = moves3D
        allowed_neighbours = allowed_neighbours_3D

    # Loop until you find the goal
    while len(open_set) > 0:

        # Get the current node
        current_node = open_set[0]
        current_index = 0
        for index, item in enumerate(open_set):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Current node removal by 'pop' from open set and addition of the same node to closed set
        open_set.pop(current_index)
        closed_set.append(current_node)

        # Case in which the goal has been found:
        if current_node == goal_node:
            path = []
            current_n = current_node

            # Loop until the root parent (i.e. 'None') is achieved: 
            while current_n is not None:
                path.append(current_n.position)
                current_n = current_n.parent

            return path[::-1] # return reversed path, i.e. listing all the nodes of the found path from the starting node to the goal node. 

        # Generate children:
        children = []

        moves_for_current_node = moves(current_node)
        #print("MOSSEEEEE", len(moves_for_current_node))
        new_allowed_positions = allowed_neighbours(moves_for_current_node, current_node, env_matrix, resolution='min')
        #print(new_allowed_positions)

        for new_position in new_allowed_positions:

            # New allowed node position:
            if (DIMENSION_2D==False):
                new_node_position = (new_position[0], new_position[1], new_position[2])
            else:
                new_node_position = (new_position[0], new_position[1])
            # New node creation
            new_node = Node(current_node, new_node_position)

            # New node addition:
            children.append(new_node)

        # Check every child:
        for child in children:

            # Case in which the child is inside the closed set:
            if child in closed_set:
                continue

            '''
            for closed_child in closed_set:
                if child == closed_child:
                    continue
            '''

            ''' #Formula per calcolare la Diztanza Euclidea, quadrata (Euclidean distance, squared)
            # This definitely runs into the scale problem. The scale of g and h need to match, because you’re adding them together to form f
            # Sconsigliata
            # Create the f, g, and h values
            child.g = current_node.g + 1
            if (DIMENSION_2D == False):
                child.h = ((child.position[0] - goal_node.position[0]) ** 2) + ((child.position[1] - goal_node.position[1]) ** 2) + ((child.position[2] - goal_node.position[2]) ** 2)
            else:
                child.h = ((child.position[0] - goal_node.position[0]) ** 2) + ((child.position[1] - goal_node.position[1]) ** 2)
            child.f = child.g + child.h
            '''

            child.g = current_node.g + 1
            nodeX = child.position[0]
            goalX = goal_node.position[0]
            nodeY= child.position[1]
            goalY = goal_node.position[1]
            nodeZ = child.position[2]
            goalZ = goal_node.position[2]
            if (DIMENSION_2D == False):
                dx = abs(nodeX - goalX)
                dy = abs(nodeY - goalY)
                child.h = math.sqrt(dx * dx + dy * dy)
            else:
                dx = abs(nodeX - goalX)
                dy = abs(nodeY - goalY)
                dz = abs(nodeZ - goalZ)
                child.h = math.sqrt(dx * dx + dy * dy + dz * dz)
            child.f = child.g + child.h


            # Case in which the child is already inside the open set:
            if (child in open_set):
                open_node = open_set[open_set.index(child)]
                
                if (child.g > (open_node.g)):
                    continue

            '''
            for open_node in open_set:
                if child == open_node and child.g > open_node.g:
                    continue
            '''
            
            for open_node in open_set:
                if child == open_node and child.g < open_node.g:
                    open_set.pop(open_set.index(open_node))

            # Child addition to the oper set:
            open_set.append(child)


def main():

    # IF YOU ARE PERFORMING SOME TESTS WITH DIFFERENT start AND goal, BE SURE TO SET start AND goal
    # ACCORDING TO THE VALUE ASSIGNED TO 'DIMENSION_2D', OTHERWISE THE PATH FOUND BY A* WILL BE OBVIOSLY EQUAL TO None.
    start = (3.5, 3.5)
    goal = (5.5, 9.5)

    load = Loader()
    load.maps_data()
    points_matrix = load._points_matrix
    cells_matrix = load._cells_matrix
    '''
    for i in range(AREA_HEIGHT):
        for j in range(AREA_WIDTH):
            if points_matrix[i][j]._status == OBS_IN:
                print("OSTACOLO")
            elif points_matrix[i][j]._status == CS_IN:
                print("CHARGING STATION")
    '''
    path = astar(points_matrix, start, goal)
    print("PATH:", path)


if __name__ == '__main__':
    main()
