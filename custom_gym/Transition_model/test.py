import math


def get_3D_components(start3D,end3D):
    d_x = end3D[0] - start3D[0] 
    d_y = end3D[1] - start3D[1] 
    d_z = end3D[2] - start3D[2] 
    d = distance_3D(start3D, end3D)
    if(d==0):
        raise Exception("Zero length vector")
    
    alpha = math.acos(d_x/d)
    beta = math.acos(d_y/d)
    gamma = math.acos(d_z/d)

    x_component = math.cos(alpha) * d
    y_component = math.cos(beta) * d 
    z_component = math.cos(gamma) * d

    return x_component, y_component, z_component
    

def distance_3D(start,end):
    return math.sqrt( (end[2] - start[2] )**2 + (end[1] - start[1] ) **2 + (end[0] - start[0] )**2 )    


print(get_3D_components([5,5,5],[1,1,1] ) )