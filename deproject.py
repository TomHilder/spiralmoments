import numpy                as np
import matplotlib.pyplot    as plt

from rotations      import rotation_matrix
from copy           import deepcopy

def deproject_points(inclination, pos_angle, x_points, y_points, lim=6, resolution=500):
    """
    :param inclination: inclination of disk in degrees
    :param pos_angle: position angle of disk in degrees
    :param x_points: 1D numpy array of x coordinates of points to deproject (in arcseconds)
    :param y_points: 1D numpy array of y coordinates of points to deproject (in arcseconds)
    :param lim: coordinate of edge of projection domain, in arcseconds
    :param resolution: integer giving number of reference points used in projection, more is better
    """
    if x_points.shape != y_points.shape:
        raise ValueError("x_points and y_points must be 1D numpy arrays of the same shape")
    
    # rotation matrices for projection
    rot_in_x = rotation_matrix(inclination*np.pi/180, "x")
    rot_pa_z = rotation_matrix(  pos_angle*np.pi/180, "z")
    
    # setup surface for projecting
    x_grid = np.linspace(-lim, lim, resolution)
    y_grid = np.linspace(-lim, lim, resolution)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    z_grid = height(np.sqrt(x_grid**2 + y_grid**2))
    
    # get original points for later
    x_grid_mid = deepcopy(x_grid)
    y_grid_mid = deepcopy(y_grid)
    
    # project surface
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            x_grid[i,j], y_grid[i,j], z_grid[i,j] = np.dot(rot_in_x, [x_grid[i,j], y_grid[i,j], z_grid[i,j]])
            x_grid[i,j], y_grid[i,j], z_grid[i,j] = np.dot(rot_pa_z, [x_grid[i,j], y_grid[i,j], z_grid[i,j]])

    # ====== perform deprojecting ====== #
    
    # get empty arrays for deprojected points
    new_x_points = np.zeros(x_points.shape[0])
    new_y_points = np.zeros(y_points.shape[0])

    # for all of our original points
    for i in range(x_points.shape[0]):
        
        # get distances from each 'grid' point
        distance_from_grid = np.sqrt((x_grid - x_points[i])**2 + (y_grid - y_points[i])**2)
        
        # initialise closest point
        minimum = 1e9
        ind1 = None
        ind2 = None

        # find closest point
        for j in range(distance_from_grid.shape[0]):
            for k in range(distance_from_grid.shape[1]):
                if distance_from_grid[j,k] < minimum:
                    minimum = distance_from_grid[j,k]
                    ind1, ind2 = j, k
        
        # set unprojected closest point as deprojected data point
        new_x_points[i], new_y_points[i] = x_grid_mid[ind1,ind2], y_grid_mid[ind1,ind2]
        
    return new_x_points, new_y_points



def tapered_powerlaw(r, z0, q, r_taper=np.inf, q_taper=1.0, r_cavity=0.0,
                    r0=1.0):
    """Exponentially tapered power law profile."""
    rr = np.clip(r - r_cavity, a_min=0.0, a_max=None)
    f = powerlaw(rr, z0, q, r_cavity=0.0, r0=r0)
    return f * np.exp(-(rr / r_taper)**q_taper)

def powerlaw(r, z0, q, r_cavity=0.0, r0=1.0):
    """Standard power law profile."""
    rr = np.clip(r - r_cavity, a_min=0.0, a_max=None)
    return z0 * (rr / r0)**q

def height(r):
    arr = np.load('tapered_surface.npy')
    return tapered_powerlaw(r, arr[0], arr[1], r_taper=arr[2], q_taper=arr[3], r_cavity=0.0,
                        r0=1.0)