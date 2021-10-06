# Import Libraries

import yaml
import numpy                as np
import matplotlib.pyplot    as plt
import pymcfost             as pmc

from phantom_interface  import PhantomDump

from matplotlib.colors  import LogNorm
from matplotlib         import ticker, cm
from casa_cube          import casa_cube    as casa


def main():
 
    ### ========================================= ###
    
    ### Model parameters

    # wakeflow parameter file
    wakeflow_params = "HD_163296_secondary_kinks/smol_planet/config_hd163_v2.yaml"

    # observations fits file (CO cube)
    observations = "HD_163296_secondary_kinks/Observation/lines.fits"

    # use perturbations or total velocity? "delta" or "total"
    v_type = "delta"

    ### Plotting parameters

    # angles in degrees for wakeflow model -- must match the observational ones!

    inclination = -225
    PA = 45
    planet_az = 70 #335

    # which channels to plot

    ### ========================================= ###

    ### Read in wakeflow model

    # import wakeflow model configuration
    model_params = yaml.load(open(wakeflow_params), Loader=yaml.FullLoader)

    # check grid is Cartesian
    if str(model_params["grid"]["type"]) != "cartesian":
        raise Exception("Must use wakeflow model with cartesian grid")
    else:
        print("Reading wakeflow parameter file")

    # check perturbations were saved
    if not bool(model_params["results"]["save_perturbations"]):
        raise Exception("Must have saved perturbations")
    else:
        pass

    # importing parameters
    model_name = str(model_params["run_info"]["name"])
    model_system = str(model_params["run_info"]["system"])

    # check if multiple planet masses, ask which to use if multiple
    try:
        model_m_planet = float(model_params["disk"]["m_planet"])
    except:
        model_m_planet = list(model_params["disk"]["m_planet"])
        if len(model_m_planet) == 1:
            model_m_planet = model_m_planet[0]
        elif len(model_m_planet) > 1:
            print("Multiple planet masses detected: ", model_m_planet)
            num = int(input(f"Select model to use (1-{len(model_m_planet)}): "))
            model_m_planet = model_m_planet[num-1]
        else:
            raise Exception("Planet mass missing")

    # grab model directory
    model_dir = f"{model_system}/{model_name}/{model_m_planet}Mj"

    # read in arrays, and grab midplane -- resultant shape is (X, Y)
    z_index = 0
    A_X = np.load(f"{model_dir}/X.npy")[:, z_index, :]
    A_Y = np.load(f"{model_dir}/Y.npy")[:, z_index, :]
    A_v_r = np.load(f"{model_dir}/{v_type}_v_r.npy")[:, z_index, :]
    A_v_phi = np.load(f"{model_dir}/{v_type}_v_phi.npy")[:, z_index, :]
    
    if True:
        PD = PhantomDump()
        X = PD.X_ph
        Y = PD.Y_ph
        v_r = PD.vr_xy
        v_phi = PD.vphi_xy

    if True:
        plt.contourf(X, Y, 1e3*v_r, levels=np.linspace(-200,200,199), cmap="RdBu")
        plt.title("v_r")
        plt.colorbar()
        plt.show()
        
        plt.contourf(X, Y, 1e3*v_phi, cmap="RdBu", levels=199)#np.linspace(-200,200,199))
        plt.title("v_phi")
        plt.colorbar()
        plt.show()

    ### Angles for rotating velocity fields

    # get angles needed in radians
    PA *= np.pi / 180.
    planet_az *= np.pi / 180.
    inclination *= np.pi / 180.

    ### Get cartesian components of the velocity fields instead of radial and azimuthal

    # get meshgrids for polar coordinates
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)

    # perform transformations
    v_x = -v_phi * np.sin(PHI) + v_r * np.cos(PHI)
    v_y = v_phi * np.cos(PHI) + v_r * np.sin(PHI)
    v_z = np.zeros(v_x.shape)
    
    # define velocity field
    v_field = np.array([v_x, v_y, v_z])
    
    # get meshgrids for polar coordinates
    A_R = np.sqrt(X**2 + Y**2)
    A_PHI = np.arctan2(Y, X)

    # perform transformations
    A_v_x = -v_phi * np.sin(PHI) + v_r * np.cos(PHI)
    A_v_y = v_phi * np.cos(PHI) + v_r * np.sin(PHI)
    A_v_z = np.zeros(v_x.shape)
    
    # define velocity field
    A_v_field = np.array([v_x, v_y, v_z])
    
    """
    
    ### project velocities onto line of sight

    # find line of sight unit vector for projection
    proj_los = np.array(unit_vector(-inclination, -planet_az))
    
    # grid shape from x grid
    N_x = X.shape[0]
    N_y = X.shape[1]
    
    # check y grid has same shape
    assert Y.shape[0] == N_x
    assert Y.shape[1] == N_y
    
    # empty array for los velocities
    v_los = np.zeros((N_x, N_y))
    
    # loop over all points finding line of sight component of velocity
    for i in range(N_x):
        for j in range(N_y):
            v_los[i,j] = np.dot(v_field[:,i,j], proj_los)
            
    # plot velocity along line of sight
    plt.contourf(X, Y, 1e3*v_los, levels=np.linspace(-200,200,199), cmap="RdBu")
    plt.title("line of sight velocity")
    plt.colorbar()
    plt.show()
    
    ### rotate meshgrids to line of sight coordinates
    
    # rotation matrices
    rot_pl_z = Rot(  planet_az, "z")
    rot_in_x = Rot(inclination, "x")
    rot_pa_z = Rot(         PA, "z")
    
    
    for i in range(N_x):
        for j in range(N_y):

            # rotate around the normal axis of the disk, corresponding the planet_az angle
            X[i,j], Y[i,j], _   = np.dot(rot_pl_z, [X[i,j], Y[i,j], 0])

            # rotate around the x-axis of the sky plane to match the inclination
            #X[i,j], Y[i,j], _   = np.dot(rot_in_x, [X[i,j], Y[i,j], 0])
            
    # plot velocity along line of sight
    plt.contourf(X, Y, 1e3*v_los, levels=np.linspace(-200,200,199), cmap="RdBu")
    plt.title("line of sight velocity")
    plt.colorbar()
    plt.show()

    """
    
    if True:
        plt.contourf(X, Y, 1e3*v_field[0,:,:], levels=np.linspace(-200,200,199), cmap="RdBu")
        plt.title("v_x")
        plt.colorbar()
        plt.show()
        
        plt.contourf(X, Y, 1e3*v_field[1,:,:], levels=np.linspace(-200,200,199), cmap="RdBu")
        plt.title("v_y")
        plt.colorbar()
        plt.show()

    ### Rotate velocity field and grid
    
    print("Rotating velocity fields")

    # grid shape from x grid
    N_x = X.shape[0]
    N_y = X.shape[1]
    
    # check y grid has same shape
    assert Y.shape[0] == N_x
    assert Y.shape[1] == N_y

    # auxilliary angle for planet_az
    xi = np.arctan(np.tan(planet_az)/np.cos(inclination))
    if planet_az == np.pi/2 or planet_az == 3*np.pi/2:
        xi = planet_az
    elif planet_az > np.pi/2 and planet_az < 3*np.pi/2:
        xi += np.pi
    elif planet_az > 3*np.pi/2:
        xi += 2*np.pi
    
    # rotation matrices
    rot_pl_z = Rot( -planet_az, "z")
    rot_in_x = Rot(inclination, "x")
    rot_pa_z = Rot(         PA, "z")

    # loop over all points
    for i in range(N_x):
        for j in range(N_y):
            
            # rotate around the normal axis of the disk, corresponding the planet_az angle
            #X[i,j], Y[i,j], _   = np.dot(rot_pl_z, [X[i,j], Y[i,j], 0])
            #v_field[:,i,j]      = np.dot(rot_pl_z, v_field[:,i,j])
            
            pos = np.array([X[i,j], Y[i,j], 0])
            
            pos = rotation(pos, PA, inclination)
            v_field[:,i,j] = rotation(v_field[:,i,j], PA, inclination)
            
            X[i,j], Y[i,j] = pos[0], pos[1]
            
            # analytics
            
            A_pos = np.array([A_X[i,j], A_Y[i,j], 0])
            
            A_pos = rotation(A_pos, PA, inclination)
            A_v_field[:,i,j] = rotation(A_v_field[:,i,j], PA, inclination)
            
            A_X[i,j], A_Y[i,j] = A_pos[0], A_pos[1]
            
            """
            # rotate around the x-axis of the sky plane to match the inclination
            X[i,j], Y[i,j], _   = np.dot(rot_in_x, [X[i,j], Y[i,j], 0])
            v_field[:,i,j]      = np.dot(rot_in_x, v_field[:,i,j])

            # rotate around the normal axis of the sky plane to match the PA
            X[i,j], Y[i,j], _   = np.dot(rot_pa_z, [X[i,j], Y[i,j], 0])
            v_field[:,i,j]      = np.dot(rot_pa_z, v_field[:,i,j])
            """
    
    lim = 6000
    ### plot z-axis velocities
    
    plt.contourf(X, Y, 1e3*v_field[2,:,:], levels=np.linspace(-5000,5000,499), cmap="RdBu")
    plt.contour(X, Y, 1e3*v_field[2,:,:], levels=np.linspace(-lim,lim,39), colors=['k'])
    plt.title("v_z")
    plt.colorbar()
    plt.show()
    
    plt.contourf(X, Y, 1e3*v_field[0,:,:], levels=np.linspace(-lim,lim,199), cmap="RdBu")
    #plt.contourf(X, Y, 1e3*v_field[2,:,:], levels=np.linspace(-5000,5000,199), cmap="RdBu")
    plt.title("v_x")
    plt.colorbar()
    plt.show()
    
    plt.contourf(X, Y, 1e3*v_field[1,:,:], levels=np.linspace(-lim,lim,199), cmap="RdBu")
    #plt.contourf(X, Y, 1e3*v_field[2,:,:], levels=np.linspace(-5000,5000,199), cmap="RdBu")
    plt.title("v_y")
    plt.colorbar()
    plt.show()
    

def Rot(ang, ax='x'):
    """Function for rotation matrices"""
    
    # get ang in [-2pi, 2pi]
    ang = ang % (2*np.pi)

    # get phi in [0, 2pi]
    if ang < 0:
        ang = 2*np.pi + ang
    
    if ax == "x":
        return [
            [1,           0,            0],
            [0, np.cos(ang), -np.sin(ang)],
            [0, np.sin(ang),  np.cos(ang)]
        ]
    elif ax == "y":
        return [
            [ np.cos(ang), 0, np.sin(ang)],
            [ 0,           1,           0],
            [-np.sin(ang), 0, np.cos(ang)]
        ]
    elif ax == "z":
        return [
            [np.cos(ang), -np.sin(ang), 0],
            [np.sin(ang),  np.cos(ang), 0],
            [0,                      0, 1]
        ]
    else:
        raise ValueError("ax must be x, y or z")
    
def unit_vector(theta, phi):
    """
    Find a unit vector in the direction of theta, phi, 
    where phi is azimuth from x axis, and theta is measured
    down from z axis
    
    Give angles in radians
    """
    # get theta in [-2pi, 2pi]
    theta = theta % (2*np.pi)
    
    # get theta in [0, 2pi]
    if theta < 0:
        theta = 2*np.pi + theta
    
    # get theta in [0, pi] by rotating phi if needed
    if theta > np.pi:
        theta = 2*np.pi - theta
        phi = phi + np.pi

    # get phi in [-2pi, 2pi]
    phi = phi % (2*np.pi)

    # get phi in [0, 2pi]
    if phi < 0:
        phi = 2*np.pi + phi
    
    print(theta,phi)
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    
    return [x, y, z]
    
def rotate_stuff_daniel(u, v, theta):
    
    # normalise v
    k = v / np.sqrt(np.dot(v,v))

    # Rodrigues rotation formula
    w = np.cross(k, u)
    
    return u*np.cos(theta) + w*np.sin(theta) + k*np.dot(k,u)*(1-np.cos(theta))
    
def rotation(vector, PA, inc):
    
    k = np.array([np.sin(PA), np.cos(PA), 0.])
    
    return rotate_stuff_daniel(vector, k, inc)
    
    
if __name__ == '__main__':
    main()