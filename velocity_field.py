# import libraries
import numpy                as np
import matplotlib.pyplot    as plt

from astropy.io import fits
from rotations  import rotation_matrix

# class for reading in and manipulating velocity fields
class VelocityField():
    
    def __init__(
        self,
        type        = "wakeflow",
        name        = "",
        phantom_R   = 500,
        obs_sys_v   = 5.750,
        obs_dist    = 101.5
    ):
        
        self.type = type
        self.distance = obs_dist
        
        print(f"Constructing VelocityField object of type {type} using {name}")

        if type == "wakeflow":
            
            # read in fields (midplane only)
            v_r     = np.load(f"wakeflow/{name}/v_r.npy")[:,0,:]
            v_phi   = np.load(f"wakeflow/{name}/v_phi.npy")[:,0,:]
            
            # read in grid (midplane again)
            self.X  = np.load(f"wakeflow/{name}/X.npy")[:,0,:]
            self.Y  = np.load(f"wakeflow/{name}/Y.npy")[:,0,:] 

        elif type == "phantom":
            
            # read in fields
            v_r     = np.loadtxt(f"phantom/{name}/vr.pix").transpose()
            v_phi   = np.loadtxt(f"phantom/{name}/vphi.pix").transpose()
            
            # create grid
            length = v_r.shape[0]
            ax = np.linspace(-phantom_R, phantom_R, length)
            self.X, self.Y = np.meshgrid(ax, ax)
            
        elif type == "observations":
            
            # open file
            file = f"observations/{name}"
            moment_1_fits = fits.open(file)
            
            # read data
            self.data = moment_1_fits[0].data - obs_sys_v*1e3
            
            # get header info to get grid
            header = moment_1_fits[0].header
            naxis1 = header['NAXIS1']
            naxis2 = header['NAXIS2']
            cdelt1 = header['CDELT1']
            cdelt2 = header['CDELT2']
            crpix1 = header['CRPIX1']
            crpix2 = header['CRPIX2']
            
            # create grid
            midpoint_ra     = crpix1 * cdelt1 * 3600
            midpoint_dec    = crpix2 * cdelt2 * 3600
            self.X = np.linspace(-midpoint_ra, midpoint_ra, naxis1)
            self.Y = np.linspace(-midpoint_dec, midpoint_dec, naxis2)
        
        else:
            raise ValueError("type must be wakeflow, phantom or observations")
        
        if type != "observations":
            
            # get meshgrids for polar coordinates
            R = np.sqrt(self.X**2 + self.Y**2)
            PHI = np.arctan2(self.Y, self.X)

            # perform transformations
            v_x = -v_phi * np.sin(PHI) + v_r * np.cos(PHI)
            v_y = v_phi * np.cos(PHI) + v_r * np.sin(PHI)
            v_z = np.zeros(v_x.shape)
            
            # define velocity field (convert to m/s)
            self.v_field = 1e3 * np.array([v_x, v_y, v_z])
    
    def rotate(
        self, 
        PA          = 0,
        inc         = 0,
        planet_az   = 0
    ):
        
        print("Rotating velocity field")
        
        # check if type is observations (we don't want to rotate those)
        if self.type == "observations":
            raise TypeError("cannot rotate observations type VelocityField")
        
        # get number of points
        N_x = self.X.shape[0]
        N_y = self.X.shape[1]
        assert self.Y.shape[0] == N_x
        assert self.Y.shape[1] == N_y
        
        # convert to radians
        PA          *= np.pi / 180.
        planet_az   *= np.pi / 180.
        inc         *= np.pi / 180.
        
        # rotation matrices
        rot_pl_z = rotation_matrix(-planet_az, "z")
        rot_in_x = rotation_matrix(       inc, "x")
        rot_pa_z = rotation_matrix(        PA, "z")
        
        # loop over all points
        for i in range(N_x):
            for j in range(N_y):
                
                # rotate around the normal axis of the disk, corresponding the planet_az angle
                self.X[i,j], self.Y[i,j], _     = np.dot(rot_pl_z, [self.X[i,j], self.Y[i,j], 0])
                self.v_field[:,i,j]             = np.dot(rot_pl_z, self.v_field[:,i,j])
                
                # rotate around the x-axis of the sky plane to match the inclination
                self.X[i,j], self.Y[i,j], _     = np.dot(rot_in_x, [self.X[i,j], self.Y[i,j], 0])
                self.v_field[:,i,j]             = np.dot(rot_in_x, self.v_field[:,i,j])

                # rotate around the normal axis of the sky plane to match the PA
                self.X[i,j], self.Y[i,j], _     = np.dot(rot_pa_z, [self.X[i,j], self.Y[i,j], 0])
                self.v_field[:,i,j]             = np.dot(rot_pa_z, self.v_field[:,i,j])
                
    def convert_to_angular_coordinates(self):
        
        # convert distance in pc to au
        D = self.distance * 206265
        
        # convert coordinates to angular coordinates in arcseconds
        self.X = (3600 * self.X / D) * (180. / np.pi)
        self.Y = (3600 * self.Y / D) * (180. / np.pi)