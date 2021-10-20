# import libraries
import numpy                as np
import matplotlib.pyplot    as plt

from rotations  import rotation_matrix
from flaring    import get_height, spatial_to_angular

class RafikovWake():
    
    angular_coords_bool = False
    
    def __init__(self, hrp=0.1, q=0.25, Rp=260, gap=False, cw=1):
        
        R = np.linspace(10, 1000, 1000)
        
        term1 =  ((R / Rp)**(q - 0.5)) / (q - 0.5)
        term2 = -((R / Rp)**(q + 1)) / (q + 1)
        term3 = -3 / ((2*q - 1) * (q + 1))
        
        PHI = cw * np.sign(R - Rp) * (hrp**-1) * (term1 + term2 + term3)
        
        if gap:
            planet_ring_R   = Rp*np.ones(360)
            planet_ring_PHI = np.linspace(0, 2*np.pi, 360)
            R   = np.concatenate((R, planet_ring_R))
            PHI = np.concatenate((PHI, planet_ring_PHI))
        
        self.X = np.zeros((len(R)))
        self.Y = np.zeros((len(R)))

        for i in range(len(R)):
            self.X[i] = R[i] * np.cos(PHI[i])
            self.Y[i] = R[i] * np.sin(PHI[i])
        
    def rotate_wake(self, PA, planet_az, inc, distance):
        
        Z = get_height(self.X, self.Y, distance=distance)
        
        print("Rotating wake")
        
        # get number of points
        N_x = self.X.shape[0]
        assert self.Y.shape[0] == N_x
        
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
                
            # rotate around the normal axis of the disk, corresponding the planet_az angle
            self.X[i], self.Y[i], Z[i]    = np.dot(rot_pl_z, [self.X[i], self.Y[i], Z[i]])
            
            # rotate around the x-axis of the sky plane to match the inclination
            self.X[i], self.Y[i], Z[i]    = np.dot(rot_in_x, [self.X[i], self.Y[i], Z[i]])

            # rotate around the normal axis of the sky plane to match the PA
            self.X[i], self.Y[i], Z[i]    = np.dot(rot_pa_z, [self.X[i], self.Y[i], Z[i]])
            
    def angular_coords(self, distance):
        
        if not self.angular_coords_bool:
        
            self.X = spatial_to_angular(self.X, distance)
            self.Y = spatial_to_angular(self.Y, distance)
            
            self.angular_coords_bool = True

        else:
            
            print("already in angular coords")