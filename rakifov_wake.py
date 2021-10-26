# import libraries
import numpy                as np
import matplotlib.pyplot    as plt

from rotations  import rotation_matrix
from flaring    import get_height, spatial_to_angular

class RafikovWake():

    angular_coords_bool = False

    def __init__(self, hrp=0.1, q=0.25, Rp=260, gap=False, rin=100, rout=600, npoints=1000, spacing=1.0, cw=1):

        R = self.lin_array(rin, rout, npoints, spacing=spacing)

        term1 =  ((R / Rp)**(q - 0.5)) / (q - 0.5)
        term2 = -((R / Rp)**(q + 1)) / (q + 1)
        term3 = -3 / ((2*q - 1) * (q + 1))

        PHI = cw * np.sign(R - Rp) * (hrp**-1) * (term1 + term2 + term3)

        if gap:
            planet_ring_R   = Rp*np.ones(360)
            planet_ring_PHI = np.linspace(0, 2*np.pi, 360)
            R   = np.concatenate((R, planet_ring_R))
            PHI = np.concatenate((PHI, planet_ring_PHI))

        self.X = R * np.cos(PHI)
        self.Y = R * np.sin(PHI)

    def lin_array(self, lower_bound, upper_bound, steps, spacing=1.0):
        span = (upper_bound - lower_bound)
        dx = 1.0/(steps - 1)
        return np.array([lower_bound + (i*dx)**spacing*span for i in range(steps)])

    def rotate_wake(self, PA, planet_az, inc, distance):

        Z = get_height(self.X, self.Y, distance=distance)

        # print("Rotating wake")

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

    def deproject_wake(self, PA, planet_az, inc, distance):

        Z = get_height(self.X, self.Y, distance=distance)

        # print("Rotating wake")

        # get number of points
        N_x = self.X.shape[0]
        assert self.Y.shape[0] == N_x

        # convert to radians
        PA          *= np.pi / 180.
        planet_az   *= np.pi / 180.
        inc         *= np.pi / 180.

        # rotation matrices
        rot_pl_z = rotation_matrix(planet_az, "z")
        rot_in_x = rotation_matrix(      -inc, "x")
        rot_pa_z = rotation_matrix(       -PA, "z")

        # loop over all points
        for i in range(N_x):
            # rotate around the normal axis of the sky plane to match the PA
            self.X[i], self.Y[i], Z[i]    = np.dot(rot_pa_z, [self.X[i], self.Y[i], Z[i]])

            # rotate around the x-axis of the sky plane to match the inclination
            self.X[i], self.Y[i], Z[i]    = np.dot(rot_in_x, [self.X[i], self.Y[i], Z[i]])

            # rotate around the normal axis of the disk, corresponding the planet_az angle
            self.X[i], self.Y[i], Z[i]    = np.dot(rot_pl_z, [self.X[i], self.Y[i], Z[i]])


    def get_height(self, X, Y, distance=101.5):

        if X.ndim == 2:

            R = np.sqrt(X**2 + Y**2)

            R = spatial_to_angular(R, distance)

            H = np.zeros(R.shape)

            #H = hr * R #* ( R / Rp )**(0.5 - q)
            for i in range(R.shape[0]):
                for j in range(R.shape[1]):

                    if R[i,j] < 2.4:
                        H[i,j] = 0.28247214678877486 * R[i,j]**1.278581229271081
                    else:
                        H[i,j] = 0.28247214678877486 * 2.4**1.278581229271081

            H = angular_to_spatial(H, distance)

        elif X.ndim == 1:

            R = np.zeros(X.shape)
            H = np.zeros(R.shape)

            for i in range(len(X)):

                R[i] = np.sqrt(X[i]**2 + Y[i]**2)
                R[i] = spatial_to_angular(R[i], distance)

                if R[i] < 2.4:
                        H[i] = 0.28247214678877486 * R[i]**1.278581229271081
                else:
                    H[i] = 0.28247214678877486 * 2.4**1.278581229271081

                H[i] = angular_to_spatial(H[i], distance)

        return H

    def spatial_to_angular(self, X, distance):

        # convert distance in pc to au
        D = distance * 206265

        # convert to arcseconds
        X = (3600 * X / D) * (180. / np.pi)

        return X

    def angular_to_spatial(self, X, distance):

        # convert distance in pc to au
        D = distance * 206265

        # convert to au
        X = (D * X * np.pi) / (3600 * 180)

        return X

    def angular_coords(self, distance):

        if not self.angular_coords_bool:

            self.X = spatial_to_angular(self.X, distance)
            self.Y = spatial_to_angular(self.Y, distance)

            self.angular_coords_bool = True

        else:

            print("already in angular coords")
