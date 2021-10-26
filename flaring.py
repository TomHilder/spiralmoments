import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline

def get_height_func(height):
    if height == 'spline':
        arr = np.load('binned_surface.npy')
        r, z, dz = arr

        w = 1/dz
        spline = UnivariateSpline(r, z, w=w, k=4)

        def height_func(r):
            h = spline(r)
            h[h<0] = 0
            return h

    elif height == 'linear':
        arr = np.load('binned_surface.npy')
        r, z, dz = arr
        interp = interp1d(r, z, bounds_error=False, fill_value=np.nan, kind='cubic')
        height_func = interp

    elif height == 'simple':
        height_func = get_height

    elif height == 'tapered':
        arr = np.load('tapered_surface.npy')

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

        def height_func(r):
            return tapered_powerlaw(r, arr[0], arr[1], r_taper=arr[2], q_taper=arr[3], r_cavity=0.0,
                                  r0=1.0)

    return height_func

def get_height(X, Y, distance=101.5):

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

def spatial_to_angular(X, distance):

    # convert distance in pc to au
    D = distance * 206265

    # convert to arcseconds
    X = (3600 * X / D) * (180. / np.pi)

    return X

def angular_to_spatial(X, distance):

    # convert distance in pc to au
    D = distance * 206265

    # convert to au
    X = (D * X * np.pi) / (3600 * 180)

    return X
