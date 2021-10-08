import numpy as np

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