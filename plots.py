# import libraries
import numpy                as np
import matplotlib.pyplot    as plt

def spiral_v_contours_plot(
    
    spiral          = None,
    spiral_max      = None,
    contours        = None,
    contour_cmap    = None,
    contour_levels  = None,
    lim             = 4.5
):

    # setup the plot and axis labels
    plt.figure(figsize=(10,10))
    plt.xlabel('$\Delta$ RA ["]')
    plt.ylabel('$\Delta$ Dec ["]')
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    
    if spiral is not None:
        
        s = spiral
        
        # set up the levels
        try:
            max = np.max(s.v_field[2,:,:])
            levels = np.linspace(-max, max, 299)
        except:
            levels = np.linspace(-spiral_max, spiral_max, 299)
        
        if spiral_max is not None:
            levels = np.linspace(-spiral_max, spiral_max, 299)
        
        # plot the spiral
        if s.type == "observations":
            plt.contourf(-s.X, s.Y, s.data, levels=levels, cmap="RdBu")
        else:
            plt.contourf(s.X, s.Y, s.v_field[2,:,:], levels=levels, cmap="RdBu")

    if contours is not None:
        
        c = contours
        
        # if using observations
        if contours.type == "observations":
            
            if contour_cmap is None:
                plt.contour(-c.X, c.Y, c.data, origin='lower', levels=contour_levels, colors=['k'], linestyles=["-"], linewidths=0.4)
            else:
                plt.contour(-c.X, c.Y, c.data, origin='lower', levels=contour_levels, cmap=contour_cmap, linestyles=["-"], linewidths=0.4)
                
        else:
            
            if contour_cmap is None:
                plt.contour(c.X, c.Y, c.v_field[2,:,:], levels=contour_levels, colors=['k'], linestyles=["-"], linewidths=0.4)
            else:
                plt.contour(c.X, c.Y, c.v_field[2,:,:], levels=contour_levels, cmap=contour_cmap, linestyles=["-"], linewidths=0.4)
                
    plt.show()