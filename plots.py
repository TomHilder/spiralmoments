# import libraries
import numpy                as np
import matplotlib.pyplot    as plt
from numpy.__config__ import show

def spiral_v_contours_plot(
    
    spiral          = None,
    spiral_max      = None,
    spiral_levels   = None,
    contours        = None,
    contour_cmap    = None,
    contour_levels  = None,
    lim             = 4.5,
    show            = True,
    rafikov_wake    = None,
    gap             = False,
    save            = None
):

    # setup the plot and axis labels
    plt.figure(figsize=(7,7), dpi=150)
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
            try:
                levels = np.linspace(-spiral_max, spiral_max, 299)
            except:
                levels = spiral_levels
        
        if spiral_max is not None:
            levels = np.linspace(-spiral_max, spiral_max, 299)
            
        if spiral_levels is not None:
            levels = spiral_levels
        
        # plot the spiral
        if s.type == "observations" or s.type == "line":
            plt.contourf(-s.X, s.Y, s.data, levels=levels, cmap="RdBu")
        else:
            plt.contourf(s.X, s.Y, s.v_field[2,:,:], levels=levels, cmap="RdBu")
            
        plt.colorbar(label="Line of sight velocity [m/s]")

    if contours is not None:
        
        c = contours
        
        # if using observations
        if c.type == "observations" or c.type == "line":
            
            if contour_cmap is None:
                plt.contour(-c.X, c.Y, c.data, origin='lower', levels=contour_levels, colors=['k'], linestyles=["-"], linewidths=0.4)
            else:
                plt.contour(-c.X, c.Y, c.data, origin='lower', levels=contour_levels, cmap=contour_cmap, linestyles=["-"], linewidths=0.4)
                
        else:
            
            if contour_cmap is None:
                plt.contour(c.X, c.Y, c.v_field[2,:,:], levels=contour_levels, colors=['k'], linestyles=["-"], linewidths=0.8)
            else:
                plt.contour(c.X, c.Y, c.v_field[2,:,:], levels=contour_levels, cmap=contour_cmap, linestyles=["-"], linewidths=0.4)
                
    if rafikov_wake is not None:
        
        r = rafikov_wake
        
        if gap:
            plt.plot(r.X[:-360], r.Y[:-360], c="k", alpha=0.2, ls="--")
            plt.plot(r.X[-360:], r.Y[-360:], c="k", alpha=0.4, ls="-")
        else:
            plt.plot(r.X[:-360], r.Y[:-360], c="k", alpha=0.2, ls="--")
    
    plt.gca().set_aspect('equal', adjustable='box')
    
    if save is not None:
        
        plt.savefig(f"{save}.pdf")
        
    if show:
        plt.show()