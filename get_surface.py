from disksurf import observation
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline


cube = observation('../observations/HD_163296_CO_220GHz.0.15arcsec.image.fits', FOV=10.0, velocity_range=[0.0, 12e3])

inc = 46.7   # deg
PA = 312.0   # deg

surface = cube.get_emission_surface(inc=inc,
                                    PA=PA,
                                    smooth=1.0)

surface.mask_surface(side='front', min_zr=0.07, max_zr=1.0)
surface.mask_surface(side='back', min_zr=-1.0, max_zr=0.0)

# r, z, dz = surface.rolling_surface()

r, z, dz = surface.binned_surface()

arr = np.array([r, z, dz])
np.save('binned_surface', arr)

w = 1/dz

print(w)

spline = UnivariateSpline(r, z, w=w, k=4)
r_spline = np.linspace(np.min(r), np.max(r), 1000)

plt.errorbar(r, z, dz, label='Binned Surface')
plt.plot(r_spline, spline(r_spline), color='k', label='Spline Fit')
plt.savefig('split_fit.png')
plt.show()

# plot the surface
fig = surface.plot_surface(side='front', return_fig=True)

# set the blue points to gray
for i in range(3):
    fig.axes[0].get_children()[i].set_facecolor('0.8')
    fig.axes[0].get_children()[i].set_edgecolor('0.8')
fig.axes[0].get_children()[1].set_label('data')

# fit an exponentitally tapered power law model and plot
median = surface.fit_emission_surface_MCMC(side='front',
                                            tapered_powerlaw=True,
                                            include_cavity=False, returns='median',
                                            plots=['corner', 'walkers'],
                                            p0=[0.26, 1.5, 3.9, 2.5])

print(median)
plt.show()

np.save('tapered_surface', median)

# # update legend
# fig.axes[0].legend()
# plt.savefig('tapered_fit.png')
# plt.show()
