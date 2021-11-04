from velocity_field import VelocityField
from plots          import spiral_v_contours_plot
from rakifov_wake   import RafikovWake
from flaring        import angular_to_spatial, spatial_to_angular
from rotations  import rotation_matrix
import matplotlib.pylab as plt
import numpy            as np
import emcee
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from multiprocessing import Pool
from scipy.interpolate import UnivariateSpline, interp1d


class FitSpiral():
    def __init__(self, data=None, uncertainties=None, rin=100, rout=700, spacing=1.0, name=None, height='fit', npoints = 100):

        self.data = data
        self.uncertainties = uncertainties
        self.ivar = 1/self.uncertainties**2
        self.name = name
        self.rin = rin
        self.rout = rout
        self.spacing = spacing
        self.height = height
        self.npoints = npoints

        self.priors = {}
        self._set_default_priors()


    def fit(self, p0, params, nwalkers, nburnin, nsteps, plots=None, savefigs=True, pool=None):
        sampler = self._run_mcmc(p0, params, nwalkers, nburnin, nsteps, pool=pool)
        idx = np.argmin(np.concatenate(sampler.lnprobability.T[nburnin:]))
        samples = sampler.get_chain(discard=nburnin, flat=True)
        p0 = np.median(samples, axis=0)
        p0 = samples[idx]
        max_likelihood = self._populate_dictionary(p0, params)
        self.max_likelihood_params = max_likelihood
        p0 = np.median(samples, axis=0)
        self.median = self._populate_dictionary(p0, params)
        print(max_likelihood)

        labels = FitSpiral._get_labels(params)
        labels_raw = []
        for label in labels:
            label_raw = label.replace('$', '').replace('{', '')
            label_raw = label_raw.replace(r'\rm ', '').replace('}', '')
            labels_raw += [label_raw]
        if len(labels) != len(p0):
            raise ValueError("Mismatch in labels and p0. Check for integers.")
        print("Assuming:\n\tp0 = [%s]." % (', '.join(labels_raw)))

        if plots is None:
            plots = ['walkers', 'corner', 'bestfit', 'residual', 'lnprob', 'best_fit']
        plots = np.atleast_1d(plots)

        if savefigs is not None:
            savename = self.name
        if 'none' in plots:
            plots = []
        if 'walkers' in plots:
            walkers = sampler.chain.T
            self.plot_walkers(walkers, nburnin, labels, savename=savename)
        if 'lnprob' in plots:
            self.plot_walkers(np.expand_dims(sampler.lnprobability.T, 0), nburnin, ['lnprob'], histogram=False, savename=savename)
        if 'corner' in plots:
            self.plot_corner(samples, labels, savename=savename)
        if 'best_fit' in plots:
            self.plot_model(self.median, self.data, savename=savename)



    def plot_walkers(self, samples, nburnin=None, labels=None, histogram=True, savename=None):
        """
        Plot the walkers to check if they are burning in.

        Args:
            samples (ndarray):
            nburnin (Optional[int])
        """

        # Check the length of the label list.

        if labels is not None:
            if samples.shape[0] != len(labels):
                raise ValueError("Not correct number of labels.")

        # Cycle through the plots.

        for s, sample in enumerate(samples):
            fig, ax = plt.subplots()
            for walker in sample.T:
                ax.plot(walker, alpha=0.1, color='k')
            ax.set_xlabel('Steps')
            if labels is not None:
                ax.set_ylabel(labels[s])
            if nburnin is not None:
                ax.axvline(nburnin, ls=':', color='r')

            # Include the histogram.

            if histogram:
                fig.set_size_inches(1.37 * fig.get_figwidth(),
                                    fig.get_figheight(), forward=True)
                ax_divider = make_axes_locatable(ax)
                bins = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 50)
                hist, _ = np.histogram(sample[nburnin:].flatten(), bins=bins,
                                       density=True)
                bins = np.average([bins[1:], bins[:-1]], axis=0)
                ax1 = ax_divider.append_axes("right", size="35%", pad="2%")
                ax1.fill_betweenx(bins, hist, np.zeros(bins.size), step='mid',
                                  color='darkgray', lw=0.0)
                ax1.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
                ax1.set_xlim(0, ax1.get_xlim()[1])
                ax1.set_yticklabels([])
                ax1.set_xticklabels([])
                ax1.tick_params(which='both', left=0, bottom=0, top=0, right=0)
                ax1.spines['right'].set_visible(False)
                ax1.spines['bottom'].set_visible(False)
                ax1.spines['top'].set_visible(False)

            if savename is not None:
                plt.savefig('{0}_{1}.png'.format(savename, labels[s]), dpi=300)


    def plot_corner(self, samples, labels=None, quantiles=[0.16, 0.5, 0.84], savename=None):
        """
        A wrapper for DFM's corner plots.

        Args:
            samples (ndarry):
            labels (Optional[list]):
            quantiles (Optional[list]): Quantiles to use for plotting.
        """
        import corner
        corner.corner(samples, labels=labels, title_fmt='.4f', bins=30,
                      quantiles=quantiles, show_titles=True)

        if savename is not None:
            plt.savefig('{0}_corner.png'.format(savename), dpi=300)


    def random_p0(self, p0, scatter, nwalkers):
        """Introduce scatter to starting positions."""
        p0 = np.squeeze(p0)
        dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
        dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
        return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)

    @staticmethod
    def _populate_dictionary(theta, dictionary_in):
        """Populate the dictionary of free parameters."""
        dictionary = dictionary_in.copy()
        for key in dictionary.keys():
            if isinstance(dictionary[key], int):
                if not isinstance(dictionary[key], bool):
                    dictionary[key] = theta[dictionary[key]]
        return dictionary

    @staticmethod
    def _get_labels(params):
        """Return the labels of the parameters to fit."""
        idxs, labs = [], []
        for k in params.keys():
            if isinstance(params[k], int):
                if not isinstance(params[k], bool):
                    idxs.append(params[k])
                    try:
                        idx = k.index('_') + 1
                        label = k[:idx] + '{{' + k[idx:] + '}}'
                    except ValueError:
                        label = k
                    label = r'${{\rm {}}}$'.format(label)
                    labs.append(label)
        return np.array(labs)[np.argsort(idxs)]


    def plot_model(self, params, data, savename=None):

        model = Spiral(hrp=params['hrp'], q=params['q'], Rp=params['Rp'], inc=params['inc'],
                            planet_az=params['planet_az'], PA=params['PA'], distance=params['distance'],
                            rin=self.rin, rout=self.rout, npoints=100, spacing=self.spacing,
                            height=self.height)
        model_data = np.array([model.X, model.Y])
        plt.figure()
        plt.scatter(data[0], data[1], label='data')
        plt.plot(model_data[0], model_data[1])
        plt.xlim([5.5,-5.5])
        plt.ylim([-5.5, 5.5])
        plt.legend()
        if savename is not None:
            plt.savefig('{0}_{1}.png'.format(savename, 'best_fit'), dpi=300)

    def get_geometric_distance(self, model, data):
        # print(model)
        # print(data)
        # print(np.shape(data))
        # print(np.shape(model))
        # plt.figure()
        # plt.plot(model[0], model[1])
        # plt.plot(data[0], data[1])
        # plt.show()
        distance = []
        for i in range(np.shape(data)[1]):
            # print(i)
            eu_distance = [np.linalg.norm(data[:, i] - model_i) for model_i in model.T]
            # print(eu_distance)
            smallest = np.argmin(eu_distance)
            distance.append(eu_distance[smallest])

        return distance

    def _ln_likelihood(self, params):
        """Log-likelihood function. Simple chi-squared likelihood."""
        model = Spiral(hrp=params['hrp'], q=params['q'], Rp=params['Rp'], inc=params['inc'],
                            planet_az=params['planet_az'], PA=params['PA'], distance=params['distance'],
                            rin=self.rin, rout=self.rout, npoints=self.npoints, height=self.height)
        model_data = np.array([model.X, model.Y])
        residual = self.get_geometric_distance(model_data, self.data)
        lnx2 = np.power(residual, 2)
        lnx2 = -0.5 * np.sum(lnx2 * self.ivar)
        return lnx2 if np.isfinite(lnx2) else -np.inf

    def _ln_probability(self, theta, *params_in):
        """Log-probablility function."""
        model = self._populate_dictionary(theta, params_in[0])
        lnp = self._ln_prior(model)
        # return  self._ln_likelihood(model)
        if np.isfinite(lnp):
            return lnp + self._ln_likelihood(model)
        return -np.inf

    def _set_default_priors(self):
        """Set the default priors."""

        # Basic Geometry.

        self.set_prior('q', [0.0, 0.75], 'flat')
        self.set_prior('rp', [10, 500], 'flat')
        self.set_prior('hrp', [0.01, 0.5], 'flat')
        self.set_prior('inc', [-360.0, 360.0], 'flat')
        self.set_prior('PA', [-360.0, 360.0], 'flat')
        self.set_prior('planet_az', [-360.0, 360.0], 'flat')


    def set_prior(self, param, args, type='flat'):
        """
        Set the prior for the given parameter. There are two types of priors
        currently usable, ``'flat'`` which requires ``args=[min, max]`` while
        for ``'gaussian'`` you need to specify ``args=[mu, sig]``.

        Args:
            param (str): Name of the parameter.
            args (list): Values to use depending on the type of prior.
            type (optional[str]): Type of prior to use.
        """
        type = type.lower()
        if type not in ['flat', 'gaussian']:
            raise ValueError("type must be 'flat' or 'gaussian'.")
        if type == 'flat':
            def prior(p):
                if not min(args) <= p <= max(args):
                    return -np.inf
                return np.log(1.0 / (args[1] - args[0]))
        else:
            def prior(p):
                return -0.5 * ((args[0] - p) / args[1])**2
        self.priors[param] = prior

    def _ln_prior(self, params):
        """Log-priors."""
        lnp = 0.0
        for key in params.keys():
            if key in self.priors.keys() and params[key] is not None:
                lnp += self.priors[key](params[key])
                if not np.isfinite(lnp):
                    return lnp
        return lnp

    def _run_mcmc(self, p0, params, nwalkers, nburnin, nsteps, **kwargs):
        """Run the MCMC sampling. Returns the sampler."""

        EnsembleSampler = emcee.EnsembleSampler

        p0 = self.random_p0(p0, kwargs.pop('scatter', 1e-3), nwalkers)
        moves = kwargs.pop('moves', None)
        pool = kwargs.pop('pool', None)

        sampler = EnsembleSampler(nwalkers,
                                  p0.shape[1],
                                  self._ln_probability,
                                  # self._ln_likelihood,
                                  args=[params, np.nan],
                                  moves=moves, #,
                                  pool=pool)

        progress = kwargs.pop('progress', True)

        sampler.run_mcmc(p0, nburnin + nsteps, progress=progress, **kwargs)

        return sampler


class Spiral(RafikovWake):
    def __init__(self, hrp=0.08, q=0.25, Rp=260, inc=45,
                        planet_az=45, PA=45, distance=100, rin=10, rout=500,
                        npoints=1000, spacing=1.0, height='spline', bottom=False):

        self.hrp = hrp
        self.q = q
        self.Rp = Rp
        self.inc = inc
        self.planet_az = planet_az
        self.PA = PA
        self.distance = distance
        self.height = height
        self.bottom = bottom

        if self.height == 'spline':
            arr = np.load('binned_surface.npy')
            r, z, dz = arr

            w = 1/dz
            spline = UnivariateSpline(r, z, w=w, k=4)

            def height(r):
                h = spline(r)
                h[h<0] = 0
                return h

            self.height_func = height

        elif self.height == 'linear':
            arr = np.load('binned_surface.npy')
            r, z, dz = arr
            interp = interp1d(r, z, bounds_error=False, fill_value=np.nan, kind='cubic')
            self.height_func = interp

        elif self.height == 'simple':
            self.height_func = self.height_simple

        elif self.height == 'tapered':
            # arr = np.load('tapered_surface.npy')
            # print(arr)
            arr = [0.388, 1.851, 2.362, 1.182]

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
                return tapered_powerlaw(r, arr[0], arr[1], r_taper=arr[2], q_taper=arr[3], r_cavity=0.0,
                                      r0=1.0)

            self.height_func = height


        super(Spiral, self).__init__(hrp=self.hrp, q=self.q, Rp=self.Rp, gap=False, rin=rin, rout=rout, npoints=npoints, spacing=spacing)

        self.rotate_wake()
        self.angular_coords(self.distance)


    def rotate_wake(self):

        Z = self.get_height(self.X, self.Y)
        if self.bottom:
            Z = -Z

        # get number of points
        N_x = self.X.shape[0]
        assert self.Y.shape[0] == N_x

        # convert to radians
        PA          = self.PA * np.pi / 180.
        planet_az   = self.planet_az * np.pi / 180.
        inc         = self.inc * np.pi / 180.

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

    def get_height(self, X, Y):

        if X.ndim == 2:
            R = np.sqrt(X**2 + Y**2)
            R = spatial_to_angular(R, self.distance)
            H = np.zeros(R.shape)
            H = self.height_func(R)
            H = angular_to_spatial(H, self.distance)

        elif X.ndim == 1:

            R = np.sqrt(X**2 + Y**2)
            R = spatial_to_angular(R, self.distance)
            H = self.height_func(R)
            H = angular_to_spatial(H, self.distance)

        return H

    def height_simple(self, R):
        H = np.zeros(np.shape(R))
        for i in range(len(R)):
            if R[i] < 2.4:
                H[i] = 0.28247214678877486 * R[i]**1.278581229271081
            else:
                 H[i] =  0.28247214678877486 * 2.4**1.278581229271081

        return H
