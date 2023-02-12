import numpy as np

# import warnings

# This file enumerates algorithms for segmenting one-dimensional data.
# The bulk of the code is derived from the astropy project,
# in particular Jake VanderPlas's implementation of the
# Bayesian blocks algorithm in Python.


class Algorithm(object):
    """Base class for Bayesian blocks algorithm functions

    Derived classes should overload the following method:

    ``algorithm(self, **kwargs)``:
      Compute the algorithm given a set of named arguments.
      Arguments accepted by algorithm must be among ``[T_k, N_k, a_k, b_k, c_k]``
      (See [1]_ for details on the meaning of these parameters).

    Additionally, other methods may be overloaded as well:

    ``__init__(self, **kwargs)``:
      Initialize the algorithm function with any parameters beyond the normal
      ``p0`` and ``gamma``.

    ``validate_input(self, t, x, sigma)``:
      Enable specific checks of the input data (``t``, ``x``, ``sigma``)
      to be performed prior to the fit.

    ``compute_ncp_prior(self, N)``: If ``ncp_prior`` is not defined explicitly,
      this function is called in order to define it before fitting. This may be
      calculated from ``gamma``, ``p0``, or whatever method you choose.

    ``p0_prior(self, N)``:
      Specify the form of the prior given the false-alarm probability ``p0``
      (See [1]_ for details).

    For examples of implemented algorithm functions, see :class:`Events`,
    :class:`RegularEvents`, and :class:`PointMeasures`.

    References
    ----------
    .. [1] Scargle, J et al. (2012)
       http://adsabs.harvard.edu/abs/2012arXiv1207.5578S
    """

    def __init__(self, p0=0.05, gamma=None, ncp_prior=None):
        self.p0 = p0
        self.gamma = gamma
        self.ncp_prior = ncp_prior
        self.prior = None
        self.best_fitness = None

    @staticmethod
    def validate_input(t, x=None, sigma=None):
        """Validate inputs to the model.

        Parameters
        ----------
        t : array_like
            times of observations
        x : array_like (optional)
            values observed at each time
        sigma : float or array_like (optional)
            errors in values x

        Returns
        -------
        t, x, sigma : array_like, float or None
            validated and perhaps modified versions of inputs
        """
        # validate array input
        t = np.asarray(t, dtype=float)
        if x is not None:
            x = np.asarray(x)
        if sigma is not None:
            sigma = np.asarray(sigma)

        # find unique values of t
        t = np.array(t)
        if t.ndim != 1:
            raise ValueError("t must be a one-dimensional array")
        unq_t, unq_ind, unq_inv = np.unique(t, return_index=True, return_inverse=True)

        # if x is not specified, x will be counts at each time
        if x is None:
            if sigma is not None:
                raise ValueError("If sigma is specified, x must be specified")
            else:
                sigma = 1

            if len(unq_t) == len(t):
                x = np.ones_like(t)
            else:
                x = np.bincount(unq_inv)

            t = unq_t

        # if x is specified, then we need to simultaneously sort t and x
        else:
            # TODO: allow broadcasted x?
            x = np.asarray(x)
            if x.shape not in [(), (1,), (t.size,)]:
                raise ValueError("x does not match shape of t")
            x += np.zeros_like(t)

            if len(unq_t) != len(t):
                raise ValueError(
                    "Repeated values in t not supported when " "x is specified"
                )
            t = unq_t
            x = x[unq_ind]

        # verify the given sigma value
        if sigma is None:
            sigma = 1
        else:
            sigma = np.asarray(sigma)
            if sigma.shape not in [(), (1,), (t.size,)]:
                raise ValueError("sigma does not match the shape of x")

        return t, x, sigma

    def fitness(self, **kwargs):
        raise NotImplementedError()

    # # the algorithm_args property will return the list of arguments accepted by
    # # the method algorithm().  This allows more efficient computation below.
    # @property
    # def _algorithm_args(self):
    #     return signature(self.algorithm).parameters.keys()

    def p0_prior(self, N):
        """
        Empirical prior, parametrized by the false alarm probability ``p0``
        See  eq. 21 in Scargle (2012)

        Note that there was an error in this equation in the original Scargle
        paper (the "log" was missing). The following corrected form is taken
        from http://arxiv.org/abs/1304.2818
        """
        return 4 - np.log(73.53 * self.p0 * (N**-0.478))

    def compute_ncp_prior(self, N):
        """
        If ``ncp_prior`` is not explicitly defined, compute it from ``gamma``
        or ``p0``.
        """
        if self.ncp_prior is not None:
            self.prior = self.ncp_prior
            return self.prior
        elif self.gamma is not None:
            self.prior = -np.log(self.gamma)
            return self.prior
        elif self.p0 is not None:
            self.prior = self.p0_prior(N)
            return self.prior
        else:
            raise ValueError(
                "``ncp_prior`` is not defined, and cannot compute "
                "it as neither ``gamma`` nor ``p0`` is defined."
            )

    def segment(self, t, x=None, sigma=None):
        raise NotImplementedError()


class OptimalPartitioning(Algorithm):
    r"""Bayesian blocks algorithm for regular events

    This is for data which has a fundamental "tick" length, so that all
    measured values are multiples of this tick length.  In each tick, there
    are either zero or one counts.

    Parameters
    ----------
    dt : float
        tick rate for data
    p0 : float (optional)
        False alarm probability, used to compute the prior on :math:`N_{\rm
        blocks}` (see eq. 21 of Scargle 2012). If gamma is specified, p0 is
        ignored.
    ncp_prior : float (optional)
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\tt ncp\_prior} = -\ln({\tt
        gamma})`.  If ``ncp_prior`` is specified, ``gamma`` and ``p0`` are
        ignored.
    """

    def __init__(self, p0=0.05, gamma=None, ncp_prior=None):
        # if p0 is not None and gamma is None and ncp_prior is None:
        #     warnings.warn(
        #         "p0 does not seem to accurately represent the false "
        #         "positive rate for event data. It is highly "
        #         "recommended that you run random trials on signal-"
        #         "free noise to calibrate ncp_prior to achieve a "
        #         "desired false positive rate."
        #     )
        super(OptimalPartitioning, self).__init__(p0, gamma, ncp_prior)

    def fitness(self, T_k, N_k):
        # Negative log of the Poisson maximum likelihood, given T_k and N_k
        return N_k * (np.log(N_k) - np.log(T_k))

    def segment(self, t, x=None, sigma=None):
        """Fit the Bayesian Blocks model given the specified algorithm function.

        Parameters
        ----------
        t : array_like
            data times (one dimensional, length N)
        x : array_like (optional)
            data values
        sigma : array_like or float (optional)
            data errors

        Returns
        -------
        edges : ndarray
            array containing the (M+1) edges defining the M optimal bins
        """

        t, x, sigma = self.validate_input(t, x, sigma)

        # create length-(N + 1) array of cell edges
        edges = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])
        block_length = t[-1] - edges

        # arrays to store the best configuration
        N = len(t)
        best = np.zeros(N, dtype=float)
        last = np.zeros(N, dtype=int)

        # Compute ncp_prior if not defined
        if self.ncp_prior is None:
            ncp_prior = self.compute_ncp_prior(N)
        else:
            ncp_prior = self.ncp_prior
            self.prior = self.ncp_prior
        # ----------------------------------------------------------------
        # Start with first data cell; add one cell at each iteration
        # ----------------------------------------------------------------

        for R in range(N):
            # Compute fit_vec : algorithm of putative last block (end at R)

            # T_k: width/duration of each block
            T_k = block_length[: R + 1] - block_length[R + 1]

            # N_k: number of elements in each block
            N_k = np.cumsum(x[: R + 1][::-1])[::-1]

            # evaluate algorithm function
            fit_vec = self.fitness(T_k, N_k)
            A_R = fit_vec - ncp_prior
            A_R[1:] += best[:R]

            i_max = np.argmax(A_R)
            last[R] = i_max
            best[R] = A_R[i_max]
        self.best_fitness = best[R]

        # Find change points
        return self.get_change_points(N, edges, last)

    @staticmethod
    def get_change_points(N, edges, last):
        # ----------------------------------------------------------------
        # Now find change points by iteratively peeling off the last block
        # ----------------------------------------------------------------
        change_points = np.zeros(N, dtype=int)
        i_cp = N
        ind = N
        while True:
            i_cp -= 1
            change_points[i_cp] = ind
            if ind == 0:
                break
            ind = last[ind - 1]
        change_points = change_points[i_cp:]
        return edges[change_points]


class BayesianBlocks(OptimalPartitioning):
    r"""Bayesian blocks algorithm for binned or unbinned events

    Parameters
    ----------
    p0 : float (optional)
        False alarm probability, used to compute the prior on
        :math:`N_{\rm blocks}` (see eq. 21 of Scargle 2012). For the Events
        type data, ``p0`` does not seem to be an accurate representation of the
        actual false alarm probability. If you are using this algorithm function
        for a triggering type condition, it is recommended that you run
        statistical trials on signal-free noise to determine an appropriate
        value of ``gamma`` or ``ncp_prior`` to use for a desired false alarm
        rate.
    gamma : float (optional)
        If specified, then use this gamma to compute the general prior form,
        :math:`p \sim {\tt gamma}^{N_{\rm blocks}}`.  If gamma is specified, p0
        is ignored.
    ncp_prior : float (optional)
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\tt ncp\_prior} = -\ln({\tt
        gamma})`.
        If ``ncp_prior`` is specified, ``gamma`` and ``p0`` is ignored.
    """

    def __init__(self, p0=0.05, gamma=None, ncp_prior=None):
        super(BayesianBlocks, self).__init__(p0, gamma, ncp_prior)

    def fitness(self, N_k, T_k):
        # eq. 19 from Scargle 2012
        return N_k * (np.log(N_k) - np.log(T_k))


class PELT(OptimalPartitioning):
    r"""Bayesian blocks algorithm for point measures

    Parameters
    ----------
    p0 : float (optional)
        False alarm probability, used to compute the prior on :math:`N_{\rm
        blocks}` (see eq. 21 of Scargle 2012). If gamma is specified, p0 is
        ignored.
    ncp_prior : float (optional)
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\tt ncp\_prior} = -\ln({\tt
        gamma})`.  If ``ncp_prior`` is specified, ``gamma`` and ``p0`` are
        ignored.
    """

    def __init__(self, p0=0.05, gamma=None, ncp_prior=None):
        super(PELT, self).__init__(p0, gamma, ncp_prior)

    def fitness(self, N_k, T_k):
        # eq. 19 from Scargle 2012
        return -N_k * (np.log(N_k) - np.log(T_k))

    def segment(self, t, x=None, sigma=None):
        """Fit the Bayesian Blocks model given the specified algorithm function.

        Parameters
        ----------
        t : array_like
            data times (one dimensional, length N)
        x : array_like (optional)
            data values
        sigma : array_like or float (optional)
            data errors

        Returns
        -------
        edges : ndarray
            array containing the (M+1) edges defining the M optimal bins
        """

        t, x, sigma = self.validate_input(t, x, sigma)

        # create length-(N + 1) array of cell edges
        edges = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])
        block_length = t[-1] - edges

        # arrays to store the best configuration
        N = len(t)
        best = np.zeros(N + 1, dtype=float)
        last = np.zeros(N, dtype=int)

        # Compute ncp_prior if not defined
        if self.ncp_prior is None:
            ncp_prior = self.compute_ncp_prior(N)
        else:
            ncp_prior = self.ncp_prior
            self.prior = self.ncp_prior

        K = 0
        unpruned = np.array([], dtype=np.int)
        lastPruned = -1

        # ----------------------------------------------------------------
        # Start with first data cell; add one cell at each iteration
        # ----------------------------------------------------------------
        for R in range(N):
            # Consider everything unpruned until proven otherwise
            unpruned = np.concatenate([unpruned, np.array([R])])

            # T_k: width/duration of each block
            T_k = block_length[unpruned] - block_length[R + 1]

            # N_k: number of elements in each block
            N_k = np.cumsum(x[unpruned][::-1])[::-1]

            # Compute fit_vec : algorithm of putative last block (end at R)
            # evaluate algorithm function
            fit_vec = self.fitness(N_k, T_k)
            A_R = fit_vec + ncp_prior
            A_R += best[lastPruned + 1 : R + 1]

            i_min = np.argmin(A_R)
            last[R] = unpruned[i_min]
            best[R + 1] = A_R[i_min]
            peltFitness = best[lastPruned + 1 : R + 1] + fit_vec

            # Pruning step; only applies if N_k ≥ 2
            candidates = (N_k[:-1] >= 2).nonzero()[0]
            if candidates.size:
                pruned = (peltFitness + K > best[R + 1]).nonzero()[0]
                if pruned.size:
                    i_max = np.max(pruned)
                    # Wait until all indexes from 0 to i_max satisfy the pruning condition; then prune i_max
                    if np.array_equal(pruned, np.arange(i_max + 1)):
                        i_max = np.max(pruned)
                    else:
                        i_max = -1
                    if i_max >= 0:
                        t_max = unpruned[i_max]
                        unpruned = np.delete(unpruned, np.arange(i_max + 1))
                        lastPruned = t_max
        self.best_fitness = best[R + 1]

        # Find change points
        return self.get_change_points(N, edges, last)


ALGORITHM_DICT = {
    "bayesian_blocks": BayesianBlocks,
    "OP": OptimalPartitioning,
    "PELT": PELT,
}
