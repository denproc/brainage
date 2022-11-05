import numpy as np
import scipy.ndimage


class GMM:
    """
    Gaussian Mixture Model
    """

    def __init__(self, n_components=3, max_iter=10, prior=False, mrf=False, verbose=False, tol=1e-3):
        """
        Initialization of the GMM
        :param n_components: number of classes
        :param max_iter: max number of iterations
        :param prior: include priors into EM (bool)
        :param mrf: include MRF into EM (bool)
        :param verbose:
        :param tol: tolerance for NLL convergence
        """
        self.n_components = n_components
        self.prior = prior
        self.mrf = mrf
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self.means = None
        self.variances = None
        self.weights = None

    def gauss(self, x, mean, variance):
        """
        Return the gaussian probability of given value
        :param x: Vector of intensities (Nx1)
        :param mean: Mean value of the Gaussian distribution (1xM)
        :param variance: Variance of the Gaussian distribution (1xM)
        :return: Gaussian probabilities (NxM)
        """
        g = np.exp(-(x - mean) ** 2 / (2 * variance)) / ((2 * np.pi * variance) ** 0.5)
        return g

    def update_probability(self, features, means, variances, mixtures):
        """
        Performs E step of the EM algorithm for GMM
        :param features: (Nx1)
        :param means: (1xM)
        :param variances: (1xM)
        :param mixtures: (1xM) of (NxM)
        :return: Posterior probabilities of the classes (NxM)
        """
        p = self.gauss(x=features, mean=means, variance=variances) * mixtures
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def update_parameters(self, features, p):
        """
        Perform M step of the EM algorithm for GMM
        :param features: (Nx1)
        :param p: posterior probabilities (NxM)
        :param prior: ???
        :return:
        """
        means = (p * features).sum(axis=0, keepdims=True) / p.sum(axis=0, keepdims=True)
        variances = (p * (features - means) ** 2).sum(axis=0, keepdims=True) / p.sum(axis=0, keepdims=True)

        if self.prior and not self.mrf:
            return means, variances, self.weights

        new_weights = p.mean(axis=0, keepdims=True)
        return means, variances, new_weights

    def nll(self, p):
        """
        Calculate normilised negative log-likelihood
        :param p: likelihood probability
        :return: normilised negative log-likelihood
        """
        return - np.mean(np.log(p))

    def init(self):
        """
        Initisalisation of the model parameters
        """
        assert self.n_components == 3
        self.means = np.array([[20., 54., 100.]])
        self.variances = np.array([[500., 500., 500.]])
        self.weights = np.array([[1 / self.n_components] * self.n_components])

    def fit_predict(self, image, priors=None):
        """
        Fits model to the given image and returns the segmentation map, probabilities and NLL scores
        :param image: 0, 255
        :param priors:
        :return: list with NLL scores, probability maps per class, segmentation map
        """
        self.init()

        active_ind = np.where(image > 0)
        features = image[active_ind].reshape(-1, 1)

        if self.prior:
            priors_features = priors[active_ind]
            self.weights = priors_features
            p = priors_features
        else:
            active_ind = np.where(image > 0)
            features = image[active_ind].reshape(-1, 1)
            priors_features = None

            p = self.update_probability(features, self.means, self.variances, self.weights)

        scores = []
        for i in range(self.max_iter):
            self.means, self.variances, self.weights = self.update_parameters(features, p)

            if self.mrf:
                # MRF
                posterior_map = np.zeros((*image.shape, self.n_components))
                for j in range(self.n_components):
                    posterior_map[..., j][active_ind] = p[:, j]
                potentials = self.get_potentials(posterior_map)
                potentials_features = potentials[active_ind]
                self.weights = np.exp(-potentials_features)/np.sum(np.exp(-potentials_features), axis=-1, keepdims=True)

                if self.prior:
                    self.weights *= priors_features

            p = self.update_probability(features, self.means, self.variances, self.weights)
            likelihood_per_class = self.gauss(features, self.means, self.variances)
            loss = self.nll((likelihood_per_class * self.weights).sum(axis=-1, keepdims=True))
            scores.append(loss)
            if i > 0 and abs(scores[-1] - scores[-2]) < self.tol:
                break

        prob_map = np.zeros((*image.shape, self.n_components))

        for i in range(self.n_components):
            prob_map[..., i][active_ind] = p[:, i]

        segmentation = np.zeros_like(image)
        segmentation[active_ind] = np.argmax(prob_map, axis=-1)[active_ind] + 1

        return scores, prob_map, segmentation

    def get_potentials(self, posterior_maps):
        """
        Calculate potentials for the MRF
        :param posterior_maps: IMG x 3
        :return:
        """
        mrf_filter = np.zeros((3, 3, 3))
        mrf_filter[0, 1, 1] = self.mrf
        mrf_filter[1, 1, 0] = self.mrf
        mrf_filter[1, 0, 1] = self.mrf
        mrf_filter[1, 2, 1] = self.mrf
        mrf_filter[1, 1, 2] = self.mrf
        mrf_filter[2, 1, 1] = self.mrf

        potentials = scipy.ndimage.convolve(1 - posterior_maps, mrf_filter[..., None], mode='constant')

        return potentials
