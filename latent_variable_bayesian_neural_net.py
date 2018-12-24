import matplotlib.pyplot as plt
import autograd.scipy.stats.norm as norm

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam

from bayesian_neural_net import bnn_predict, shapes_and_num

rs = npr.RandomState(0)


def gaussian_entropy(log_std):
    return 0.5 * log_std.shape[0] * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)  # [ns]

def sample_weights(params, N_samples):
    mean, log_std = params
    return rs.randn(N_samples, mean.shape[0]) * np.exp(log_std) + mean  # [ns, nw]

def sample_latents(params):
    mean, log_std = params # [nd, 1]
    return rs.randn(*mean.shape) * np.exp(log_std) + mean  # [nd, 1]

def sample_bnn(params, x, N_samples, layer_sizes, act, noise=0.0):
    qw_mean, qw_log_std, qz_mean, qz_log_std = params
    weights = sample_weights((qw_mean, qw_log_std), N_samples)
    latents = sample_latents((qz_mean, qz_log_std)) # []
    inputs = np.concatenate([x, latents], -1)
    return bnn_predict(weights, inputs, layer_sizes, act)[:, :, 0]   # [ns, nd]


def vlb_objective(params, x, y, layer_sizes, n_samples, model_sd=0.1, act=np.tanh):
    """ estimates lower bound = -H[q(w))]- E_q(w)[log p(D,w)] """
    qw_mean, qw_log_std, qz_mean, qz_log_std = params

    weights = sample_weights((qw_mean, qw_log_std), n_samples)
    latents = sample_latents((qz_mean, qz_log_std))  # []
    entropy = gaussian_entropy(qw_log_std)+gaussian_entropy(qz_log_std)
    print(x.shape, latents.shape)
    f_bnn= bnn_predict(weights, np.concatenate([x, latents], 1), layer_sizes, act)[:, :, 0]   # [ns, nd]


    #f_bnn = sample_bnn(params, x, n_samples,layer_sizes, act)
    log_likelihood = diag_gaussian_log_density(y.T, f_bnn, .1)
    qw_log_prior = diag_gaussian_log_density(weights, 0, 1)
    qz_log_prior = diag_gaussian_log_density(latents, 0, 1)

    return -entropy - np.mean(log_likelihood+qw_log_prior) -np.mean(qz_log_prior)


def init_var_params(layer_sizes, inputs, scale=-5, scale_mean=1):
    _, num_weights = shapes_and_num(layer_sizes)
    qw_init = [rs.randn(num_weights)*scale_mean, np.ones(num_weights)*scale]  # mean, log_std
    qz_init = [rs.randn(*inputs.shape)*scale_mean, np.ones(shape=inputs.shape)*scale]  # mean, log_std
    return qw_init + qz_init


def build_toy_dataset(n_data=80, noise_std=0.1):
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(0, 3, num=n_data/2),
                              np.linspace(6, 8, num=n_data/2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 2.0
    inputs  = inputs[:, np.newaxis]
    targets = targets[:, np.newaxis] / 2.0
    return inputs, targets


def train_bnn(inputs, targets, arch = [2, 20, 20, 1], lr=0.01, iters=500, n_samples=10, act=np.tanh):

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show(block=False)

    def objective(params, t):
        return vlb_objective(params, inputs, targets, arch, n_samples, act)

    def callback(params, t, g):
        # Sample functions from posterior f ~ p(f|phi) or p(f|varphi)
        N_samples, nd = 5, 80
        plot_inputs = np.linspace(-8, 8, num=80)
        f_bnn = sample_bnn(params, plot_inputs[:,None], N_samples, arch, act)

        plt.cla()
        ax.plot(inputs.ravel(), targets.ravel(), 'k.')
        ax.plot(plot_inputs, f_bnn.T, color='r')
        ax.set_ylim([-5, 5])
        plt.draw()
        plt.pause(1.0 / 60.0)

        print("ITER {} | OBJ {}".format(t, -objective(params, t)))

    var_params = adam(grad(objective), init_var_params(arch, inputs),
                      step_size=lr, num_iters=iters, callback=callback)

    return var_params


if __name__ == '__main__':
    arch = [1, 20, 20, 1]
    inputs, targets = build_toy_dataset()
    train_bnn(inputs, targets)


