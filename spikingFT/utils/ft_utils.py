#!/usr/bin/env python3
"""
Helper functions for Fourier transform algorithms
"""
# Standard libraries
import numpy as np


def twiddle_factor(k,N, type='exp'):
    """
    Return twiddle factors.
    """
    if type=='cos':
        twiddle_factor = np.cos(2*np.pi*k/N)
    elif type=='sin':
        twiddle_factor = np.sin(2*np.pi*k/N)
    elif type=='exp':
        twiddle_factor = np.exp(2j*np.pi*k/N)
    return twiddle_factor


def normalize(weights, method):
    """
    Normalize the weights for computing the 1-D FT
    """
    if method == "traditional":
        weights /= weights.size
    if method == "loihi":
        correction_coef = 127 / weights.max()
        weights_normalized = np.rint(weights * correction_coef)
    return weights_normalized*2


def dft_connection_matrix(nsamples, platform):
    """
    Calculate network weights based on Fourier transform equation

    Parameters:
        nsamples (int): Number of samples in a chirp
        platform (str [loihi|traditional]): If "loihi", values are normalized
         between the limits imposed by the chip; Re-scale weights to the range
         admitted by Loihi (8-bit even values -257 to 254). If "traditional",
         each weight is divided by the total length of the chirp, as in a
         conventional Fourier transform
    Returns:
        real_weight_norm: weights for the connections to the "real" compartments
        imag_weight_norm: weights for the connections to the "imag" compartments
    """
    c = 2 * np.pi/nsamples
    n = np.arange(nsamples).reshape(nsamples, 1)
    k = np.arange(nsamples).reshape(1, nsamples)
    trig_factors = np.dot(n, k) * c
    real_weights = np.cos(trig_factors)
    imag_weights = -np.sin(trig_factors)
    # Normalize the weights based on the used platform
    real_weights_norm = normalize(real_weights, platform)
    imag_weights_norm = normalize(imag_weights, platform)
    return (real_weights_norm, imag_weights_norm)


def fft_connection_matrix(l):
    """
    Connection matrix for a radix-4 fft
    """
    radix = 4
    n_layers = int(np.log(nsamples)/np.log(radix)) # number of layers
    n_bfs = int(radix**n_layers/radix) # number of butterflies
    n_blocks = radix**(l) # number of blocks
    n_bfs_per_block = int(n_bfs/n_blocks) # number of butterflies in one block
    distance_between_datapoints = radix**(n_layers-l-1) 
    distance_between_blocks = radix**(l) 

    n = np.tile(np.arange(0,radix**(n_layers-l-1)),radix**(l+1))
    c = np.tile(np.repeat(np.arange(0,4**(l+1),4**l),4**(n_layers-l-1)),4**l)

      # radix4 butterfly
    W = np.array([
        [1, 1,   1,  1],
        [1, -1j, -1, 1j],
        [1, -1,  1,  -1],
        [1, 1j,  -1, -1j]
    ])

      # build 4 matrices of radix4 butterfly
    W_rr = np.real(W) # real input real weights real ouput
    W_ir = -1*np.imag(W) # imag input imag weights real output
    W_ri = np.imag(W) # real input imag weights imag ouput
    W_ii = np.real(W) # imag input real weights imag output 

    # Fancy matrix structure 
    G = np.eye(4**l, 4**l)
    B = np.eye(int(nsamples/4**(l+1)), int(nsamples/4**(l+1)))

    # Kronecker magic
    M_rr = np.kron(G, np.kron(W_rr, B))
    M_ir = np.kron(G, np.kron(W_ir, B))
    M_ri = np.kron(G, np.kron(W_ri, B))
    M_ii = np.kron(G, np.kron(W_ii, B))

    # Stacking 4 matrices to 2x2 
    M = np.vstack([np.hstack([M_rr, M_ir]), np.hstack([M_ri, M_ii])])

    # twiddle factor as 2x2 matrix
    tf_real = np.hstack([np.diag(twiddle_factor(n*c, nsamples, 'cos')),
                         -np.diag(twiddle_factor(n*c,nsamples,'sin'))])
    tf_imag = np.hstack([np.diag(twiddle_factor(n*c, nsamples, 'sin')),
                         np.diag(twiddle_factor(n*c,nsamples,'cos'))])
    tf = np.vstack([tf_real, tf_imag])

    # Twiddle factors times butterfly matrix
    M = np.matmul(tf.transpose(), M)
    return M
