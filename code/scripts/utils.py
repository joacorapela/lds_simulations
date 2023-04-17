
import numpy as np
import torch


def buildQfromQe_np(Qe, sigma_ax, sigma_ay):
    Q = np.zeros_like(Qe)
    lower_slice = slice(0, int(Q.shape[0]/2))
    upper_slice = slice(int(Q.shape[0]/2), Q.shape[0])
    Q[lower_slice, lower_slice] = sigma_ax**2*Qe[lower_slice, lower_slice]
    Q[upper_slice, upper_slice] = sigma_ay**2*Qe[upper_slice, upper_slice]
    return Q


def buildQfromQe_torch(Qe, sigma_ax, sigma_ay):
    Q = torch.zeros_like(Qe)
    lower_slice = slice(0, int(Q.shape[0]/2))
    upper_slice = slice(int(Q.shape[0]/2), Q.shape[0])
    Q[lower_slice, lower_slice] = sigma_ax**2*Qe[lower_slice, lower_slice]
    Q[upper_slice, upper_slice] = sigma_ay**2*Qe[upper_slice, upper_slice]
    return Q
