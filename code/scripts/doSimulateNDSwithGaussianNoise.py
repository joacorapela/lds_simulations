import sys
import os
import random
import argparse
import configparser
import pickle
import numpy as np
import torch
import lds.simulation
import lds.tracking.utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation_params_filename", type=str,
                        default="../../metadata/00000023_simulation.ini",
                        help="simulation parameters filename")
    parser.add_argument("--results_filename_pattern", type=str,
                        default="../../results/{:08d}_simulation.{:s}",
                        help="results filename pattern")
    args = parser.parse_args()

    simulation_params_filename = args.simulation_params_filename
    results_filename_pattern = args.results_filename_pattern

    metadata = configparser.ConfigParser()
    metadata.read(simulation_params_filename)
    dt = float(metadata["other"]["dt"])
    alpha = float(metadata["other"]["alpha"])
    sim_len = int(metadata["other"]["sim_len"])
    pos_x_mean0 = float(metadata["initial_state"]["pos_x_mean"])
    pos_y_mean0 = float(metadata["initial_state"]["pos_y_mean"])
    vel_x_mean0 = float(metadata["initial_state"]["vel_x_mean"])
    vel_y_mean0 = float(metadata["initial_state"]["vel_y_mean"])
    acc_x_mean0 = float(metadata["initial_state"]["acc_x_mean"])
    acc_y_mean0 = float(metadata["initial_state"]["acc_y_mean"])
    cos_theta_mean0 = float(metadata["initial_state"]["cos_theta_mean"])
    sin_theta_mean0 = float(metadata["initial_state"]["sin_theta_mean"])
    omega_mean0 = float(metadata["initial_state"]["omega_mean"])
    pos_x_std0 = float(metadata["initial_state"]["pos_x_std"])
    pos_y_std0 = float(metadata["initial_state"]["pos_y_std"])
    vel_x_std0 = float(metadata["initial_state"]["vel_x_std"])
    vel_y_std0 = float(metadata["initial_state"]["vel_y_std"])
    acc_x_std0 = float(metadata["initial_state"]["acc_x_std"])
    acc_y_std0 = float(metadata["initial_state"]["acc_y_std"])
    cos_theta_std0 = float(metadata["initial_state"]["cos_theta_std"])
    sin_theta_std0 = float(metadata["initial_state"]["sin_theta_std"])
    omega_std0 = float(metadata["initial_state"]["omega_std"])
    sigma_a = float(metadata["state_cov"]["sigma_a"])
    cos_theta_Q_std = float(metadata["state_cov"]["cos_theta_Q_std"])
    sin_theta_Q_std = float(metadata["state_cov"]["sin_theta_Q_std"])
    omega_Q_std = float(metadata["state_cov"]["omega_Q_std"])
    pos_x_R_std = float(metadata["measurements_cov"]["pos_x_R_std"])
    pos_y_R_std = float(metadata["measurements_cov"]["pos_y_R_std"])
    cos_theta_R_std = float(metadata["measurements_cov"]["cos_theta_R_std"])
    sin_theta_R_std = float(metadata["measurements_cov"]["sin_theta_R_std"])

    m0 = torch.tensor([pos_x_mean0, vel_x_mean0, acc_x_mean0,
                       pos_y_mean0, vel_y_mean0, acc_y_mean0,
                       cos_theta_mean0, sin_theta_mean0,
                       omega_mean0], dtype=torch.double)
    V0 = torch.diag(torch.FloatTensor(
        [pos_x_std0, vel_x_std0, acc_x_std0,
         pos_y_std0, vel_y_std0, acc_y_std0,
         cos_theta_std0, sin_theta_std0,
         omega_std0])**2)

    B, Bdot, Z, Zdot, Q, R = lds.tracking.utils.getNDSwithGaussianNoiseFunctionsForKinematicsAndHO_torch(
        dt=dt,
        alpha=alpha,
        sigma_a=sigma_a,
        cos_theta_Q_std=cos_theta_Q_std,
        sin_theta_Q_std=sin_theta_Q_std,
        omega_Q_std=omega_Q_std,
        pos_x_R_std=pos_x_R_std,
        pos_y_R_std=pos_y_R_std,
        cos_theta_R_std=cos_theta_R_std,
        sin_theta_R_std=sin_theta_R_std)
    T = int(sim_len / dt)
    x0, x, y = lds.simulation.simulateNDSgaussianNoise(T=T, B=B, Q=Q, m0=m0,
                                                       V0=V0, Z=Z, R=R)

    res_prefix_used = True
    while res_prefix_used:
        res_number = random.randint(0, 10**8)
        metadata_filename = results_filename_pattern.format(res_number, "ini")
        if not os.path.exists(metadata_filename):
            res_prefix_used = False

    results_filename = results_filename_pattern.format(res_number, "npz")
    np.savez(results_filename, x0=x0, x=x, y=y)
    print(f"Saved smoothing results to {results_filename}")

    metadata_filename = results_filename_pattern.format(res_number, "ini")
    metadata = configparser.ConfigParser()
    metadata.read(metadata_filename)
    metadata = configparser.ConfigParser()
    metadata["params"] = {
        "simulation_params_filename": simulation_params_filename,
    }
    with open(metadata_filename, "w") as f:
        metadata.write(f)

    print(f"Done with simulation {res_number}")
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
