
import sys
import os
import random
import pickle
import math
import argparse
import configparser
import numpy as np
import torch

import lds.tracking.utils
import lds.inference


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_res_num", type=int, help="simulation result number",
                        default=67505436)
    parser.add_argument("--start_offset_secs", type=int, default=0,
                        help="start offset in seconds")
    parser.add_argument("--duration_secs", type=int, default=-1,
                        help="duration in seconds")
    parser.add_argument("--filtering_params_filename", type=str,
                        default="../../metadata/00000009_filtering.ini",
                        help="filtering parameters filename")
    parser.add_argument("--initial_state_section", type=str,
                        default="initial_state",
                        help=("section of ini file containing the initial state "
                              "params"))
    parser.add_argument("--state_cov_section", type=str,
                        default="state_cov",
                        help=("section of ini file containing the state cov "
                              "params"))
    parser.add_argument("--measurements_cov_section", type=str,
                        default="measurements_cov",
                        help=("section of ini file containing the measurement cov "
                              "params"))
    parser.add_argument("--sim_res_filename_pattern", type=str,
                        default="../../results/{:08d}_simulation.{:s}",
                        help="simulation results filename pattern")
    parser.add_argument("--results_filename_pattern", type=str,
                        default="../../results/{:08d}_filtered.{:s}")
    args = parser.parse_args()

    sim_res_num = args.sim_res_num
    start_offset_secs = args.start_offset_secs
    duration_secs = args.duration_secs
    filtering_params_filename = args.filtering_params_filename 
    initial_state_section = args.initial_state_section
    state_cov_section = args.state_cov_section
    measurements_cov_section = args.measurements_cov_section
    sim_res_filename_pattern = args.sim_res_filename_pattern
    results_filename_pattern = args.results_filename_pattern

    sim_res_filename = sim_res_filename_pattern.format(sim_res_number, "npz")
    sim_res = np.load(sim_res_filename)
    data = sim_res["y"].T

    metadata_filename = sim_res_filename_pattern.format(sim_res_number, "ini")
    metadata = configparser.ConfigParser()
    metadata.read(metadata_filename)
    dt = float(metadata["params"]["dt"])
    sigma_a = float(metadata["params"]["sigma_a"])

    start_sample = int(start_offset_secs / dt)

    if duration_secs < 0:
        number_samples = data.shape[0] - start_sample
    else:
        number_samples = int(duration_secs / dt)

    times = np.arange(start_sample,
                      start_sample+number_samples) * dt
    data = data[start_sample:(start_sample+number_samples), :]

    # make sure that the first data point is not NaN
    first_not_nan_index = np.where(~np.isnan(data).any(axis=1))[0][0]
    data = data[first_not_nan_index:,]
    #

    filtering_params = configparser.ConfigParser()
    filtering_params.read(filtering_params_filename)

    pos_x_m0 = float(filtering_params[initial_state_section]["pos_x_mean"])
    vel_x_m0 = float(filtering_params[initial_state_section]["vel_x_mean"])
    acc_x_m0 = float(filtering_params[initial_state_section]["acc_x_mean"])
    pos_y_m0 = float(filtering_params[initial_state_section]["pos_y_mean"])
    vel_y_m0 = float(filtering_params[initial_state_section]["vel_y_mean"])
    acc_y_m0 = float(filtering_params[initial_state_section]["acc_y_mean"])
    cos_theta_m0 = float(filtering_params[initial_state_section]["cos_theta_mean"])
    sin_theta_m0 = float(filtering_params[initial_state_section]["sin_theta_mean"])
    omega_m0 = float(filtering_params[initial_state_section]["omega_mean"])

    pos_x_V0_sigma = float(filtering_params[initial_state_section]["pos_x_sigma"])
    vel_x_V0_sigma = float(filtering_params[initial_state_section]["vel_x_sigma"])
    acc_x_V0_sigma = float(filtering_params[initial_state_section]["acc_x_sigma"])
    pos_y_V0_sigma = float(filtering_params[initial_state_section]["pos_y_sigma"])
    vel_y_V0_sigma = float(filtering_params[initial_state_section]["vel_y_sigma"])
    acc_y_V0_sigma = float(filtering_params[initial_state_section]["acc_y_sigma"])
    cos_theta_V0_sigma = float(filtering_params[initial_state_section]["cos_theta_sigma"])
    sin_theta_V0_sigma = float(filtering_params[initial_state_section]["sin_theta_sigma"])
    omega_V0_sigma = float(filtering_params[initial_state_section]["omega_sigma"])

    sigma_a = float(filtering_params[state_cov_section]["sigma_a"])
    cos_theta_Q_sigma = float(filtering_params[state_cov_section]["cos_theta_Q_sigma"])
    sin_theta_Q_sigma = float(filtering_params[state_cov_section]["sin_theta_Q_sigma"])
    omega_Q_sigma = float(filtering_params[state_cov_section]["omega_Q_sigma"])

    pos_x_R_sigma = float(filtering_params[measurements_cov_section]["pos_x_R_sigma"])
    pos_y_R_sigma = float(filtering_params[measurements_cov_section]["pos_y_R_sigma"])
    cos_theta_R_sigma = float(filtering_params[measurements_cov_section]["cos_theta_R_sigma"])
    sin_theta_R_sigma = float(filtering_params[measurements_cov_section]["sin_theta_R_sigma"])

    alpha = float(filtering_params["other"]["alpha"])

    if math.isnan(pos_x_m0):
        pos_x_m0 = data[0, 0]
    if math.isnan(pos_y_m0):
        pos_y_m0 = data[0, 1]
    if math.isnan(cos_theta_m0):
        cos_theta_m0 = data[0, 2]
    if math.isnan(sin_theta_m0):
        sin_theta_m0 = data[0, 3]

    m0 = np.array([pos_x_m0, vel_x_m0, acc_x_m0, pos_y_m0, vel_y_m0, acc_y_m0,
                   cos_theta_m0, sin_theta_m0, omega_m0], dtype=np.double)
    V0 = np.diag([pos_x_V0_sigma, vel_x_V0_sigma, acc_x_V0_sigma,
                  pos_y_V0_sigma, vel_y_V0_sigma, acc_y_V0_sigma,
                  cos_theta_V0_sigma, sin_theta_V0_sigma, omega_V0_sigma])

    B, Bdot, Z, Zdot, Q, R = lds.tracking.utils.getNDSwithGaussianNoiseFunctionsForKinematicsAndHO_torch(
        dt=dt, alpha=alpha, sigma_a=sigma_a,
        cos_theta_Q_sigma=cos_theta_Q_sigma,
        sin_theta_Q_sigma=sin_theta_Q_sigma,
        omega_Q_sigma=omega_Q_sigma,
        pos_x_R_sigma=pos_x_R_sigma,
        pos_y_R_sigma=pos_y_R_sigma,
        cos_theta_R_sigma=cos_theta_R_sigma,
        sin_theta_R_sigma=sin_theta_R_sigma)
    data = torch.from_numpy(data)
    m0 = torch.from_numpy(m0)
    V0 = torch.from_numpy(V0)
    filter_res = lds.inference.filterEKF_withMissingValues_torch(
        y=data.T, B=B, Bdot=Bdot, Q=Q, m0=m0, V0=V0, Z=Z, Zdot=Zdot, R=R)

    results = {"times": times,
               "measurements": data.T,
               "filter_res": filter_res}

    # save results
    res_prefix_used = True
    while res_prefix_used:
        res_number = random.randint(0, 10**8)
        metadata_filename = results_filename_pattern.format(res_number, "ini")
        if not os.path.exists(metadata_filename):
            res_prefix_used = False
    results_filename = results_filename_pattern.format(res_number, "pickle")

    with open(results_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved EK filter results to {results_filename}")

    metadata = configparser.ConfigParser()
    metadata["params"] = {
        "sim_res_num": sim_res_num,
        "times": times,
        "start_sample": start_sample,
        "number_samples": number_samples,
        "filtering_params_filename": filtering_params_filename,
    }
    with open(metadata_filename, "w") as f:
        metadata.write(f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
