
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
    parser.add_argument("sim_res_num", type=int, help="simulation result number")
    parser.add_argument("--start_offset_secs", type=int, default=0,
                        help="start offset in seconds")
    parser.add_argument("--duration_secs", type=int, default=-1,
                        help="duration in seconds")
    parser.add_argument("--filtering_params_filename", type=str,
                        default="../../metadata/00000023_simulation.ini",
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

    sim_res_filename = sim_res_filename_pattern.format(sim_res_num, "npz")
    sim_res = np.load(sim_res_filename)
    data = sim_res["y"].T
    data = data[:, :2]

    metadata_filename = sim_res_filename_pattern.format(sim_res_num, "ini")
    metadata = configparser.ConfigParser()
    metadata.read(metadata_filename)
    simulation_params_filename = metadata["params"]["simulation_params_filename"]
    simulation_params = configparser.ConfigParser()
    simulation_params.read(simulation_params_filename)
    dt = float(simulation_params["other"]["dt"])

    start_sample = int(start_offset_secs / dt)

    if duration_secs < 0:
        number_samples = data.shape[0] - start_sample
    else:
        number_samples = int(duration_secs / dt)

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

    pos_x_V0_std = float(filtering_params[initial_state_section]["pos_x_std"])
    vel_x_V0_std = float(filtering_params[initial_state_section]["vel_x_std"])
    acc_x_V0_std = float(filtering_params[initial_state_section]["acc_x_std"])
    pos_y_V0_std = float(filtering_params[initial_state_section]["pos_y_std"])
    vel_y_V0_std = float(filtering_params[initial_state_section]["vel_y_std"])
    acc_y_V0_std = float(filtering_params[initial_state_section]["acc_y_std"])

    sigma_a = float(filtering_params[state_cov_section]["sigma_a"])
    pos_x_R_std = float(filtering_params[measurements_cov_section]["pos_x_R_std"])
    pos_y_R_std = float(filtering_params[measurements_cov_section]["pos_y_R_std"])

    if math.isnan(pos_x_m0):
        pos_x_m0 = data[0, 0]
    if math.isnan(pos_y_m0):
        pos_y_m0 = data[0, 1]

    m0 = np.array([pos_x_m0, vel_x_m0, acc_x_m0, pos_y_m0, vel_y_m0, acc_y_m0],
                   dtype=np.double)
    V0 = np.diag([pos_x_V0_std, vel_x_V0_std, acc_x_V0_std,
                  pos_y_V0_std, vel_y_V0_std, acc_y_V0_std])

    B, Q, _, Z, R = lds.tracking.utils.getLDSmatricesForKinematics_torch(
        dt=dt, sigma_a=sigma_a,
        pos_x_R_std=pos_x_R_std,
        pos_y_R_std=pos_y_R_std)
    data = torch.from_numpy(data)
    m0 = torch.from_numpy(m0)
    V0 = torch.from_numpy(V0)
    filter_res = lds.inference.filterLDS_SS_withMissingValues_torch(
        y=data.T, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)

    # save results
    res_prefix_used = True
    while res_prefix_used:
        res_number = random.randint(0, 10**8)
        metadata_filename = results_filename_pattern.format(res_number, "ini")
        if not os.path.exists(metadata_filename):
            res_prefix_used = False
    results_filename = results_filename_pattern.format(res_number, "pickle")

    with open(results_filename, "wb") as f:
        pickle.dump(filter_res, f)
    print(f"Saved Kalman filter results to {results_filename}")

    metadata = configparser.ConfigParser()
    metadata["params"] = {
        "sim_res_num": sim_res_num,
        "sim_res_filename_pattern": sim_res_filename_pattern,
        "start_sample": start_sample,
        "number_samples": number_samples,
        "filtering_params_filename": filtering_params_filename,
    }
    with open(metadata_filename, "w") as f:
        metadata.write(f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
