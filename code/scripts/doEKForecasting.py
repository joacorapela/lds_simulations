import sys
import os
import pickle
import math
import random
import numpy as np
import argparse
import configparser
import lds.inference
import lds.tracking.utils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("filtering_results_number", type=int,
                        help="number corresponding to filtered results filename")
    parser.add_argument("--h", type=int,
                        default=2,
                        help="forecast horizon in samples")
    parser.add_argument("--filtering_results_filenames_pattern", type=str,
                        default="../../results/{:08d}_filtered.{:s}",
                        help="filtering_results filename pattern")
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
    parser.add_argument("--results_filename_pattern",
                        help="results filename pattern",
                        default="../../results/{:08d}_forecasting.{:s}")

    args = parser.parse_args()

    filtering_results_number = args.filtering_results_number
    h = args.h
    initial_state_section = args.initial_state_section
    state_cov_section = args.state_cov_section
    measurements_cov_section = args.measurements_cov_section
    filtering_results_filenames_pattern = \
        args.filtering_results_filenames_pattern
    results_filename_pattern = args.results_filename_pattern

    filtering_results_filename = \
        filtering_results_filenames_pattern.format(filtering_results_number, "pickle")
    with open(filtering_results_filename, "rb") as f:
        filtering_results = pickle.load(f)

    xnn = filtering_results["filter_res"]["xnn"]
    Pnn = filtering_results["filter_res"]["Pnn"]
    # Pnn = filtering_results["filter_res"]["Vnn"]

    filtered_metadata_filename = \
        filtering_results_filenames_pattern.format(filtering_results_number, "ini")
    filtered_metadata = configparser.ConfigParser()
    filtered_metadata.read(filtered_metadata_filename)
    sim_res_number = int(filtered_metadata["params"]["sim_res_num"])
    filtering_params_filename = filtered_metadata["params"]["filtering_params_filename"]
    # sim_res_filename_pattern = filtered_metadata["params"]["sim_res_filename_pattern"]
    sim_res_filename_pattern = "../../results/{:08d}_simulation.{:s}"
    sim_res_filename = sim_res_filename_pattern.format(sim_res_number, "npz")
    sim_res = np.load(sim_res_filename)
    sim_metadata_filename = sim_res_filename_pattern.format(sim_res_number, "ini")
    sim_metadata = configparser.ConfigParser()
    sim_metadata.read(sim_metadata_filename)
    simulation_params_filename = sim_metadata["params"]["simulation_params_filename"]
    simulation_params = configparser.ConfigParser()
    simulation_params.read(simulation_params_filename)
    dt = float(simulation_params["other"]["dt"])

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

    pos_x_V0_std = float(filtering_params[initial_state_section]["pos_x_std"])
    vel_x_V0_std = float(filtering_params[initial_state_section]["vel_x_std"])
    acc_x_V0_std = float(filtering_params[initial_state_section]["acc_x_std"])
    pos_y_V0_std = float(filtering_params[initial_state_section]["pos_y_std"])
    vel_y_V0_std = float(filtering_params[initial_state_section]["vel_y_std"])
    acc_y_V0_std = float(filtering_params[initial_state_section]["acc_y_std"])
    cos_theta_V0_std = float(filtering_params[initial_state_section]["cos_theta_std"])
    sin_theta_V0_std = float(filtering_params[initial_state_section]["sin_theta_std"])
    omega_V0_std = float(filtering_params[initial_state_section]["omega_std"])

    sigma_a = float(filtering_params[state_cov_section]["sigma_a"])
    cos_theta_Q_std = float(filtering_params[state_cov_section]["cos_theta_std"])
    sin_theta_Q_std = float(filtering_params[state_cov_section]["sin_theta_std"])
    omega_Q_std = float(filtering_params[state_cov_section]["omega_std"])

    pos_x_R_std = float(filtering_params[measurements_cov_section]["pos_x_std"])
    pos_y_R_std = float(filtering_params[measurements_cov_section]["pos_y_std"])
    cos_theta_R_std = float(filtering_params[measurements_cov_section]["cos_theta_std"])
    sin_theta_R_std = float(filtering_params[measurements_cov_section]["sin_theta_std"])

    alpha = float(filtering_params["other"]["alpha"])

    data = sim_res["y"].T
    data = data

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
    V0 = np.diag([pos_x_V0_std, vel_x_V0_std, acc_x_V0_std,
                  pos_y_V0_std, vel_y_V0_std, acc_y_V0_std,
                  cos_theta_V0_std, sin_theta_V0_std, omega_V0_std])

    B, Bdot, Z, Zdot, Q, R = lds.tracking.utils.getNDSwithGaussianNoiseFunctionsForKinematicsAndHO_torch(
        dt=dt, alpha=alpha, sigma_a=sigma_a,
        cos_theta_Q_std=cos_theta_Q_std,
        sin_theta_Q_std=sin_theta_Q_std,
        omega_Q_std=omega_Q_std,
        pos_x_R_std=pos_x_R_std,
        pos_y_R_std=pos_y_R_std,
        cos_theta_R_std=cos_theta_R_std,
        sin_theta_R_std=sin_theta_R_std)

    x_pred, P_pred = lds.inference.ekf_forecast_batch(xnn=xnn, Pnn=Pnn, B=B, Bdot=Bdot, Q=Q, m0=m0, V0=V0, h=h)
    log_like = lds.inference.log_like_observations_given_forecasts_ekf(
        h=h, y=data.T, x_pred=x_pred, P_pred=P_pred, Z=Z, Zdot=Zdot, R=R)
    print(f"log likelihood: {log_like}")

    forecast_res = dict(x=x_pred, P=P_pred, log_like=log_like)

    # save results
    res_prefix_used = True
    while res_prefix_used:
        res_number = random.randint(0, 10**8)
        metadata_filename = results_filename_pattern.format(res_number, "ini")
        if not os.path.exists(metadata_filename):
            res_prefix_used = False
    results_filename = results_filename_pattern.format(res_number, "pickle")

    with open(results_filename, "wb") as f:
        pickle.dump(forecast_res, f)
    print(f"Saved forecasting results to {results_filename}")

    metadata = configparser.ConfigParser()
    metadata["params"] = {
        "horizon": h,
        "filtering_results_number": filtering_results_number,
        "filtering_results_filenames_pattern": filtering_results_filenames_pattern,
    }
    with open(metadata_filename, "w") as f:
        metadata.write(f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
