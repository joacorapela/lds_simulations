import sys
import os.path
import argparse
import configparser
import math
import random
import pickle
import numpy as np
import torch

import lds.learning


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--simRes_number", help="simulation result number",
                        type=int, default=34315983)
                        # type=int, default=91309191)
                        # type=int, default=60736713)
    parser.add_argument("--estMeta_number", help="estimation metadata number",
                        type=int, default=28)
    parser.add_argument("--estimate_sigma_a", help="estimate sigma_a",
                        type=bool, default=False)
    parser.add_argument("--estimate_cos_theta_Q_std",
                        help="estimate cos_theta_Q_std",
                        type=bool, default=True)
    parser.add_argument("--estimate_sin_theta_Q_std",
                        help="estimate sin_theta_Q_std",
                        type=bool, default=True)
    parser.add_argument("--estimate_omega_Q_std",
                        help="estimate omega_Q_std",
                        type=bool, default=True)
    parser.add_argument("--estimate_pos_x_R_std",
                        help="estimate pos_x_R_std",
                        type=bool, default=False)
    parser.add_argument("--estimate_pos_y_R_std",
                        help="estimate pos_y_R_std",
                        type=bool, default=False)
    parser.add_argument("--estimate_cos_theta_R_std",
                        help="estimate cos_theta_R_std",
                        type=bool, default=True)
    parser.add_argument("--estimate_sin_theta_R_std",
                        help="estimate sin_theta_R_std",
                        type=bool, default=True)
    parser.add_argument("--estimate_alpha",
                        help="estimate alpha",
                        type=bool, default=True)
    parser.add_argument("--estimate_m0_kinematics",
                        help="estimate m0_kinematics",
                        type=bool, default=False)
    parser.add_argument("--estimate_m0_HO",
                        help="estimate m0_HO",
                        type=bool, default=True)
    parser.add_argument("--estimate_sqrt_diag_V0_kinematics",
                        help="estimate sqrt_diag_V0_kinematics",
                        type=bool, default=False)
    parser.add_argument("--estimate_sqrt_diag_V0_HO",
                        help="estimate sqrt_diag_V0_HO",
                        type=bool, default=True)
    parser.add_argument("--simRes_filename_pattern",
                        help="simulation result filename pattern",
                        default="../../results/{:08d}_simulation.npz")
    parser.add_argument("--estInit_metadata_filename_pattern", type=str,
                        default="../../metadata/{:08d}_estimation.ini",
                        help="estimation initialization metadata filename pattern")
    parser.add_argument("--estRes_metadata_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.ini",
                        help="estimation results metadata filename pattern")
    parser.add_argument("--estRes_data_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.pickle",
                        help="estimation results data filename pattern")
    args = parser.parse_args()

    simRes_number = args.simRes_number
    estMeta_number = args.estMeta_number
    estimate_sigma_a = args.estimate_sigma_a
    estimate_cos_theta_Q_std = args.estimate_cos_theta_Q_std
    estimate_sin_theta_Q_std = args.estimate_sin_theta_Q_std
    estimate_omega_Q_std = args.estimate_omega_Q_std
    estimate_pos_x_R_std = args.estimate_pos_x_R_std
    estimate_pos_y_R_std = args.estimate_pos_y_R_std
    estimate_cos_theta_R_std = args.estimate_cos_theta_R_std
    estimate_sin_theta_R_std = args.estimate_sin_theta_R_std
    estimate_alpha = args.estimate_alpha
    estimate_m0_kinematics = args.estimate_m0_kinematics
    estimate_m0_HO = args.estimate_m0_HO
    estimate_sqrt_diag_V0_kinematics = args.estimate_sqrt_diag_V0_kinematics
    estimate_sqrt_diag_V0_HO = args.estimate_sqrt_diag_V0_HO
    simRes_filename_pattern = args.simRes_filename_pattern
    estInit_metadata_filename_pattern = args.estInit_metadata_filename_pattern
    estRes_metadata_filename_pattern = args.estRes_metadata_filename_pattern
    estRes_data_filename_pattern = args.estRes_data_filename_pattern

    simRes_filename = simRes_filename_pattern.format(simRes_number)
    simRes = np.load(simRes_filename)
    y = simRes["y"]

    estInit_metadata_filename = \
        estInit_metadata_filename_pattern.format(estMeta_number)

    estMeta = configparser.ConfigParser()
    estMeta.read(estInit_metadata_filename)
    pos_x_m0 = float(estMeta["initial_state"]["pos_x_mean"])
    pos_y_m0 = float(estMeta["initial_state"]["pos_y_mean"])
    vel_x_m0 = float(estMeta["initial_state"]["vel_x_mean"])
    vel_y_m0 = float(estMeta["initial_state"]["vel_y_mean"])
    acc_x_m0 = float(estMeta["initial_state"]["acc_x_mean"])
    acc_y_m0 = float(estMeta["initial_state"]["acc_y_mean"])
    cos_theta_m0 = float(estMeta["initial_state"]["cos_theta_mean"])
    sin_theta_m0 = float(estMeta["initial_state"]["sin_theta_mean"])
    omega_m0 = float(estMeta["initial_state"]["omega_mean"])
    pos_x_V0_std = float(estMeta["initial_state"]["pos_x_std"])
    pos_y_V0_std = float(estMeta["initial_state"]["pos_y_std"])
    vel_x_V0_std = float(estMeta["initial_state"]["vel_x_std"])
    vel_y_V0_std = float(estMeta["initial_state"]["vel_y_std"])
    acc_x_V0_std = float(estMeta["initial_state"]["acc_x_std"])
    acc_y_V0_std = float(estMeta["initial_state"]["acc_y_std"])
    cos_theta_V0_std = float(estMeta["initial_state"]["cos_theta_std"])
    sin_theta_V0_std = float(estMeta["initial_state"]["sin_theta_std"])
    omega_V0_std = float(estMeta["initial_state"]["omega_std"])

    sigma_a = torch.DoubleTensor([float(estMeta["state_cov"]["sigma_a"])])
    cos_theta_Q_std = \
        torch.DoubleTensor([float(estMeta["state_cov"]["cos_theta_Q_std"])])
    sin_theta_Q_std = \
        torch.DoubleTensor([float(estMeta["state_cov"]["sin_theta_Q_std"])])
    omega_Q_std = torch.DoubleTensor([float(estMeta["state_cov"]["omega_Q_std"])])

    pos_x_R_std = \
        torch.DoubleTensor([float(estMeta["measurements_cov"]["pos_x_R_std"])])
    pos_y_R_std = \
        torch.DoubleTensor([float(estMeta["measurements_cov"]["pos_y_R_std"])])
    cos_theta_R_std = \
        torch.DoubleTensor([float(estMeta["measurements_cov"]["cos_theta_R_std"])])
    sin_theta_R_std = \
        torch.DoubleTensor([float(estMeta["measurements_cov"]["sin_theta_R_std"])])

    alpha = torch.DoubleTensor([float(estMeta["other"]["alpha"])])
    dt = torch.DoubleTensor([float(estMeta["other"]["dt"])])

    max_iter = int(estMeta["optim_params"]["max_iter"])
    lr = float(estMeta["optim_params"]["learning_rate"])
    n_epochs = int(estMeta["optim_params"]["n_epochs"])
    line_search_fn = estMeta["optim_params"]["line_search_fn"]
    tolerance_grad = float(estMeta["optim_params"]["tolerance_grad"])
    tolerance_change = float(estMeta["optim_params"]["tolerance_change"])

    if math.isnan(pos_x_m0):
        pos_x_m0 = y[0, 0]
    if math.isnan(pos_y_m0):
        pos_y_m0 = y[1, 0]

    m0_kinematics_0 = torch.DoubleTensor([pos_x_m0, vel_x_m0, acc_x_m0,
                                          pos_y_m0, vel_y_m0, acc_y_m0])
    m0_HO_0 = torch.DoubleTensor([cos_theta_m0, sin_theta_m0, omega_m0])
    sqrt_diag_V0_kinematics_0 = torch.DoubleTensor([pos_x_V0_std,
                                                    vel_x_V0_std,
                                                    acc_x_V0_std,
                                                    pos_y_V0_std,
                                                    vel_y_V0_std,
                                                    acc_y_V0_std])
    sqrt_diag_V0_HO_0 = torch.DoubleTensor([cos_theta_V0_std,
                                            sin_theta_V0_std,
                                            omega_V0_std])

    y = torch.from_numpy(y.astype(np.double))

    vars_to_estimate = {}
    if estimate_sigma_a:
        vars_to_estimate["sigma_a"] = True
    else:
        vars_to_estimate["sigma_a"] = False

    if estimate_cos_theta_Q_std:
        vars_to_estimate["cos_theta_Q_std"] = True
    else:
        vars_to_estimate["cos_theta_Q_std"] = False

    if estimate_sin_theta_Q_std:
        vars_to_estimate["sin_theta_Q_std"] = True
    else:
        vars_to_estimate["sin_theta_Q_std"] = False

    if estimate_omega_Q_std:
        vars_to_estimate["omega_Q_std"] = True
    else:
        vars_to_estimate["omega_Q_std"] = False

    if estimate_pos_x_R_std:
        vars_to_estimate["pos_x_R_std"] = True
    else:
        vars_to_estimate["pos_x_R_std"] = False

    if estimate_pos_y_R_std:
        vars_to_estimate["pos_y_R_std"] = True
    else:
        vars_to_estimate["pos_y_R_std"] = False

    if estimate_cos_theta_R_std:
        vars_to_estimate["cos_theta_R_std"] = True
    else:
        vars_to_estimate["cos_theta_R_std"] = False

    if estimate_sin_theta_R_std:
        vars_to_estimate["sin_theta_R_std"] = True
    else:
        vars_to_estimate["sin_theta_R_std"] = False

    if estimate_alpha:
        vars_to_estimate["alpha"] = True
    else:
        vars_to_estimate["alpha"] = False

    if estimate_m0_kinematics:
        vars_to_estimate["m0_kinematics"] = True
    else:
        vars_to_estimate["m0_kinematics"] = False

    if estimate_m0_HO:
        vars_to_estimate["m0_HO"] = True
    else:
        vars_to_estimate["m0_HO"] = False

    if estimate_sqrt_diag_V0_kinematics:
        vars_to_estimate["sqrt_diag_V0_kinematics"] = True
    else:
        vars_to_estimate["sqrt_diag_V0_kinematics"] = False

    if estimate_sqrt_diag_V0_HO:
        vars_to_estimate["sqrt_diag_V0_HO"] = True
    else:
        vars_to_estimate["sqrt_diag_V0_HO"] = False

    optim_res = lds.learning.torch_lbfgs_optimize_kinematicsHO_logLikeEKF_diagV0(
        dt=dt, y=y, sigma_a0=sigma_a,
        cos_theta_Q_std0=cos_theta_Q_std,
        sin_theta_Q_std0=sin_theta_Q_std,
        omega_Q_std0=omega_Q_std,
        pos_x_R_std0=pos_x_R_std,
        pos_y_R_std0=pos_y_R_std,
        cos_theta_R_std0=cos_theta_R_std,
        sin_theta_R_std0=sin_theta_R_std,
        alpha0=alpha,
        m0_kinematics_0=m0_kinematics_0,
        m0_HO_0=m0_HO_0,
        sqrt_diag_V0_kinematics_0=sqrt_diag_V0_kinematics_0,
        sqrt_diag_V0_HO_0=sqrt_diag_V0_HO_0,
        max_iter=max_iter, lr=lr, tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change, line_search_fn=line_search_fn,
        n_epochs=n_epochs, vars_to_estimate=vars_to_estimate)

    # save results
    est_prefix_used = True
    while est_prefix_used:
        estRes_number = random.randint(0, 10**8)
        estRes_metadata_filename = \
            estRes_metadata_filename_pattern.format(estRes_number)
        if not os.path.exists(estRes_metadata_filename):
            est_prefix_used = False
    estRes_data_filename = estRes_data_filename_pattern.format(estRes_number)

    estimRes_metadata = configparser.ConfigParser()
    estimRes_metadata["simulation_params"] = {"simResNumber": simRes_number}
    estimRes_metadata["estimation_params"] = \
        {"estInitNumber": estMeta_number} | vars_to_estimate
    with open(estRes_metadata_filename, "w") as f:
        estimRes_metadata.write(f)

    with open(estRes_data_filename, "wb") as f:
        pickle.dump(optim_res, f)
    print("Saved results to {:s}".format(estRes_data_filename))
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
