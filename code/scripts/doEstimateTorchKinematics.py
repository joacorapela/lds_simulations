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
                        type=int, default=26)
    parser.add_argument("--skip_sigma_a",
                        help="skip the estimation of sigma_a",
                        action="store_true")
    parser.add_argument("--skip_pos_x_R_std",
                        help="skip the estimation of pos_x_R_std",
                        action="store_true")
    parser.add_argument("--skip_pos_y_R_std",
                        help="skip the estimation of pos_y_R_std",
                        action="store_true")
    parser.add_argument("--skip_m0",
                        help="skip the estimation of m0",
                        action="store_true")
    parser.add_argument("--skip_sqrt_diag_V0",
                        help="skip the estimation of sqrt_diag_V0",
                        action="store_true")
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
    skip_sigma_a = args.skip_sigma_a
    skip_pos_x_R_std = args.skip_pos_x_R_std
    skip_pos_y_R_std = args.skip_pos_y_R_std
    skip_m0 = args.skip_m0
    skip_sqrt_diag_V0 = args.skip_sqrt_diag_V0
    simRes_filename_pattern = args.simRes_filename_pattern
    estInit_metadata_filename_pattern = args.estInit_metadata_filename_pattern
    estRes_metadata_filename_pattern = args.estRes_metadata_filename_pattern
    estRes_data_filename_pattern = args.estRes_data_filename_pattern

    simRes_filename = simRes_filename_pattern.format(simRes_number)
    simRes = np.load(simRes_filename)
    y = simRes["y"][:2,:]

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
    pos_x_V0_std = float(estMeta["initial_state"]["pos_x_std"])
    pos_y_V0_std = float(estMeta["initial_state"]["pos_y_std"])
    vel_x_V0_std = float(estMeta["initial_state"]["vel_x_std"])
    vel_y_V0_std = float(estMeta["initial_state"]["vel_y_std"])
    acc_x_V0_std = float(estMeta["initial_state"]["acc_x_std"])
    acc_y_V0_std = float(estMeta["initial_state"]["acc_y_std"])

    sigma_a = torch.DoubleTensor([float(estMeta["state_cov"]["sigma_a"])])

    pos_x_R_std = \
        torch.DoubleTensor([float(estMeta["measurements_cov"]["pos_x_std"])])
    pos_y_R_std = \
        torch.DoubleTensor([float(estMeta["measurements_cov"]["pos_y_std"])])

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

    m0_0 = torch.DoubleTensor([pos_x_m0, vel_x_m0, acc_x_m0,
                               pos_y_m0, vel_y_m0, acc_y_m0])
    sqrt_diag_V0_0 = torch.DoubleTensor([pos_x_V0_std, vel_x_V0_std, acc_x_V0_std,
                                         pos_y_V0_std, vel_y_V0_std, acc_y_V0_std])

    y = torch.from_numpy(y.astype(np.double))

    vars_to_estimate = {}
    if skip_sigma_a:
        vars_to_estimate["sigma_a"] = False
    else:
        vars_to_estimate["sigma_a"] = True

    if skip_pos_x_R_std:
        vars_to_estimate["pos_x_R_std"] = False
    else:
        vars_to_estimate["pos_x_R_std"] = True

    if skip_pos_y_R_std:
        vars_to_estimate["pos_y_R_std"] = False
    else:
        vars_to_estimate["pos_y_R_std"] = True

    if skip_m0:
        vars_to_estimate["m0"] = False
    else:
        vars_to_estimate["m0"] = True

    if skip_sqrt_diag_V0:
        vars_to_estimate["sqrt_diag_V0"] = False
    else:
        vars_to_estimate["sqrt_diag_V0"] = True

    B, Q, Qe, Z, R_0 = lds.tracking.utils.getLDSmatricesForKinematics_torch(dt=dt,
                                                                            sigma_a=sigma_a,
                                                                            pos_x_R_std=pos_x_R_std,
                                                                            pos_y_R_std=pos_y_R_std)

    optim_res = lds.learning.torch_lbfgs_optimize_SS_tracking_diagV0(
        y=y, B=B, sigma_a0=sigma_a, Qe=Qe, Z=Z,
        pos_x_R_std0=pos_x_R_std, pos_y_R_std0=pos_y_R_std,
        m0_0=m0_0, sqrt_diag_V0_0=sqrt_diag_V0_0,
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
