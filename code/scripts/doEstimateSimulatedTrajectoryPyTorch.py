import sys
import os.path
import argparse
import configparser
import math
import random
import pickle
import numpy as np
import torch

import lds_python.learning


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simRes_number", help="simulation result number",
                        type=int)
    parser.add_argument("estMeta_number", help="estimation metadata number",
                        type=int)
    parser.add_argument("--skip_estimation_sqrt_noise_intensity",
                        help=("use this option to skip the estimation of the "
                              "sqrt noise inensity"), action="store_true")
    parser.add_argument("--skip_estimation_R",
                        help="use this option to skip the estimation of R",
                        action="store_true")
    parser.add_argument("--skip_estimation_m0",
                        help="use this option to skip the estimation of m0",
                        action="store_true")
    parser.add_argument("--skip_estimation_V0",
                        help="use this option to skip the estimation of V0",
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
    skip_estimation_sqrt_noise_intensity = \
        args.skip_estimation_sqrt_noise_intensity
    skip_estimation_R = args.skip_estimation_R
    skip_estimation_m0 = args.skip_estimation_m0
    skip_estimation_V0 = args.skip_estimation_V0
    simRes_filename_pattern = args.simRes_filename_pattern
    estInit_metadata_filename_pattern = args.estInit_metadata_filename_pattern
    estRes_metadata_filename_pattern = args.estRes_metadata_filename_pattern
    estRes_data_filename_pattern = args.estRes_data_filename_pattern

    simRes_filename = simRes_filename_pattern.format(simRes_number)
    simRes = np.load(simRes_filename)
    estInit_metadata_filename = \
        estInit_metadata_filename_pattern.format(estMeta_number)

    estMeta = configparser.ConfigParser()
    estMeta.read(estInit_metadata_filename)
    pos_x0 = float(estMeta["initial_params"]["pos_x0"])
    pos_y0 = float(estMeta["initial_params"]["pos_y0"])
    vel_x0 = float(estMeta["initial_params"]["vel_x0"])
    vel_y0 = float(estMeta["initial_params"]["vel_y0"])
    ace_x0 = float(estMeta["initial_params"]["ace_x0"])
    ace_y0 = float(estMeta["initial_params"]["ace_y0"])
    sqrt_noise_intensity0 = float(estMeta["initial_params"]["sqrt_noise_intensity"])
    sigma_x0 = float(estMeta["initial_params"]["sigma_x"])
    sigma_y0 = float(estMeta["initial_params"]["sigma_y"])
    sqrt_diag_V0_value = float(estMeta["initial_params"]["sqrt_diag_v0_value"])
    em_max_iter = int(estMeta["optim_params"]["em_max_iter"])
    tolerance_grad = float(estMeta["optim_params"]["tolerance_grad"])
    tolerance_change = float(estMeta["optim_params"]["tolerance_change"])

    if math.isnan(pos_x0):
        pos_x0 = simRes["y"][0, 0]
    if math.isnan(pos_y0):
        pos_y0 = simRes["y"][1, 0]

    sqrt_diag_R_0 = torch.DoubleTensor([sigma_x0, sigma_y0])
    m0_0 = torch.DoubleTensor([pos_x0, vel_x0, ace_x0, pos_y0, vel_y0, ace_y0])
    sqrt_diag_V0_0 = torch.DoubleTensor([sqrt_diag_V0_value
                                         for i in range(len(m0_0))])
    m0_0 = torch.unsqueeze(m0_0, dim=1)

    y = torch.from_numpy(simRes["y"].astype(np.double))
    B = torch.from_numpy(simRes["B"].astype(np.double))
    Qe = torch.from_numpy(simRes["Qe"].astype(np.double))
    Z = torch.from_numpy(simRes["Z"].astype(np.double))
    vars_to_estimate = {}

    if skip_estimation_sqrt_noise_intensity:
        vars_to_estimate["sqrt_noise_intensity"] = False
    else:
        vars_to_estimate["sqrt_noise_intensity"] = True

    if skip_estimation_R:
        vars_to_estimate["skip_estimation_R"] = False
    else:
        vars_to_estimate["skip_estimation_R"] = True

    if skip_estimation_m0:
        vars_to_estimate["skip_estimation_m0"] = False
    else:
        vars_to_estimate["skip_estimation_m0"] = True

    if skip_estimation_V0:
        vars_to_estimate["skip_estimation_V0"] = False
    else:
        vars_to_estimate["skip_estimation_V0"] = True

    if len(vars_to_estimate) == 0:
        ValueError("No variable to estimate.")

    optim_res = lds_python.learning.torch_optimize_SS_tracking_diagV0(
        y=y, B=B, sqrt_noise_intensity0=sqrt_noise_intensity0, Qe=Qe,
        Z=Z, sqrt_diag_R_0=sqrt_diag_R_0, m0_0=m0_0,
        sqrt_diag_V0_0=sqrt_diag_V0_0, max_iter=em_max_iter,
        vars_to_estimate=vars_to_estimate, tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change)

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
    estimRes_metadata["estimation_params"] = {"estInitNumber": estMeta_number}
    with open(estRes_metadata_filename, "w") as f:
        estimRes_metadata.write(f)

    with open(estRes_data_filename, "wb") as f:
        pickle.dump(optim_res, f)
    print("Saved results to {:s}".format(estRes_data_filename))
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
