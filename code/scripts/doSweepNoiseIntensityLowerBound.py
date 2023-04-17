import sys
import argparse
import configparser
import numpy as np
import plotly.graph_objects as go

import lds_python.inference
import lds_python.learning


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simRes_number", type=int,
                        help="simulation result number")
    parser.add_argument("--sqrt_noise_intensity_old", type=float, default=0.05,
                        help="old value for sqrt_noise_intensity")
    parser.add_argument("--sqrt_noise_intensity_min", type=float, default=0.01,
                        help="minimum value for sqrt_noise_intensity")
    parser.add_argument("--sqrt_noise_intensity_max", type=float, default=2.1,
                        help="maximum value for sqrt_noise_intensity")
    parser.add_argument("--sqrt_noise_intensity_step", type=float, default=0.001,
                        help="step value for sqrt_noise_intensity")
    parser.add_argument("--simRes_filename_pattern", type=str,
                        default="../../results/{:08d}_simulation.{:s}",
                        help="simulation results filename pattern")
    parser.add_argument("--filtering_params_filename", type=str,
                        default="", help="filtering parameters filename")
    parser.add_argument("--filtering_params_section", type=str,
                        default="initial_params",
                        help=("section of ini file containing the filtering "
                              "params"))
    parser.add_argument("--fig_filename_pattern", type=str,
                        default=("../../figures/{:08d}_lowerBound_sqrt_noise_intensity_"
                                 "sweep.{:s}"),
                        help="figure filename pattern")
    args = parser.parse_args()

    simRes_number = args.simRes_number
    sqrt_noise_intensity_old = args.sqrt_noise_intensity_old
    sqrt_noise_intensity_min = args.sqrt_noise_intensity_min
    sqrt_noise_intensity_max = args.sqrt_noise_intensity_max
    sqrt_noise_intensity_step = args.sqrt_noise_intensity_step
    simRes_filename_pattern = args.simRes_filename_pattern
    filtering_params_filename = args.filtering_params_filename
    filtering_params_section = args.filtering_params_section
    fig_filename_pattern = args.fig_filename_pattern

    simRes_filename = simRes_filename_pattern.format(simRes_number, "npz")
    simRes = np.load(simRes_filename)

    y = simRes["y"]
    B = simRes["B"]
    Qe = simRes["Qe"]
    m0 = simRes["m0"]
    V0 = simRes["V0"]
    Z = simRes["Z"]
    R = simRes["R"]

    Q = Qe * sqrt_noise_intensity_old**2

    if m0.ndim == 1:
        m0 = np.expand_dims(m0, axis=1)

    N = simRes["y"].shape[1]
    M = Qe.shape[0]

    kf = lds_python.inference.filterLDS_SS_withMissingValues_np(
        y=y, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
    ks = lds_python.inference.smoothLDS_SS(B=B, xnn=kf["xnn"], Vnn=kf["Vnn"],
                                           xnn1=kf["xnn1"], Vnn1=kf["Vnn1"],
                                           m0=m0, V0=V0)
    S11, S10, S00 = lds_python.learning.posteriorCorrelationMatrices(
        Z=Z, B=B, KN=kf["KN"], Vnn=kf["Vnn"], xnN=ks["xnN"], VnN=ks["VnN"],
        x0N=ks["x0N"], V0N=ks["V0N"], Jn=ks["Jn"], J0=ks["J0"])
    W = S11 - S10 @ B.T - B @ S10.T + B @ S00 @ B.T
    Qe_inv = np.linalg.inv(Qe)
    U = W @ Qe_inv
    K = np.trace(U)

    sqrt_noise_intensitys = np.arange(sqrt_noise_intensity_min, sqrt_noise_intensity_max, sqrt_noise_intensity_step)
    lower_bounds = np.empty(len(sqrt_noise_intensitys))
    for i, sqrt_noise_intensity in enumerate(sqrt_noise_intensitys):
        noise_intensity = sqrt_noise_intensity**2
        lower_bounds[i] = -N*M/2*np.log(noise_intensity)-K/(2*noise_intensity)
        print(f"lower bound for sqrt_noise_intensity={sqrt_noise_intensity:.04f}: {lower_bounds[i]}")
    fig = go.Figure()
    trace = go.Scatter(x=sqrt_noise_intensitys, y=lower_bounds, mode="lines+markers")
    fig.add_trace(trace)
    fig.update_layout(xaxis_title="Noise Intensity", yaxis_title="Lower Bound")
    fig.write_image(fig_filename_pattern.format(simRes_number, "png"))
    fig.write_html(fig_filename_pattern.format(simRes_number, "html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
