import sys
import argparse
import configparser
import numpy as np
import plotly.graph_objects as go

sys.path.append("../../src")
import inference
import learning


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simRes_number", type=int,
                        help="simulation result number")
    parser.add_argument("--sigma_a_old", type=float, default=1.3,
                        help="old value for sigma_a")
    parser.add_argument("--sigma_a_min", type=float, default=0.1,
                        help="minimum value for sigma_a")
    parser.add_argument("--sigma_a_max", type=float, default=10.1,
                        help="maximum value for sigma_a")
    parser.add_argument("--sigma_a_step", type=float, default=0.1,
                        help="step value for sigma_a")
    parser.add_argument("--simRes_filename_pattern", type=str,
                        default="../../../results/{:08d}_simulation.{:s}",
                        help="simulation results filename pattern")
    parser.add_argument("--filtering_params_filename", type=str,
                        default="", help="filtering parameters filename")
    parser.add_argument("--filtering_params_section", type=str,
                        default="initial_params",
                        help=("section of ini file containing the filtering "
                              "params"))
    parser.add_argument("--fig_filename_pattern", type=str,
                        default=("../../../figures/{:08d}_lowerBound_sigma_a_"
                                 "sweep.{:s}"),
                        help="figure filename pattern")
    args = parser.parse_args()

    simRes_number = args.simRes_number
    sigma_a_old = args.sigma_a_old
    sigma_a_min = args.sigma_a_min
    sigma_a_max = args.sigma_a_max
    sigma_a_step = args.sigma_a_step
    simRes_filename_pattern = args.simRes_filename_pattern
    filtering_params_filename = args.filtering_params_filename
    filtering_params_section = args.filtering_params_section
    fig_filename_pattern = args.fig_filename_pattern

    simRes_filename = simRes_filename_pattern.format(simRes_number, "npz")
    simRes = np.load(simRes_filename)

    if len(filtering_params_filename) > 0:
        smoothing_params = configparser.ConfigParser()
        smoothing_params.read(filtering_params_filename)
        pos_x0 = float(smoothing_params[filtering_params_section]["pos_x0"])
        pos_y0 = float(smoothing_params[filtering_params_section]["pos_y0"])
        vel_x0 = float(smoothing_params[filtering_params_section]["vel_x0"])
        vel_y0 = float(smoothing_params[filtering_params_section]["vel_x0"])
        acc_x0 = float(smoothing_params[filtering_params_section]["acc_x0"])
        acc_y0 = float(smoothing_params[filtering_params_section]["acc_x0"])
        # sigma_ax = float(smoothing_params[filtering_params_section]["sigma_ax"])
        # sigma_ay = float(smoothing_params[filtering_params_section]["sigma_ay"])
        sigma_x = float(smoothing_params[filtering_params_section]["sigma_x"])
        sigma_y = float(smoothing_params[filtering_params_section]["sigma_y"])
        sqrt_diag_V0_value = float(smoothing_params[filtering_params_section]
                                                   ["sqrt_diag_V0_value"])

        m0 = np.array([pos_x0, vel_x0, acc_x0, pos_y0, vel_y0, acc_y0],
                      dtype=np.double)
        V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)
        R = np.diag([sigma_x**2, sigma_y**2])
    else:
        # sigma_ax = simRes["sigma_ax"]
        # sigma_ay = simRes["sigma_ay"]
        # sigma_ax = simRes["sigma_a"]
        # sigma_ay = simRes["sigma_a"]
        m0 = simRes["m0"]
        V0 = simRes["V0"]
        R = simRes["R"]
        Qe = simRes["Qe"]
        Z = simRes["Z"]
        B = simRes["B"]

    Q = Qe * sigma_a_old**2
    kf = inference.filterLDS_SS_withMissingValues_np(
        y=simRes["y"], B=simRes["B"], Q=Q, m0=m0, V0=V0, Z=simRes["Z"], R=R)
    ks = inference.smoothLDS_SS(B=simRes["B"], xnn=kf["xnn"], Vnn=kf["Vnn"],
                                xnn1=kf["xnn1"], Vnn1=kf["Vnn1"],
                                m0=np.expand_dims(m0, 1), V0=V0)
    # begin compute K
    Vnn1N = learning.lag1CovSmootherLDS_SS(Z=Z, KN=kf["KN"], B=B,
                                           Vnn=kf["Vnn"],
                                           Jn=ks["Jn"], J0=ks["J0"])
    S11 = ks["xnN"][:, :, 0] @ ks["xnN"][:, :, 0].T + \
        ks["VnN"][:, :, 0]
    S10 = ks["xnN"][:, :, 0] @ ks["x0N"].T + Vnn1N[:, :, 0]
    S00 = ks["x0N"] @ ks["x0N"].T + ks["V0N"]
    N = simRes["y"].shape[1]
    M = len(m0)
    for i in range(1, N):
        S11 = S11 + ks["xnN"][:, :, i] @ ks["xnN"][:, :, i].T + \
              ks["VnN"][:, :, i]
        S10 = S10 + ks["xnN"][:, :, i] @ ks["xnN"][:, :, i-1].T + \
            Vnn1N[:, :, i]
        S00 = S00 + ks["xnN"][:, :, i-1] @ ks["xnN"][:, :, i-1].T + \
            ks["VnN"][:, :, i-1]
    # sigma_a
    W = S11 - S10 @ B.T - B @ S10.T + B @ S00 @ B.T
    # U = (np.linalg.solve(Qe.T, W.T)).T
    Qe_inv = np.linalg.pinv(Qe)
    U = W @ Qe_inv
    K = np.trace(U)
    # end compute K

    sigma_as = np.arange(sigma_a_min, sigma_a_max, sigma_a_step)
    lower_bounds = np.empty(len(sigma_as))
    for i, sigma_a in enumerate(sigma_as):
        lower_bounds[i] = -N*M/2*np.log(sigma_a**2)-K/(2*sigma_a**2)
        print(f"lower bound for sigma_a={sigma_a:.02f}: {lower_bounds[i]}")
    fig = go.Figure()
    trace = go.Scatter(x=sigma_as, y=lower_bounds, mode="lines+markers")
    fig.add_trace(trace)
    fig.update_layout(xaxis_title=r"$\sigma_a$", yaxis_title="Lower Bound")
    fig.write_image(fig_filename_pattern.format(simRes_number, "png"))
    fig.write_html(fig_filename_pattern.format(simRes_number, "html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
