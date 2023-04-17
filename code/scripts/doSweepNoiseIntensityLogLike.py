import sys
import argparse
import configparser
import numpy as np
import plotly.graph_objects as go

import lds_python.inference


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simRes_number", type=int,
                        help="simulation result number")
    parser.add_argument("--sqrt_noise_intensity_min", type=float, default=0.001,
                        help="minimum value for sqrt_noise_intensity")
    parser.add_argument("--sqrt_noise_intensity_max", type=float, default=2.001,
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
                        default="../../figures/{:08d}_logLike_sqrt_noise_intensity_sweep.{:s}",
                        help="figure filename pattern")
    args = parser.parse_args()

    simRes_number = args.simRes_number
    sqrt_noise_intensity_min = args.sqrt_noise_intensity_min
    sqrt_noise_intensity_max = args.sqrt_noise_intensity_max
    sqrt_noise_intensity_step = args.sqrt_noise_intensity_step
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
        # sqrt_noise_intensityx = float(smoothing_params[filtering_params_section]["sqrt_noise_intensityx"])
        # sqrt_noise_intensityy = float(smoothing_params[filtering_params_section]["sqrt_noise_intensityy"])
        sigma_x = float(smoothing_params[filtering_params_section]["sigma_x"])
        sigma_y = float(smoothing_params[filtering_params_section]["sigma_y"])
        sqrt_diag_V0_value = float(smoothing_params[filtering_params_section]
                                                   ["sqrt_diag_V0_value"])

        m0 = np.array([pos_x0, vel_x0, acc_x0, pos_y0, vel_y0, acc_y0],
                      dtype=np.double)
        V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)
        R = np.diag([sigma_x**2, sigma_y**2])
    else:
        # sqrt_noise_intensityx = simRes["sqrt_noise_intensityx"]
        # sqrt_noise_intensityy = simRes["sqrt_noise_intensityy"]
        # sqrt_noise_intensityx = simRes["sqrt_noise_intensity"]
        # sqrt_noise_intensityy = simRes["sqrt_noise_intensity"]
        m0 = simRes["m0"]
        V0 = simRes["V0"]
        R = simRes["R"]
        Qe = simRes["Qe"]

    if m0.ndim == 1:
        m0 = np.expand_dims(m0, axis=1)

    sqrt_noise_intensities = np.arange(sqrt_noise_intensity_min, sqrt_noise_intensity_max, sqrt_noise_intensity_step)
    log_likes = np.empty(len(sqrt_noise_intensities))
    for i, sqrt_noise_intensity in enumerate(sqrt_noise_intensities):
        Q = Qe * sqrt_noise_intensity**2
        filterRes = lds_python.inference.filterLDS_SS_withMissingValues_np(
            y=simRes["y"], B=simRes["B"], Q=Q, m0=m0, V0=V0, Z=simRes["Z"],
            R=R)
        log_likes[i] = filterRes["logLike"]
        print(f"log likelihood for sqrt_noise_intensity={sqrt_noise_intensity:.04f}: {log_likes[i]}")

    argmax = np.argmax(log_likes)
    max_ll = log_likes[argmax]
    max_sqrt_noise_intensity = sqrt_noise_intensities[argmax]
    print(f"max log-likelihood: {max_ll}, "
          f"max sqrt noise intensity: {max_sqrt_noise_intensity}")

    fig = go.Figure()
    trace = go.Scatter(x=sqrt_noise_intensities, y=log_likes, mode="lines+markers")
    fig.add_trace(trace)
    fig.update_layout(xaxis_title=r"$Noise Intensity$",
                      yaxis_title="Log Likelihood")
    fig.write_image(fig_filename_pattern.format(simRes_number, "png"))
    fig.write_html(fig_filename_pattern.format(simRes_number, "html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
