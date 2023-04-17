import sys
import os.path
import configparser
import argparse
import random
import numpy as np
import plotly.graph_objs as go

import lds_python.simulation


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_metadata_number",
                        help="simulation metadata number", type=int)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--simMeta_filename_pattern", type=str,
                        default="../../metadata/{:08d}_simulation.ini",
                        help="simulation metadata filename pattern")
    parser.add_argument("--simRes_filename_pattern", type=str,
                        default="../../results/{:08d}_simulation.{:s}",
                        help="simulation results filename pattern")
    args = parser.parse_args()

    sim_metadata_number = args.sim_metadata_number
    no_plot = args.no_plot
    simMeta_filename_pattern = args.simMeta_filename_pattern
    simRes_filename_pattern = args.simRes_filename_pattern

    sim_metadata_filename = simMeta_filename_pattern.format(
        sim_metadata_number)
    sim_metadata = configparser.ConfigParser()
    sim_metadata.read(sim_metadata_filename)
    pos_x0 = float(sim_metadata["control_variables"]["pos_x0"])
    pos_y0 = float(sim_metadata["control_variables"]["pos_x0"])
    vel_x0 = float(sim_metadata["control_variables"]["vel_x0"])
    vel_y0 = float(sim_metadata["control_variables"]["vel_x0"])
    ace_x0 = float(sim_metadata["control_variables"]["ace_x0"])
    ace_y0 = float(sim_metadata["control_variables"]["ace_x0"])
    dt = float(sim_metadata["control_variables"]["dt"])
    num_pos = int(sim_metadata["control_variables"]["num_pos"])
    sqrt_noise_intensity = float(sim_metadata["control_variables"]["sqrt_noise_intensity"])
    sigma_x = float(sim_metadata["control_variables"]["sigma_x"])
    sigma_y = float(sim_metadata["control_variables"]["sigma_y"])
    sqrt_diag_V0_value = float(sim_metadata["control_variables"]["sqrt_diag_V0_value"])

    # Taken from the book
    # barShalomEtAl01-estimationWithApplicationToTrackingAndNavigation.pdf
    # section 6.2.3
    # Eq. 6.2.3-7
    B = np.array([[1, dt, .5*dt**2, 0, 0, 0],
                  [0, 1,  dt,       0, 0, 0],
                  [0, 0,  1,        0, 0, 0],
                  [0, 0,  0,        1, dt, .5*dt**2],
                  [0, 0,  0,        0, 1,  dt],
                  [0, 0,  0,        0, 0,  1]], dtype=np.double)
    Z = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0]], dtype=np.double)
    # Eq. 6.2.3-8
    Qe = np.array([[dt**5/20, dt**4/8, dt**3/6, 0, 0, 0],
                   [dt**4/8, dt**3/3,  dt**2/2, 0, 0, 0],
                   [dt**3/6, dt**2/2,  dt,      0, 0, 0],
                   [0, 0, 0,                    dt**5/20, dt**4/8, dt**3/6],
                   [0, 0, 0,                    dt**4/8, dt**3/3,  dt**2/2],
                   [0, 0, 0,                    dt**3/6, dt**2/2,  dt]],
                  dtype=np.double)
    Q = Qe*sqrt_noise_intensity**2
    R = np.diag([sigma_x**2, sigma_y**2])
    m0 = np.array([pos_x0, vel_x0, ace_x0, pos_y0, vel_y0, ace_y0],
                  dtype=np.double)
    V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)

    x0, x, y = lds_python.simulation.simulateLDS(N=num_pos, B=B, Q=Q, Z=Z, R=R,
                                                 m0=m0, V0=V0)

    # save simulation results
    sim_prefix_used = True
    while sim_prefix_used:
        simRes_number = random.randint(0, 10**8)
        simRes_metadata_filename = simRes_filename_pattern.format(
            simRes_number, "ini")
        if not os.path.exists(simRes_metadata_filename):
            sim_prefix_used = False
    simRes_results_filename = simRes_filename_pattern.format(simRes_number,
                                                             "npz")

    simResConfig = configparser.ConfigParser()
    simResConfig["simulation_params"] = {"sim_metadata_filename":
                                         sim_metadata_filename}
    simResConfig["simulation_results"] = {"sim_results_filename":
                                          simRes_results_filename}
    with open(simRes_metadata_filename, "w") as f:
        simResConfig.write(f)
    print("Saving results to {:s}".format(simRes_results_filename))
    np.savez(simRes_results_filename, dt=dt, x0=x0, x=x, y=y, B=B, Z=Z, Qe=Qe,
             sqrt_noise_intensity=sqrt_noise_intensity, R=R, m0=m0, V0=V0)

    if not no_plot:
        fig = go.Figure()
        trace_x = go.Scatter(x=x[0, :], y=x[3, :], mode="lines+markers",
                             showlegend=True, name="x")
        trace_y = go.Scatter(x=y[0, :], y=y[1, :], mode="lines+markers",
                             showlegend=True, name="y", opacity=0.3)
        trace_start = go.Scatter(x=[x0[0]], y=[x0[3]], mode="markers",
                                 text="x0", marker={"size": 7},
                                 showlegend=False)
        fig.add_trace(trace_x)
        fig.add_trace(trace_y)
        fig.add_trace(trace_start)
        fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
