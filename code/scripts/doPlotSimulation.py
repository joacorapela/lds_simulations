import sys
import numpy as np
import argparse
import configparser
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_res_number", type=int,
                        help="simulation result number")
    parser.add_argument("--variable", type=str, default="pos",
                        help="variable to plot: pos2D, pos, vel, acc, hoOmega")
    parser.add_argument("--color_measured", type=str, default="black",
                        help="color for measured markers")
    parser.add_argument("--color_true", type=str, default="blue",
                        help="color for true markers")
    parser.add_argument("--symbol_x", type=str, default="circle",
                        help="color for x markers")
    parser.add_argument("--symbol_y", type=str, default="circle-open",
                        help="color for y markers")
    parser.add_argument("--sim_res_filename_pattern", type=str,
                        default="../../results/{:08d}_simulation.{:s}",
                        help="simulation results filename pattern")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/{:08d}_simulation_{:s}.{:s}")

    args = parser.parse_args()

    sim_res_number = args.sim_res_number
    variable = args.variable
    color_measured = args.color_measured
    color_true = args.color_true
    symbol_x = args.symbol_x
    symbol_y = args.symbol_y
    sim_res_filename_pattern = args.sim_res_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    sim_res_filename = sim_res_filename_pattern.format(sim_res_number, "npz")
    sim_res = np.load(sim_res_filename)

    metadata_filename = sim_res_filename_pattern.format(sim_res_number, "ini")
    metadata = configparser.ConfigParser()
    metadata.read(metadata_filename)
    simulation_params_filename = metadata["params"]["simulation_params_filename"]
    simulation_params = configparser.ConfigParser()
    simulation_params.read(simulation_params_filename)
    dt = float(simulation_params["other"]["dt"])

    N = sim_res["y"].shape[1]
    time = np.arange(0, N*dt, dt)
    fig = go.Figure()
    if variable == "pos2D":
        trace_mes = go.Scatter(x=sim_res["y"][0, :], y=sim_res["y"][1, :],
                               mode="markers",
                               marker={"color": color_measured},
                               customdata=time,
                               hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                               name="measured",
                               showlegend=True,
                               )
        trace_true = go.Scatter(x=sim_res["x"][0, :], y=sim_res["x"][3, :],
                                mode="markers",
                                marker={"color": color_true},
                                customdata=time,
                                hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                                name="state position",
                                showlegend=True,
                                )
        fig.add_trace(trace_mes)
        fig.add_trace(trace_true)
        fig.update_layout(xaxis_title="x (pixels)",
                          yaxis_title="y (pixels)",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    elif variable == "pos":
        trace_true_x = go.Scatter(x=time, y=sim_res["x"][0, :],
                                  mode="markers",
                                  marker={"color": color_true,
                                          "symbol": symbol_x},
                                  name="true x",
                                  showlegend=True,
                                  )
        trace_true_y = go.Scatter(x=time, y=sim_res["x"][3, :],
                                  mode="markers",
                                  marker={"color": color_true,
                                          "symbol": symbol_y},
                                  name="true y",
                                  showlegend=True,
                                  )
        fig.add_trace(trace_true_x)
        fig.add_trace(trace_true_y)
        fig.update_layout(xaxis_title="time", yaxis_title="position",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    elif variable == "vel":
        trace_true_x = go.Scatter(x=time, y=sim_res["x"][1, :],
                                  mode="markers",
                                  marker={"color": color_true,
                                          "symbol": symbol_x},
                                  name="true x",
                                  showlegend=True,
                                  )
        trace_true_y = go.Scatter(x=time, y=sim_res["x"][4, :],
                                  mode="markers",
                                  marker={"color": color_true,
                                          "symbol": symbol_y},
                                  name="true y",
                                  showlegend=True,
                                  )
        fig.add_trace(trace_true_x)
        fig.add_trace(trace_true_y)
        fig.update_layout(xaxis_title="time", yaxis_title="velocity",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    elif variable == "acc":
        trace_true_x = go.Scatter(x=time, y=sim_res["x"][2, :],
                                  mode="markers",
                                  marker={"color": color_true,
                                          "symbol": symbol_x},
                                  name="true x",
                                  showlegend=True,
                                  )
        trace_true_y = go.Scatter(x=time, y=sim_res["x"][5, :],
                                  mode="markers",
                                  marker={"color": color_true,
                                          "symbol": symbol_y},
                                  name="true y",
                                  showlegend=True,
                                  )
        fig.add_trace(trace_true_x)
        fig.add_trace(trace_true_y)
        fig.update_layout(xaxis_title="time", yaxis_title="acceleration",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    elif variable == "hoOmega":
        trace_true_omega = go.Scatter(x=time, y=sim_res["x"][8, :],
                                      mode="markers",
                                      marker={"color": color_true,
                                              "symbol": symbol_x},
                                      name="true x",
                                      showlegend=True,
                                      )
        fig.add_trace(trace_true_omega)
        fig.update_layout(xaxis_title="time", yaxis_title=r"$\omega$",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    else:
        raise ValueError("variable={:s} is invalid. It should be: pos2D, pos, vel, acc,  hoOmega".format(variable))


    fig.write_image(fig_filename_pattern.format(sim_res_number, variable,
                                                "png"))
    fig.write_html(fig_filename_pattern.format(sim_res_number, variable,
                                               "html"))

    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
