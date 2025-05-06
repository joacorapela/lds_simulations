import sys
import pickle
import numpy as np
import argparse
import configparser
import plotly.graph_objects as go
import plotly.figure_factory as ff


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("filtered_data_number", type=int,
                        help="number corresponding to filtered results filename")
    parser.add_argument("--variable", type=str, default="posAndHO2D",
                        help="variable to plot: pos, vel, acc")
    parser.add_argument("--color_measured", type=str, default="black",
                        help="color for measured markers")
    parser.add_argument("--color_true", type=str, default="blue",
                        help="color for true markers")
    parser.add_argument("--color_filtered", type=str, default="red",
                        help="color for filtered markers")
    # parser.add_argument("--color_smoothed", type=str, default="green",
    #                     help="color for smoothed markers")
    parser.add_argument("--symbol_x", type=str, default="circle",
                        help="color for x markers")
    parser.add_argument("--symbol_y", type=str, default="circle-open",
                        help="color for y markers")
    parser.add_argument("--filtered_data_filenames_pattern", type=str,
                        default="../../results/{:08d}_filtered.{:s}",
                        help="filtered_data filename pattern")
    parser.add_argument("--sim_res_filename_pattern", type=str,
                        default="../../results/{:08d}_simulation.{:s}",
                        help="simulation results filename pattern")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/{:08d}_{:s}_filtered.{:s}")

    args = parser.parse_args()

    filtered_data_number = args.filtered_data_number
    variable = args.variable
    color_measured = args.color_measured
    color_true = args.color_true
    color_filtered = args.color_filtered
    # color_smoothed = args.color_smoothed
    symbol_x = args.symbol_x
    symbol_y = args.symbol_y
    filtered_data_filenames_pattern = \
        args.filtered_data_filenames_pattern
    sim_res_filename_pattern = args.sim_res_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    filtered_data_filename = \
        filtered_data_filenames_pattern.format(filtered_data_number, "pickle")
    with open(filtered_data_filename, "rb") as f:
        filtered_data = pickle.load(f)

    filtered_metadata_filename = \
        filtered_data_filenames_pattern.format(filtered_data_number, "ini")
    filtered_metadata = configparser.ConfigParser()
    filtered_metadata.read(filtered_metadata_filename)
    sim_res_number = int(filtered_metadata["params"]["sim_res_num"])
    sim_res_filename = sim_res_filename_pattern.format(sim_res_number, "pickle")
    with open(sim_res_filename, "rb") as f:
        sim_res = pickle.load(f)
    sim_metadata_filename = sim_res_filename_pattern.format(sim_res_number, "ini")
    sim_metadata = configparser.ConfigParser()
    sim_metadata.read(sim_metadata_filename)
    dt = float(sim_metadata["params"]["dt"])

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
                                name="true",
                                showlegend=True,
                                )
        trace_filtered = go.Scatter(x=filtered_data["filter_res"]["xnn"][0,0,:],
                                    y=filtered_data["filter_res"]["xnn"][3,0,:],
                                    mode="markers",
                                    marker={"color": color_filtered},
                                    customdata=time,
                                    hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                                    name="filtered",
                                    showlegend=True,
                                    )
        # trace_filtered = go.Scatter(x=filtered_data["spos1"],
        #                             y=filtered_data["spos2"],
        #                             mode="markers",
        #                             marker={"color": color_filtered},
        #                             customdata=time,
        #                             hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
        #                             name="filtered",
        #                             showlegend=True,
        #                             )
        fig.add_trace(trace_mes)
        fig.add_trace(trace_true)
        fig.add_trace(trace_filtered)
        # fig.add_trace(trace_filtered)
        fig.update_layout(xaxis_title="x (pixels)", yaxis_title="y (pixels)",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    elif variable == "posAndHO2D":
        trace_mes = go.Scatter(x=sim_res["y"][0, :], y=sim_res["y"][1, :],
                               mode="markers",
                               marker={"color": color_measured},
                               customdata=time,
                               hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                               name="measured",
                               showlegend=True,
                               )
        arrow_scale_factor = 1e-3
        fig_quiver_mes = ff.create_quiver(x=sim_res["y"][0, :], y=sim_res["y"][1, :],
                                          u=sim_res["y"][2, :]*arrow_scale_factor,
                                          v=sim_res["y"][3, :]*arrow_scale_factor,
                                          line={"color": color_measured},
                                          name="measured",
                                          legendgroup="measured")
        trace_true = go.Scatter(x=sim_res["x"][0, :], y=sim_res["x"][3, :],
                                mode="markers",
                                marker={"color": color_true},
                                customdata=time,
                                hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                                name="true",
                                showlegend=True,
                                )
        fig_quiver_true = ff.create_quiver(x=sim_res["x"][0, :], y=sim_res["x"][3, :],
                                           u=sim_res["x"][6, :]*arrow_scale_factor,
                                           v=sim_res["x"][7, :]*arrow_scale_factor,
                                           line={"color": color_true},
                                           name="true",
                                           legendgroup="true")
        trace_filtered = go.Scatter(x=filtered_data["filter_res"]["xnn"][0, 0, :],
                                    y=filtered_data["filter_res"]["xnn"][3, 0, :],
                                    mode="markers",
                                    marker={"color": color_filtered},
                                    customdata=time,
                                    hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                                    name="filtered",
                                    showlegend=True,
                                    )
        fig_quiver_filtered = ff.create_quiver(x=filtered_data["filter_res"]["xnn"][0, 0, :],
                                          y=filtered_data["filter_res"]["xnn"][3, 0, :],
                                          u=filtered_data["filter_res"]["xnn"][6, 0, :]*arrow_scale_factor,
                                          v=filtered_data["filter_res"]["xnn"][7, 0, :]*arrow_scale_factor,
                                          line={"color": color_filtered},
                                          name="filtered",
                                          legendgroup="filtered")
        # trace_filtered = go.Scatter(x=filtered_data["spos1"],
        #                             y=filtered_data["spos2"],
        #                             mode="markers",
        #                             marker={"color": color_filtered},
        #                             customdata=time,
        #                             hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
        #                             name="filtered",
        #                             showlegend=True,
        #                             )
        fig.add_trace(trace_mes)
        fig.add_traces(data=fig_quiver_mes.data)
        fig.add_trace(trace_true)
        fig.add_traces(data=fig_quiver_true.data)
        fig.add_trace(trace_filtered)
        fig.add_traces(data=fig_quiver_filtered.data)
        # fig.add_trace(trace_filtered)
        fig.update_layout(xaxis_title="x (pixels)", yaxis_title="y (pixels)",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    elif variable == "vel":
        dt = time[1] - time[0]
        vel_finite_diff_x = np.diff(sim_res["y"][0, :])/dt
        vel_finite_diff_y = np.diff(sim_res["y"][1, :])/dt
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
        trace_finite_diff_x = go.Scatter(x=time, y=vel_finite_diff_x,
                                         mode="markers",
                                         marker={"color": color_measured,
                                                 "symbol": symbol_x},
                                         name="finite diff x",
                                         showlegend=True)
        trace_finite_diff_y = go.Scatter(x=time, y=vel_finite_diff_y,
                                         mode="markers",
                                         marker={"color": color_measured,
                                                 "symbol": symbol_y},
                                         name="finite diff y",
                                         showlegend=True)
        trace_filtered_x = go.Scatter(x=time,
                                      y=filtered_data["fvel1"],
                                      mode="markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_x},
                                      name="filtered x",
                                      showlegend=True,
                                      )
        trace_filtered_y = go.Scatter(x=time,
                                      y=filtered_data["fvel2"],
                                      mode="markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_y},
                                      name="filtered y",
                                      showlegend=True,
                                      )
        trace_filtered_x = go.Scatter(x=time,
                                      y=filtered_data["svel1"],
                                      mode="markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_x},
                                      name="filtered x",
                                      showlegend=True,
                                      )
        trace_filtered_y = go.Scatter(x=time,
                                      y=filtered_data["svel2"],
                                      mode="markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_y},
                                      name="filtered y",
                                      showlegend=True,
                                      )
        fig.add_trace(trace_true_x)
        fig.add_trace(trace_true_y)
        fig.add_trace(trace_finite_diff_x)
        fig.add_trace(trace_finite_diff_y)
        fig.add_trace(trace_filtered_x)
        fig.add_trace(trace_filtered_y)
        fig.add_trace(trace_filtered_x)
        fig.add_trace(trace_filtered_y)
        fig.update_layout(xaxis_title="time", yaxis_title="velocity",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    elif variable == "acc":
        dt = time[1] - time[0]
        vel_finite_diff_x = np.diff(sim_res["y"][0, :])/dt
        vel_finite_diff_y = np.diff(sim_res["y"][1, :])/dt
        acc_finite_diff_x = np.diff(vel_finite_diff_x)/dt
        acc_finite_diff_y = np.diff(vel_finite_diff_y)/dt
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
        trace_finite_diff_x = go.Scatter(x=time, y=acc_finite_diff_x,
                                         mode="markers",
                                         marker={"color": color_measured,
                                                 "symbol": symbol_x},
                                         name="finite diff x",
                                         showlegend=True)
        trace_finite_diff_y = go.Scatter(x=time, y=acc_finite_diff_y,
                                         mode="markers",
                                         marker={"color": color_measured,
                                                 "symbol": symbol_y},
                                         name="finite diff y",
                                         showlegend=True)
        trace_filtered_x = go.Scatter(x=time,
                                      y=filtered_data["facc1"],
                                      mode="markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_x},
                                      name="filtered x",
                                      showlegend=True,
                                      )
        trace_filtered_y = go.Scatter(x=time,
                                      y=filtered_data["facc2"],
                                      mode="markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_y},
                                      name="filtered y",
                                      showlegend=True,
                                      )
        trace_filtered_x = go.Scatter(x=time,
                                      y=filtered_data["sacc1"],
                                      mode="markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_x},
                                      name="filtered x",
                                      showlegend=True,
                                      )
        trace_filtered_y = go.Scatter(x=time,
                                      y=filtered_data["sacc2"],
                                      mode="markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_y},
                                      name="filtered x",
                                      showlegend=True,
                                      )
        fig.add_trace(trace_true_x)
        fig.add_trace(trace_true_y)
        fig.add_trace(trace_finite_diff_x)
        fig.add_trace(trace_finite_diff_y)
        fig.add_trace(trace_filtered_x)
        fig.add_trace(trace_filtered_y)
        fig.add_trace(trace_filtered_x)
        fig.add_trace(trace_filtered_y)
        fig.update_layout(xaxis_title="time", yaxis_title="acceleration",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    else:
        raise ValueError("variable={:s} is invalid. It should be: pos, vel, acc".format(variable))

    fig.write_image(fig_filename_pattern.format(filtered_data_number, variable, "png"))
    fig.write_html(fig_filename_pattern.format(filtered_data_number, variable, "html"))
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
