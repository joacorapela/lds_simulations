import sys
import pickle
import numpy as np
import argparse
import configparser
import plotly.graph_objects as go
import plotly.figure_factory as ff
import lds.tracking.plotting
import lds.tracking.utils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("filtered_data_number", type=int,
                        help="number corresponding to filtered results filename")
    parser.add_argument("--variable", type=str, default="posAndHO2D",
                        help="variable to plot: pos2D, posAndHO2D, pos, vel, acc, hor, avel")
    parser.add_argument("--arrow_scale_factor", type=float, default="1.0",
                        help="arrow scale factor")
    parser.add_argument("--color_measured", type=str, default="black",
                        help="color for measured markers")
    parser.add_argument("--color_true", type=str, default="blue",
                        help="color for true markers")
    parser.add_argument("--color_filtered", type=str, default="red",
                        help="color for filtered markers")
    parser.add_argument("--color_pattern_filtered", type=str,
                        default="rgba(255,0,0,{:f})",
                        help="color pattern for filtered trace")
    parser.add_argument("--cb_alpha", type=float,
                        default=0.3,
                        help="transparency alpha for confidence band")
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
    arrow_scale_factor = args.arrow_scale_factor
    color_measured = args.color_measured
    color_true = args.color_true
    color_filtered = args.color_filtered
    color_pattern_filtered = args.color_pattern_filtered
    cb_alpha = args.cb_alpha
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
    sim_res_filename = sim_res_filename_pattern.format(sim_res_number, "npz")
    sim_res = np.load(sim_res_filename)
    sim_metadata_filename = sim_res_filename_pattern.format(sim_res_number, "ini")
    sim_metadata = configparser.ConfigParser()
    sim_metadata.read(sim_metadata_filename)
    simulation_params_filename = sim_metadata["params"]["simulation_params_filename"]
    simulation_params = configparser.ConfigParser()
    simulation_params.read(simulation_params_filename)
    dt = float(simulation_params["other"]["dt"])

    N = sim_res["y"].shape[1]
    time = np.arange(0, N*dt, dt)
    if variable == "pos2D":
        measurements = sim_res["y"]
        true_values = sim_res["x"][(0, 3),]
        filtered_means = filtered_data["filter_res"]["xnn"][(0, 3),0,:]
        fig = lds.tracking.plotting.get_positions_x_vs_y_fig(
            time=time, measurements=measurements, true_values=true_values,
            filtered_means=filtered_means)
    elif variable == "cosTheta":
        measurements = sim_res["y"][2,]
        true_values = sim_res["x"][6,]
        filtered_mean = filtered_data["filter_res"]["xnn"][6, 0,].numpy()
        filtered_stds = np.sqrt(filtered_data["filter_res"]["Pnn"][6, 6,]).numpy()
        filtered_ci_upper = filtered_mean + 1.96*filtered_stds
        filtered_ci_lower = filtered_mean - 1.96*filtered_stds

        fig = go.Figure()

        trace = go.Scatter(
            x=time, y=measurements,
            mode="lines+markers",
            marker={"color": color_measured},
            name="measurements",
            showlegend=True,
        )
        fig.add_trace(trace)

        trace = go.Scatter(
            x=time, y=true_values,
            mode="lines+markers",
            marker={"color": color_true},
            name="true",
            showlegend=True,
        )
        fig.add_trace(trace)

        trace = go.Scatter(
            x=time, y=filtered_mean,
            mode="lines+markers",
            marker={"color": color_pattern_filtered.format(1.0)},
            name="filtered",
            showlegend=True,
            legendgroup="filtered",
        )
        fig.add_trace(trace)

        trace = go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([filtered_ci_upper,
                              filtered_ci_lower[::-1]]),
            fill="toself",
            fillcolor=color_pattern_filtered.format(cb_alpha),
            line=dict(color=color_pattern_filtered.format(0.0)),
            showlegend=False,
            legendgroup="filtered",
        )
        fig.add_trace(trace)

        fig.update_layout(title=f"Log-Likelihood: {filtered_data['filter_res']['logLike']}")
        fig.update_xaxes(title="Time (sec)")
        fig.update_yaxes(title=r"$\cos(\theta)$")

    elif variable == "posAndHO2D":
        trace_mes = go.Scatter(x=sim_res["y"][0, :], y=sim_res["y"][1, :],
                               mode="markers",
                               marker={"color": color_measured},
                               customdata=time,
                               hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                               name="measured",
                               showlegend=True,
                               )
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
        fig = go.Figure()
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
    elif variable == "pos":
        measurements = sim_res["y"][(0, 1), :]
        true_values = sim_res["x"][(0, 3), :]
        filtered_means = filtered_data["filter_res"]["xnn"][(0, 3), 0, :].numpy()
        filtered_stds = np.sqrt(np.diagonal(a=filtered_data["filter_res"]["Vnn"], axis1=0, axis2=1)[:, (0, 3)].T)
        fig = lds.tracking.plotting.get_x_and_y_time_series_vs_time_fig(
            time=time,
            ylabel="Position",
            measurements=measurements,
            true_values=true_values,
            filtered_means=filtered_means,
            filtered_stds=filtered_stds)
    elif variable == "vel":
        vel_true_diff_x = np.diff(sim_res["x"][0, :])/dt
        vel_true_diff_y = np.diff(sim_res["x"][3, :])/dt
        vel_true_diff = np.vstack([vel_true_diff_x, vel_true_diff_y])
        vel_measurements_diff_x = np.diff(sim_res["y"][0, :])/dt
        vel_measurements_diff_y = np.diff(sim_res["y"][3, :])/dt
        vel_measurements_diff = np.vstack([vel_measurements_diff_x, vel_measurements_diff_y])
        filtered_means = filtered_data["filter_res"]["xnn"][(1, 4), 0, :].numpy()
        filtered_stds = np.sqrt(np.diagonal(a=filtered_data["filter_res"]["Vnn"], axis1=0, axis2=1)[:, (1, 4)].T)
        fig = lds.tracking.plotting.get_x_and_y_time_series_vs_time_fig(
            time=time,
            ylabel="Velocity",
            measurements=vel_measurements_diff,
            true_values=vel_true_diff,
            filtered_means=filtered_means,
            filtered_stds=filtered_stds)
    elif variable == "acc":
        vel_true_diff_x = np.diff(sim_res["x"][0, :])/dt
        vel_true_diff_y = np.diff(sim_res["x"][3, :])/dt
        acc_true_diff_x = np.diff(vel_true_diff_x)/dt
        acc_true_diff_y = np.diff(vel_true_diff_y)/dt
        acc_true_diff = np.vstack([acc_true_diff_x, acc_true_diff_y])
        vel_measurements_diff_x = np.diff(sim_res["y"][0, :])/dt
        vel_measurements_diff_y = np.diff(sim_res["y"][3, :])/dt
        acc_measurements_diff_x = np.diff(vel_measurements_diff_x)/dt
        acc_measurements_diff_y = np.diff(vel_measurements_diff_y)/dt
        acc_measurements_diff = np.vstack([acc_measurements_diff_x, acc_measurements_diff_y])
        filtered_means = filtered_data["filter_res"]["xnn"][(2, 5), 0, :].numpy()
        filtered_stds = np.sqrt(np.diagonal(a=filtered_data["filter_res"]["Vnn"], axis1=0, axis2=1)[:, (2, 5)].T)
        fig = lds.tracking.plotting.get_x_and_y_time_series_vs_time_fig(
            time=time,
            ylabel="Acceleration",
            measurements=acc_measurements_diff,
            true_values=acc_true_diff,
            filtered_means=filtered_means,
            filtered_stds=filtered_stds)
    elif variable == "hor":
        measurements = sim_res["y"][(2, 3), :]
        true_values = sim_res["x"][(6, 7), :]
        filtered_means = filtered_data["filter_res"]["xnn"][(6, 7), 0, :].numpy()
        filtered_stds = np.sqrt(np.diagonal(a=filtered_data["filter_res"]["Vnn"], axis1=0, axis2=1)[:, (6, 7)].T)
        fig = lds.tracking.plotting.get_x_and_y_time_series_vs_time_fig(
            time=time,
            ylabel="Cosine (x) or Sine (y) of Head Orientation Angle",
            measurements=measurements,
            true_values=true_values,
            filtered_means=filtered_means,
            filtered_stds=filtered_stds)
    elif variable == "avel":
        true_values = sim_res["x"][8, :]
        filtered_means = filtered_data["filter_res"]["xnn"][8, 0, :].numpy()
        filtered_stds = np.sqrt(np.diagonal(a=filtered_data["filter_res"]["Vnn"], axis1=0, axis2=1)[:, 8].T)
        fig = lds.tracking.plotting.get_x_time_series_vs_time_fig(
            time=time,
            ylabel="Angular Velocity",
            true_values=true_values,
            filtered_means=filtered_means,
            filtered_stds=filtered_stds)
    else:
        raise ValueError("variable={:s} is invalid. It should be: pos, vel, acc".format(variable))

    fig.update_layout(title=f'Log-Likelihood: {filtered_data["filter_res"]["logLike"]}')
    fig.write_image(fig_filename_pattern.format(filtered_data_number, variable, "png"))
    fig.write_html(fig_filename_pattern.format(filtered_data_number, variable, "html"))
    fig.show()
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
