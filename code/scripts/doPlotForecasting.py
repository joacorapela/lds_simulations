
import sys
import argparse
import configparser
import pickle
import numpy as np
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("forecasting_results_number", type=int,
                        help="number corresponding to filtered results filename")
    parser.add_argument("--variable", type=str, default="pos",
                        help="variable to plot: pos, vel, acc")
    parser.add_argument("--forecasting_results_filename_pattern",
                        help="forecasting results filename pattern",
                        default="../../results/{:08d}_forecasting.{:s}")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/{:08d}_{:s}_forecasting.{:s}")

    args = parser.parse_args()

    forecasting_results_number = args.forecasting_results_number
    variable = args.variable
    forecasting_results_filename_pattern = args.forecasting_results_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    forecasting_metadata_filename = forecasting_results_filename_pattern.format(
        forecasting_results_number, "ini")
    forecasting_results_filename = forecasting_results_filename_pattern.format(
        forecasting_results_number, "pickle")

    forecasting_metadata = configparser.ConfigParser()
    forecasting_metadata.read(forecasting_metadata_filename)
    h = int(forecasting_metadata["params"]["horizon"])
    filtering_results_number = int(forecasting_metadata["params"]["filtering_results_number"])
    filtering_results_filenames_pattern = forecasting_metadata["params"]["filtering_results_filenames_pattern"]
    filtered_metadata_filename = \
        filtering_results_filenames_pattern.format(filtering_results_number, "ini")
    filtered_metadata = configparser.ConfigParser()
    filtered_metadata.read(filtered_metadata_filename)
    sim_res_number = int(filtered_metadata["params"]["sim_res_num"])
    filtering_params_filename = filtered_metadata["params"]["filtering_params_filename"]

    # sim_res_filename_pattern = filtered_metadata["params"]["sim_res_filename_pattern"]
    sim_res_filename_pattern = "../../results/{:08d}_simulation.{:s}"

    sim_metadata_filename = sim_res_filename_pattern.format(sim_res_number, "ini")
    sim_metadata = configparser.ConfigParser()
    sim_metadata.read(sim_metadata_filename)
    simulation_params_filename = sim_metadata["params"]["simulation_params_filename"]
    simulation_params = configparser.ConfigParser()
    simulation_params.read(simulation_params_filename)
    dt = float(simulation_params["other"]["dt"])

    sim_res_filename = sim_res_filename_pattern.format(sim_res_number, "npz")
    sim_res = np.load(sim_res_filename)

    data = sim_res["y"].T
    data = data

    with open(forecasting_results_filename, "rb") as f:
        forecasting_results = pickle.load(f)
    x_pred = forecasting_results["x"]
    P_pred = forecasting_results["P"]
    log_like = forecasting_results["log_like"]

    N = data.shape[0]
    time = np.arange(N) * dt
    time_pred = np.concatenate((time[(h-1):], time[-1]+np.arange(1, h+1) * dt))

    if variable == "pos":
        true_x = sim_res["x"][0, :]
        x_forecast_mean = x_pred[0, 0, :]
        x_forecast_std = np.sqrt(P_pred[0, 0, :])
        x_forecast_95ci_down = x_forecast_mean - 1.96 * x_forecast_std
        x_forecast_95ci_up = x_forecast_mean + 1.96 * x_forecast_std

        true_y = sim_res["x"][3, :]
        y_forecast_mean = x_pred[3, 0, :]
        y_forecast_std = np.sqrt(P_pred[3, 3, :])
        y_forecast_95ci_down = y_forecast_mean - 1.96 * y_forecast_std
        y_forecast_95ci_up = y_forecast_mean + 1.96 * y_forecast_std

        color_pattern_filtered = "rgba(255,0,0,{:f})"
        cb_alpha = 0.3
        fig = go.Figure()

        trace = go.Scatter(x=time, y=true_x, name="true x", marker_color="blue")
        fig.add_trace(trace)

        trace = go.Scatter(x=time, y=true_y, name="true y", marker_color="blue")
        fig.add_trace(trace)

        trace = go.Scatter(x=time, y=data[:, 0], name="measurement x", marker_color="black")
        fig.add_trace(trace)

        trace = go.Scatter(x=time, y=data[:, 1], name="measurement y", marker_color="black")
        fig.add_trace(trace)

        trace = go.Scatter(x=time_pred, y=x_forecast_mean, name="forecast x", marker_color="red", legendgroup="forecast_x")
        fig.add_trace(trace)

        trace = go.Scatter(x=time_pred, y=y_forecast_mean, name="forecast y", marker_color="red", legendgroup="forecast_y")
        fig.add_trace(trace)

        trace = go.Scatter(x=np.concatenate((time_pred, time_pred[::-1])),
                           y=np.concatenate((x_forecast_95ci_up,
                                             x_forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast_x"
                           )
        fig.add_trace(trace)

        trace = go.Scatter(x=np.concatenate((time_pred, time_pred[::-1])),
                           y=np.concatenate((y_forecast_95ci_up,
                                             y_forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast_y"
                           )
        fig.add_trace(trace)
        fig.update_layout(title=f"Forecasting Horizon: {h} samples, Log-Likelihood: {log_like}")
        fig.update_xaxes(title="Time (sec)")
        fig.update_yaxes(title="Position")

    elif variable == "vel":
        true_x = sim_res["x"][1, :]
        x_forecast_mean = x_pred[1, 0, :]
        x_forecast_std = np.sqrt(P_pred[1, 1, :])
        x_forecast_95ci_down = x_forecast_mean - 1.96 * x_forecast_std
        x_forecast_95ci_up = x_forecast_mean + 1.96 * x_forecast_std

        true_y = sim_res["x"][4, :]
        y_forecast_mean = x_pred[4, 0, :]
        y_forecast_std = np.sqrt(P_pred[4, 4, :])
        y_forecast_95ci_down = y_forecast_mean - 1.96 * y_forecast_std
        y_forecast_95ci_up = y_forecast_mean + 1.96 * y_forecast_std

        color_pattern_filtered = "rgba(255,0,0,{:f})"
        cb_alpha = 0.3
        fig = go.Figure()

        trace = go.Scatter(x=time, y=true_x, name="true x", marker_color="blue")
        fig.add_trace(trace)

        trace = go.Scatter(x=time, y=true_y, name="true y", marker_color="blue")
        fig.add_trace(trace)

        trace = go.Scatter(x=time_pred, y=x_forecast_mean, name="forecast x", marker_color="red", legendgroup="forecast_x")
        fig.add_trace(trace)

        trace = go.Scatter(x=time_pred, y=y_forecast_mean, name="forecast y", marker_color="red", legendgroup="forecast_y")
        fig.add_trace(trace)

        trace = go.Scatter(x=np.concatenate((time_pred, time_pred[::-1])),
                           y=np.concatenate((x_forecast_95ci_up,
                                             x_forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast_x"
                           )
        fig.add_trace(trace)

        trace = go.Scatter(x=np.concatenate((time_pred, time_pred[::-1])),
                           y=np.concatenate((y_forecast_95ci_up,
                                             y_forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast_y"
                           )
        fig.add_trace(trace)
        fig.update_layout(title=f"Forecasting Horizon: {h} samples, Log-Likelihood: {log_like}")
        fig.update_xaxes(title="Time (sec)")
        fig.update_yaxes(title="Velocity")

    elif variable == "acc":
        true_x = sim_res["x"][2, :]
        x_forecast_mean = x_pred[2, 0, :]
        x_forecast_std = np.sqrt(P_pred[2, 2, :])
        x_forecast_95ci_down = x_forecast_mean - 1.96 * x_forecast_std
        x_forecast_95ci_up = x_forecast_mean + 1.96 * x_forecast_std

        true_y = sim_res["x"][5, :]
        y_forecast_mean = x_pred[5, 0, :]
        y_forecast_std = np.sqrt(P_pred[5, 5, :])
        y_forecast_95ci_down = y_forecast_mean - 1.96 * y_forecast_std
        y_forecast_95ci_up = y_forecast_mean + 1.96 * y_forecast_std

        color_pattern_filtered = "rgba(255,0,0,{:f})"
        cb_alpha = 0.3
        fig = go.Figure()

        trace = go.Scatter(x=time, y=true_x, name="true x", marker_color="blue")
        fig.add_trace(trace)

        trace = go.Scatter(x=time, y=true_y, name="true y", marker_color="blue")
        fig.add_trace(trace)

        trace = go.Scatter(x=time_pred, y=x_forecast_mean, name="forecast x", marker_color="red", legendgroup="forecast_x")
        fig.add_trace(trace)

        trace = go.Scatter(x=time_pred, y=y_forecast_mean, name="forecast y", marker_color="red", legendgroup="forecast_y")
        fig.add_trace(trace)

        trace = go.Scatter(x=np.concatenate((time_pred, time_pred[::-1])),
                           y=np.concatenate((x_forecast_95ci_up,
                                             x_forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast_x"
                           )
        fig.add_trace(trace)

        trace = go.Scatter(x=np.concatenate((time_pred, time_pred[::-1])),
                           y=np.concatenate((y_forecast_95ci_up,
                                             y_forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast_y"
                           )
        fig.add_trace(trace)
        fig.update_layout(title=f"Forecasting Horizon: {h} samples, Log-Likelihood: {log_like}")
        fig.update_xaxes(title="Time (sec)")
        fig.update_yaxes(title="Acceleration")

    elif variable == "omega":
        true = sim_res["x"][8, :]
        forecast_mean = x_pred[8, 0, :]
        forecast_std = np.sqrt(P_pred[8, 8, :])
        forecast_95ci_down = forecast_mean - 1.96 * forecast_std
        forecast_95ci_up = forecast_mean + 1.96 * forecast_std

        color_pattern_filtered = "rgba(255,0,0,{:f})"
        cb_alpha = 0.3
        fig = go.Figure()

        trace = go.Scatter(x=time, y=true, name="true", marker_color="blue")
        fig.add_trace(trace)

        trace = go.Scatter(x=time_pred, y=forecast_mean, name="forecast", marker_color="red", legendgroup="forecast")
        fig.add_trace(trace)

        trace = go.Scatter(x=np.concatenate((time_pred, time_pred[::-1])),
                           y=np.concatenate((forecast_95ci_up,
                                             forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast"
                           )
        fig.add_trace(trace)

        fig.update_layout(title=f"Forecasting Horizon: {h} samples, Log-Likelihood: {log_like}")
        fig.update_xaxes(title="Time (sec)")
        fig.update_yaxes(title=r"\omega")

    elif variable == "HO":
        cosTheta_true = sim_res["x"][6, :]
        cosTheta_forecast_mean = x_pred[6, 0, :]
        cosTheta_forecast_std = np.sqrt(P_pred[6, 6, :])
        cosTheta_forecast_95ci_down = cosTheta_forecast_mean - 1.96 * cosTheta_forecast_std
        cosTheta_forecast_95ci_up = cosTheta_forecast_mean + 1.96 * cosTheta_forecast_std

        sinTheta_true = sim_res["x"][7, :]
        sinTheta_forecast_mean = x_pred[7, 0, :]
        sinTheta_forecast_std = np.sqrt(P_pred[7, 7, :])
        sinTheta_forecast_95ci_down = sinTheta_forecast_mean - 1.96 * sinTheta_forecast_std
        sinTheta_forecast_95ci_up = sinTheta_forecast_mean + 1.96 * sinTheta_forecast_std

        color_pattern_filtered = "rgba(255,0,0,{:f})"
        cb_alpha = 0.3
        fig = go.Figure()

        trace = go.Scatter(x=time, y=cosTheta_true, name="true cos(theta)", marker_color="blue")
        fig.add_trace(trace)

        trace = go.Scatter(x=time, y=sinTheta_true, name="true sin(theta)", marker_color="blue")
        fig.add_trace(trace)

        trace = go.Scatter(x=time, y=data[:, 2], name="measurement cos(theta)", marker_color="black")
        fig.add_trace(trace)

        trace = go.Scatter(x=time, y=data[:, 3], name="measurement sin(theta)", marker_color="black")
        fig.add_trace(trace)

        trace = go.Scatter(x=time_pred, y=cosTheta_forecast_mean, name="forecast cos(theta)", marker_color="red", legendgroup="forecast_x")
        fig.add_trace(trace)

        trace = go.Scatter(x=time_pred, y=sinTheta_forecast_mean, name="forecast sin(theta)", marker_color="red", legendgroup="forecast_y")
        fig.add_trace(trace)

        trace = go.Scatter(x=np.concatenate((time_pred, time_pred[::-1])),
                           y=np.concatenate((cosTheta_forecast_95ci_up,
                                             cosTheta_forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast_x"
                           )
        fig.add_trace(trace)

        trace = go.Scatter(x=np.concatenate((time_pred, time_pred[::-1])),
                           y=np.concatenate((sinTheta_forecast_95ci_up,
                                             sinTheta_forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast_y"
                           )
        fig.add_trace(trace)
        fig.update_layout(title=f"Forecasting Horizon: {h} samples, Log-Likelihood: {log_like}")
        fig.update_xaxes(title="Time (sec)")
        # fig.update_yaxes(title="Position")

    fig.write_image(fig_filename_pattern.format(forecasting_results_number, variable, "png"))
    fig.write_html(fig_filename_pattern.format(forecasting_results_number, variable, "html"))
    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
