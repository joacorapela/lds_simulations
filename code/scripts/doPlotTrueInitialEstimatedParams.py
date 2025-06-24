import sys
import argparse
import configparser
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--true_params_filename", type=str,
                        default="../../metadata/00000021_simulation.ini",
                        help="true parameters filename")
    parser.add_argument("--initial_params_filename", type=str,
                        default="../../metadata/00000026_estimation.ini",
                        help="initial parameters filename")
    parser.add_argument("--estimated_params_filename", type=str,
                        default="../../metadata/00000012_filtering.ini",
                        help="estimated parameters filename")
    parser.add_argument("--fig_filename_pattern", type=str,
                        default="../../figures/true_initial_estimated_params_1_{:s}.{:s}",
                        help="figure filename pattern")
    args = parser.parse_args()

    true_params_filename = args.true_params_filename
    initial_params_filename = args.initial_params_filename
    estimated_params_filename = args.estimated_params_filename
    fig_filename_pattern = args.fig_filename_pattern

    true_meta = configparser.ConfigParser()
    true_meta.read(true_params_filename)
    true_pos_x_m0 = float(true_meta["initial_state"]["pos_x_mean"])
    true_pos_y_m0 = float(true_meta["initial_state"]["pos_y_mean"])
    true_vel_x_m0 = float(true_meta["initial_state"]["vel_x_mean"])
    true_vel_y_m0 = float(true_meta["initial_state"]["vel_y_mean"])
    true_acc_x_m0 = float(true_meta["initial_state"]["acc_x_mean"])
    true_acc_y_m0 = float(true_meta["initial_state"]["acc_y_mean"])
    true_cos_theta_m0 = float(true_meta["initial_state"]["cos_theta_mean"])
    true_sin_theta_m0 = float(true_meta["initial_state"]["sin_theta_mean"])
    true_omega_m0 = float(true_meta["initial_state"]["omega_mean"])
    true_m0 = [true_pos_x_m0, true_pos_y_m0, true_vel_x_m0, true_vel_y_m0, true_acc_x_m0, true_acc_y_m0, true_cos_theta_m0, true_sin_theta_m0]

    true_pos_x_V0_std = float(true_meta["initial_state"]["pos_x_std"])
    true_pos_y_V0_std = float(true_meta["initial_state"]["pos_y_std"])
    true_vel_x_V0_std = float(true_meta["initial_state"]["vel_x_std"])
    true_vel_y_V0_std = float(true_meta["initial_state"]["vel_y_std"])
    true_acc_x_V0_std = float(true_meta["initial_state"]["acc_x_std"])
    true_acc_y_V0_std = float(true_meta["initial_state"]["acc_y_std"])
    true_cos_theta_V0_std = float(true_meta["initial_state"]["cos_theta_std"])
    true_sin_theta_V0_std = float(true_meta["initial_state"]["sin_theta_std"])
    true_omega_V0_std = float(true_meta["initial_state"]["omega_std"])
    true_sqrt_diag_V0 = [true_pos_x_V0_std, true_pos_y_V0_std, true_vel_x_V0_std, true_vel_y_V0_std, true_acc_x_V0_std, true_acc_y_V0_std, true_cos_theta_V0_std, true_sin_theta_V0_std]

    true_sigma_a = float(true_meta["state_cov"]["sigma_a"])
    true_cos_theta_Q_std = float(true_meta["state_cov"]["cos_theta_std"])
    true_sin_theta_Q_std = float(true_meta["state_cov"]["sin_theta_std"])
    true_omega_std = float(true_meta["state_cov"]["omega_std"])
    true_pos_x_std = float(true_meta["measurements_cov"]["pos_x_std"])
    true_pos_y_std = float(true_meta["measurements_cov"]["pos_y_std"])
    true_cos_theta_R_std = float(true_meta["measurements_cov"]["cos_theta_std"])
    true_sin_theta_R_std = float(true_meta["measurements_cov"]["sin_theta_std"])
    true_alpha = float(true_meta["other"]["alpha"])
    true_model_params = [true_sigma_a, true_pos_x_std, true_pos_y_std, true_cos_theta_Q_std, true_sin_theta_Q_std, true_omega_std, true_cos_theta_R_std, true_sin_theta_R_std, true_alpha]

    initial_meta = configparser.ConfigParser()
    initial_meta.read(initial_params_filename)
    initial_pos_x_m0 = float(initial_meta["initial_state"]["pos_x_mean"])
    initial_pos_y_m0 = float(initial_meta["initial_state"]["pos_y_mean"])
    initial_vel_x_m0 = float(initial_meta["initial_state"]["vel_x_mean"])
    initial_vel_y_m0 = float(initial_meta["initial_state"]["vel_y_mean"])
    initial_acc_x_m0 = float(initial_meta["initial_state"]["acc_x_mean"])
    initial_acc_y_m0 = float(initial_meta["initial_state"]["acc_y_mean"])
    initial_cos_theta_m0 = float(initial_meta["initial_state"]["cos_theta_mean"])
    initial_sin_theta_m0 = float(initial_meta["initial_state"]["sin_theta_mean"])
    initial_omega_m0 = float(initial_meta["initial_state"]["omega_mean"])
    initial_m0 = [initial_pos_x_m0, initial_pos_y_m0, initial_vel_x_m0, initial_vel_y_m0, initial_acc_x_m0, initial_acc_y_m0, initial_cos_theta_m0, initial_sin_theta_m0]

    initial_pos_x_V0_std = float(initial_meta["initial_state"]["pos_x_std"])
    initial_pos_y_V0_std = float(initial_meta["initial_state"]["pos_y_std"])
    initial_vel_x_V0_std = float(initial_meta["initial_state"]["vel_x_std"])
    initial_vel_y_V0_std = float(initial_meta["initial_state"]["vel_y_std"])
    initial_acc_x_V0_std = float(initial_meta["initial_state"]["acc_x_std"])
    initial_acc_y_V0_std = float(initial_meta["initial_state"]["acc_y_std"])
    initial_cos_theta_V0_std = float(initial_meta["initial_state"]["cos_theta_std"])
    initial_sin_theta_V0_std = float(initial_meta["initial_state"]["sin_theta_std"])
    initial_omega_V0_std = float(initial_meta["initial_state"]["omega_std"])
    initial_sqrt_diag_V0 = [initial_pos_x_V0_std, initial_pos_y_V0_std, initial_vel_x_V0_std, initial_vel_y_V0_std, initial_acc_x_V0_std, initial_acc_y_V0_std, initial_cos_theta_V0_std, initial_sin_theta_V0_std]

    initial_sigma_a = float(initial_meta["state_cov"]["sigma_a"])
    initial_cos_theta_Q_std = float(initial_meta["state_cov"]["cos_theta_std"])
    initial_sin_theta_Q_std = float(initial_meta["state_cov"]["sin_theta_std"])
    initial_omega_std = float(initial_meta["state_cov"]["omega_std"])
    initial_pos_x_std = float(initial_meta["measurements_cov"]["pos_x_std"])
    initial_pos_y_std = float(initial_meta["measurements_cov"]["pos_y_std"])
    initial_cos_theta_R_std = float(initial_meta["measurements_cov"]["cos_theta_std"])
    initial_sin_theta_R_std = float(initial_meta["measurements_cov"]["sin_theta_std"])
    initial_alpha = float(initial_meta["other"]["alpha"])
    initial_model_params = [initial_sigma_a, initial_pos_x_std, initial_pos_y_std, initial_cos_theta_Q_std, initial_sin_theta_Q_std, initial_omega_std, initial_cos_theta_R_std, initial_sin_theta_R_std, initial_alpha]

    est_meta = configparser.ConfigParser()
    est_meta.read(estimated_params_filename)
    est_pos_x_m0 = float(est_meta["initial_state"]["pos_x_mean"])
    est_pos_y_m0 = float(est_meta["initial_state"]["pos_y_mean"])
    est_vel_x_m0 = float(est_meta["initial_state"]["vel_x_mean"])
    est_vel_y_m0 = float(est_meta["initial_state"]["vel_y_mean"])
    est_acc_x_m0 = float(est_meta["initial_state"]["acc_x_mean"])
    est_acc_y_m0 = float(est_meta["initial_state"]["acc_y_mean"])
    est_cos_theta_m0 = float(est_meta["initial_state"]["cos_theta_mean"])
    est_sin_theta_m0 = float(est_meta["initial_state"]["sin_theta_mean"])
    est_omega_m0 = float(est_meta["initial_state"]["omega_mean"])
    estimated_m0 = [est_pos_x_m0, est_pos_y_m0, est_vel_x_m0, est_vel_y_m0, est_acc_x_m0, est_acc_y_m0, est_cos_theta_m0, est_sin_theta_m0]

    est_pos_x_V0_std = float(est_meta["initial_state"]["pos_x_std"])
    est_pos_y_V0_std = float(est_meta["initial_state"]["pos_y_std"])
    est_vel_x_V0_std = float(est_meta["initial_state"]["vel_x_std"])
    est_vel_y_V0_std = float(est_meta["initial_state"]["vel_y_std"])
    est_acc_x_V0_std = float(est_meta["initial_state"]["acc_x_std"])
    est_acc_y_V0_std = float(est_meta["initial_state"]["acc_y_std"])
    est_cos_theta_V0_std = float(est_meta["initial_state"]["cos_theta_std"])
    est_sin_theta_V0_std = float(est_meta["initial_state"]["sin_theta_std"])
    est_omega_V0_std = float(est_meta["initial_state"]["omega_std"])
    estimated_sqrt_diag_V0 = [est_pos_x_V0_std, est_pos_y_V0_std, est_vel_x_V0_std, est_vel_y_V0_std, est_acc_x_V0_std, est_acc_y_V0_std, est_cos_theta_V0_std, est_sin_theta_V0_std]

    est_sigma_a = float(est_meta["state_cov"]["sigma_a"])
    est_cos_theta_Q_std = float(est_meta["state_cov"]["cos_theta_std"])
    est_sin_theta_Q_std = float(est_meta["state_cov"]["sin_theta_std"])
    est_omega_std = float(est_meta["state_cov"]["omega_std"])
    est_pos_x_std = float(est_meta["measurements_cov"]["pos_x_std"])
    est_pos_y_std = float(est_meta["measurements_cov"]["pos_y_std"])
    est_cos_theta_R_std = float(est_meta["measurements_cov"]["cos_theta_std"])
    est_sin_theta_R_std = float(est_meta["measurements_cov"]["sin_theta_std"])
    est_alpha = float(est_meta["other"]["alpha"])
    estimated_model_params = [est_sigma_a, est_pos_x_std, est_pos_y_std, est_cos_theta_Q_std, est_sin_theta_Q_std, est_omega_std, est_cos_theta_R_std, est_sin_theta_R_std, est_alpha]

    param_names = ["sigma_a", "pos_x_std", "pos_y_std", "cos_theta_Q_std", "sin_theta_Q_std", "omega_Q_std", "cos_theta_R_std", "sin_theta_R_std", "alpha"]
    bar_true = go.Bar(name="true", x=param_names, y=true_model_params)
    bar_initial = go.Bar(name="initial", x=param_names, y=initial_model_params)
    bar_estimated = go.Bar(name="estimated", x=param_names, y=estimated_model_params)
    fig = go.Figure([bar_true, bar_initial, bar_estimated])
    fig.update_yaxes(title="Parameter Value")
    fig.write_image(fig_filename_pattern.format("modelParams", "png"))
    fig.write_html(fig_filename_pattern.format("modelParams", "html"))

    fig.show()

    param_names = ["pos_x", "pos_y", "vel_x", "vel_y", "acc_x", "acc_y", "cos_theta", "sin_theta", "omega"]

    fig = go.Figure()
    trace = go.Scatter(x=param_names, y=true_m0, name="true")
    fig.add_trace(trace)
    trace = go.Scatter(x=param_names, y=initial_m0, name="initial")
    fig.add_trace(trace)
    trace = go.Scatter(x=param_names, y=estimated_m0, name="estimated")
    fig.add_trace(trace)
    fig.update_yaxes(title=r"m_0")
    fig.write_image(fig_filename_pattern.format("m0", "png"))
    fig.write_html(fig_filename_pattern.format("m0", "html"))

    fig.show()

    fig = go.Figure()
    trace = go.Scatter(x=param_names, y=true_sqrt_diag_V0, name="true")
    fig.add_trace(trace)
    trace = go.Scatter(x=param_names, y=initial_sqrt_diag_V0, name="initial")
    fig.add_trace(trace)
    trace = go.Scatter(x=param_names, y=estimated_sqrt_diag_V0, name="estimated")
    fig.add_trace(trace)
    fig.update_yaxes(title=r"sqrt diag V_0")
    fig.write_image(fig_filename_pattern.format("sqrt_diag_V0", "png"))
    fig.write_html(fig_filename_pattern.format("sqrt_diag_V0", "html"))

    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
