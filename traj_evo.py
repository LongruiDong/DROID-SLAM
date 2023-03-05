#!/usr/bin/env python3

########### ATTENTION!!! ###########
# Run `evo_config set plot_backend Agg`
#   to switch backend to Agg for plotting,
#   before running this python program.
# Or it will not be able to run on a non-GUI environment.

import numpy as np
import copy
from typing import Tuple

from evo import main_ape, main_rpe
from evo.core import sync, trajectory, geometry
from evo.core import metrics
from evo.tools import file_interface, plot
from evo.tools.plot import PlotMode
from evo.core.metrics import PoseRelation, Unit

import matplotlib.pyplot as plt
plt.switch_backend('Agg')


def trajectory_evaluation(
        ref_traj_type: str,
        ref_traj_path: str,
        est_traj_path: str,
        plot_save_path_prefix: str) -> Tuple[float, float, float, float, float]:

    if ref_traj_type == "tum":
        traj_ref = file_interface.read_tum_trajectory_file(ref_traj_path)
    elif ref_traj_type == "euroc":
        traj_ref = file_interface.read_euroc_csv_trajectory(ref_traj_path)
    else:
        raise Exception(f"invalid ref_traj_type: {ref_traj_type}")

    traj_est = file_interface.read_tum_trajectory_file(est_traj_path)
    traj_ref, traj_est = sync.associate_trajectories(
        traj_ref, traj_est, max_diff=0.02)

    traj_est.align(traj_ref, False, False)

    # compute ape, are, rpe, rre stats

    ape_metric = metrics.APE(PoseRelation.translation_part)
    data_ape = (traj_ref, traj_est)
    ape_metric.process_data(data_ape)
    ape_result = ape_metric.get_result()

    are_metric = metrics.APE(PoseRelation.rotation_angle_deg)
    data_are = (traj_ref, traj_est)
    are_metric.process_data(data_are)
    are_result = are_metric.get_result()

    rpe_metric = metrics.RPE(
        PoseRelation.translation_part, delta=1.0, delta_unit=Unit.meters, all_pairs=True)
    data_rpe = (traj_ref, traj_est)
    rpe_metric.process_data(data_rpe)
    rpe_result = rpe_metric.get_result()

    rre_metric = metrics.RPE(
        PoseRelation.rotation_angle_deg, delta=1.0, delta_unit=Unit.meters, all_pairs=True)
    data_rre = (traj_ref, traj_est)
    rre_metric.process_data(data_rre)
    rre_result = rre_metric.get_result()

    ape_rmse = ape_result.stats['rmse']
    are_rmse = are_result.stats['rmse']
    rpe_rmse = rpe_result.stats['rmse']
    rre_rmse = rre_result.stats['rmse']
    _, _, scale = geometry.umeyama_alignment(
        traj_est.positions_xyz.T, traj_ref.positions_xyz.T, True)
    
    traj_est_sim = traj_est
    traj_est_sim.align(traj_ref, True, False)
    ape_metric_sim = metrics.APE(PoseRelation.translation_part)
    data_ape_sim = (traj_ref, traj_est_sim)
    ape_metric_sim.process_data(data_ape_sim)
    ape_result_sim = ape_metric_sim.get_result()

    ape_sim_rmse = ape_result_sim.stats['rmse']

    print(f"ape_rmse: {ape_rmse}")
    print(f"ape_rmse_sim3: {ape_sim_rmse}")
    print(f"are_rmse: {are_rmse}")
    print(f"rpe_rmse: {rpe_rmse}")
    print(f"rre_rmse: {rre_rmse}")
    print(f"scale: {scale}")

    # plot ape, are, rpe, rre with trajectory
    # ape plot
    fig = plt.figure(num="trajectory", figsize=(16, 16))
    ax = plot.prepare_axis(fig, plot_mode=PlotMode.xy, subplot_arg=221)
    plot.traj(ax, plot_mode=PlotMode.xy, traj=traj_ref, style="--", alpha=0.5)
    plot.traj_colormap(
        ax, traj_est, ape_result.np_arrays["error_array"],
        PlotMode.xy, min_map=ape_result.stats["min"], max_map=ape_result.stats["max"]
    )
    plt.title(
        f"ape(rmse: {ape_rmse:.3f}m, median: {ape_result.stats['median']:.3f}m)")
    plt.grid(True)
    # are plot
    ax = plot.prepare_axis(fig, plot_mode=PlotMode.xy, subplot_arg=222)
    plot.traj(ax, plot_mode=PlotMode.xy, traj=traj_ref, style="--", alpha=0.5)
    plot.traj_colormap(
        ax, traj_est, are_result.np_arrays["error_array"],
        PlotMode.xy, min_map=are_result.stats["min"], max_map=are_result.stats["median"] +
        are_result.stats["std"]
    )
    plt.title(
        f"are(rmse: {are_rmse:.3f}°, median: {are_result.stats['median']:.3f}°)")
    plt.grid(True)
    # rpe plot
    rpe_traj_ref = copy.deepcopy(traj_ref)
    rpe_traj_est = copy.deepcopy(traj_est)
    rpe_traj_ref.reduce_to_ids(rpe_metric.delta_ids)
    rpe_traj_est.reduce_to_ids(rpe_metric.delta_ids)

    ax = plot.prepare_axis(fig, plot_mode=PlotMode.xy, subplot_arg=223)
    plot.traj(ax, plot_mode=PlotMode.xy,
              traj=rpe_traj_ref, style="--", alpha=0.5)
    plot.traj_colormap(
        ax, rpe_traj_est, rpe_result.np_arrays["error_array"],
        PlotMode.xy, min_map=rpe_result.stats["min"], max_map=rpe_result.stats["mean"] +
        2 * rpe_result.stats["std"]
    )
    plt.title(
        f"rpe(rmse: {rpe_rmse:.3f}m, median: {rpe_result.stats['median']:.3f}m)")
    plt.grid(True)
    # rre plot
    rre_traj_ref = copy.deepcopy(traj_ref)
    rre_traj_est = copy.deepcopy(traj_est)
    rre_traj_ref.reduce_to_ids(rre_metric.delta_ids)
    rre_traj_est.reduce_to_ids(rre_metric.delta_ids)
    # plot rre result
    ax = plot.prepare_axis(fig, plot_mode=PlotMode.xy, subplot_arg=224)
    plot.traj(ax, plot_mode=PlotMode.xy,
              traj=rre_traj_ref, style="--", alpha=0.5)
    plot.traj_colormap(
        ax, rre_traj_est, rre_result.np_arrays["error_array"],
        PlotMode.xy, min_map=rre_result.stats["min"], max_map=rre_result.stats["median"] +
        rre_result.stats["std"]
    )
    plt.title(
        f"rre(rmse: {rre_rmse:.3f}°, rre: {rre_result.stats['median']:.3f}°)")
    plt.grid(True)

    plt.savefig(f"{plot_save_path_prefix}-ape-are-rpe-rre-plot.png")
    # always close fig after savefig, or the memory will not be released
    plt.close(fig)

    # plot ape, are, rpe, rre histogram
    ape_step = 0.01  # 0.01m
    ape_error_array = np.array(ape_result.np_arrays["error_array"])
    print(f"ape_error_array.len: {len(ape_error_array)}")
    ape_mean, ape_min, ape_max = ape_result.stats["mean"], ape_result.stats["min"], ape_result.stats["max"]
    ape_error_p_d = np.zeros(int(ape_max / ape_step) + 2, dtype=float)
    ape_error_accum_p = np.zeros(int(ape_max / ape_step) + 2, dtype=float)
    sum_size = len(ape_error_array)
    for i in range(1, ape_error_p_d.shape[0]):
        min_error = (i - 1) * ape_step
        max_error = i * ape_step
        ape_error_p_d[i] = float(np.sum((ape_error_array >= min_error) & (
            ape_error_array < max_error)) / sum_size)
        ape_error_accum_p[i] = ape_error_accum_p[i - 1]
        ape_error_accum_p[i] += ape_error_p_d[i]

    are_step = 0.2  # 0.2°
    are_error_array = np.array(are_result.np_arrays["error_array"])
    print(f"are_error_array.len: {len(are_error_array)}")
    are_mean, are_min, are_max = are_result.stats["mean"], are_result.stats["min"], are_result.stats["max"]
    are_error_p_d = np.zeros(int(are_max / are_step) + 2, dtype=float)
    are_error_accum_p = np.zeros(int(are_max / are_step) + 2, dtype=float)
    sum_size = len(are_error_array)
    for i in range(1, are_error_p_d.shape[0]):
        min_error = (i - 1) * are_step
        max_error = i * are_step
        are_error_p_d[i] = float(np.sum((are_error_array >= min_error) & (
            are_error_array < max_error)) / sum_size)
        are_error_accum_p[i] = are_error_accum_p[i - 1]
        are_error_accum_p[i] += are_error_p_d[i]

    rpe_step = 0.01  # 0.01m
    rpe_error_array = np.array(rpe_result.np_arrays["error_array"])
    print(f"rpe_error_array.len: {len(rpe_error_array)}")
    rpe_mean, rpe_min, rpe_max = rpe_result.stats["mean"], rpe_result.stats["min"], rpe_result.stats["max"]
    rpe_error_p_d = np.zeros(int(rpe_max / rpe_step) + 2, dtype=float)
    rpe_error_accum_p = np.zeros(int(rpe_max / rpe_step) + 2, dtype=float)
    sum_size = len(rpe_error_array)
    for i in range(1, rpe_error_p_d.shape[0]):
        min_error = (i - 1) * rpe_step
        max_error = i * rpe_step
        rpe_error_p_d[i] = float(np.sum((rpe_error_array >= min_error) & (
            rpe_error_array < max_error)) / sum_size)
        rpe_error_accum_p[i] = rpe_error_accum_p[i - 1]
        rpe_error_accum_p[i] += rpe_error_p_d[i]

    rre_step = 0.2  # 0.2°
    rre_error_array = np.array(rre_result.np_arrays["error_array"])
    print(f"rre_error_array.len: {len(rre_error_array)}")
    rre_mean, rre_min, rre_max = rre_result.stats["mean"], rre_result.stats["min"], rre_result.stats["max"]
    rre_error_p_d = np.zeros(int(rre_max / rre_step) + 2, dtype=float)
    rre_error_accum_p = np.zeros(int(rre_max / rre_step) + 2, dtype=float)
    sum_size = len(rre_error_array)
    for i in range(1, rre_error_p_d.shape[0]):
        min_error = (i - 1) * rre_step
        max_error = i * rre_step
        rre_error_p_d[i] = float(np.sum((rre_error_array >= min_error) & (
            rre_error_array < max_error)) / sum_size)
        rre_error_accum_p[i] = rre_error_accum_p[i - 1]
        rre_error_accum_p[i] += rre_error_p_d[i]

    # plot and save fig
    plt.figure(num="density", figsize=(16, 16))
    # ape
    ax = plt.subplot2grid((2, 2), (0, 0))
    ax.set_xscale('log')
    plt.plot(np.arange(ape_error_p_d.shape[0]) * ape_step,
             ape_error_p_d, label="ape_density")
    plt.plot(np.arange(ape_error_p_d.shape[0]) * ape_step,
             ape_error_accum_p, label="ape_cum_density")
    plt.vlines(ape_mean, 0, 1.0, colors='r', label='mean_error')
    plt.ylabel("p")
    plt.xlabel("ape(m)")
    plt.legend()
    plt.title("ape_density(mean: %.3fm, min: %.3fm, max: %.3fm)" %
              (ape_mean, ape_min, ape_max))
    plt.grid(True)
    # are
    ax = plt.subplot2grid((2, 2), (0, 1))
    plt.plot(np.arange(are_error_p_d.shape[0]) * are_step,
             are_error_p_d, label="are_density")
    plt.plot(np.arange(are_error_p_d.shape[0]) * are_step,
             are_error_accum_p, label="are_cum_density")
    plt.vlines(are_mean, 0, 1.0, colors='r', label='mean_error')
    plt.ylabel("p")
    plt.xlabel("are(°)")
    plt.legend()
    plt.title("are_density(mean: %.3f°, min: %.3f°, max: %.3f°)" % (
        are_mean, are_min, are_max))
    plt.grid(True)
    # rpe
    ax = plt.subplot2grid((2, 2), (1, 0))
    ax.set_xscale('log')
    plt.plot(np.arange(rpe_error_p_d.shape[0]) * rpe_step,
             rpe_error_p_d, label="rpe_density")
    plt.plot(np.arange(rpe_error_p_d.shape[0]) * rpe_step,
             rpe_error_accum_p, label="rpe_cum_density")
    plt.vlines(rpe_mean, 0, 1.0, colors='r', label='mean_error')
    plt.ylabel("p")
    plt.xlabel("rpe(m)")
    plt.legend()
    plt.title("rpe_density(mean: %.3fm, min: %.3fm, max: %.3fm)" %
              (rpe_mean, rpe_min, rpe_max))
    plt.grid(True)
    # rre
    ax = plt.subplot2grid((2, 2), (1, 1))
    # ax.set_xscale('log')
    plt.plot(np.arange(rre_error_p_d.shape[0]) * rre_step,
             rre_error_p_d, label="rre_density")
    plt.plot(np.arange(rre_error_p_d.shape[0]) * rre_step,
             rre_error_accum_p, label="rre_cum_density")
    plt.vlines(rre_mean, 0, 1.0, colors='r', label='mean_error')
    plt.ylabel("p")
    plt.xlabel("rre(°)")
    plt.legend()
    plt.title("rre_density(mean: %.3f°, min: %.3f°, max: %.3f°)" % (
        rre_mean, rre_min, rre_max))
    plt.grid(True)

    plt.savefig(f"{plot_save_path_prefix}-ape-are-rpe-rre-density.png")
    plt.close(fig)

    return (ape_rmse,ape_sim_rmse, are_rmse, rpe_rmse, rre_rmse, scale)


if __name__ == "__main__":
    import sys

    evo_result = trajectory_evaluation(
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])