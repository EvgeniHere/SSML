import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from scipy.interpolate import make_interp_spline

last_preds = []


def setup_plot():
    fig, ax = plt.subplots(figsize=(6, 9), dpi=150)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return fig, ax


def draw_field(ax):
    ax.clear()
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='black', facecolor='none'))


from matplotlib import cm


def draw_predictions(ax, current_all, preds_all):
    color_map = cm.get_cmap('Blues')

    x_path = [current_all[0]] + [p[0] for p in preds_all]
    y_path = [current_all[1]] + [p[1] for p in preds_all]
    z_vals = [p[2] for p in preds_all]

    t = np.linspace(0, 1, len(x_path))
    t_smooth = np.linspace(0, 1, 100)
    spline_x = make_interp_spline(t, x_path)(t_smooth)
    spline_y = make_interp_spline(t, y_path)(t_smooth)

    # Draw fading segments
    for i in range(len(spline_x) - 1):
        alpha = 1.0 - 0.7 * (i / (len(spline_x) - 1))
        ax.plot(spline_x[i:i + 2], spline_y[i:i + 2], color="blue", linestyle='dashed', alpha=alpha)

    z_norm = np.clip(np.array(z_vals) / np.max(z_vals), 0, 1)
    for i, (x, y, z) in enumerate(preds_all):
        alpha = 1 - 0.7 * (i / (len(preds_all) - 1))
        ax.scatter([x], [y], s=z * 100 + 15, color=color_map(z_norm[i]), alpha=alpha)


def draw_boostpads(ax, boostpad_positions, boost_state):
    ax.scatter(boostpad_positions[:, 0], boostpad_positions[:, 1], color='gold', s=10)
    active_indices = np.where(boost_state == 1)[0]
    ax.scatter(boostpad_positions[active_indices, 0], boostpad_positions[active_indices, 1], s=30, color='gold')


def draw_players(ax, x_vals, y_vals, z_vals):
    colors = ["green", "blue", "blue", "orangered", "orangered", "orangered", "gray"]
    ax.scatter(x_vals, y_vals, s=np.array(z_vals) * 100 + 40, c=colors)


def plot_frame(ax, data, frame_idx, model, boostPads, boostpad_positions, draw_interval=5):
    global last_preds

    x_vals = data[frame_idx, [x * 10 for x in range(7)]]
    y_vals = data[frame_idx, [x * 10 + 1 for x in range(7)]]
    z_vals = data[frame_idx, [x * 10 + 2 for x in range(7)]]

    if frame_idx % draw_interval == 0:
        prediction = model.predict(np.array([data[frame_idx]]), verbose=0)[0]
        last_preds = [
            (prediction[0], prediction[1], prediction[2]),
            (prediction[3], prediction[4], prediction[5]),
            (prediction[6], prediction[7], prediction[8])
        ]

    draw_field(ax)
    draw_predictions(
        ax,
        current_all=(x_vals[0], y_vals[0], z_vals[0]),
        preds_all=last_preds
    )
    draw_boostpads(ax, boostpad_positions, boostPads[frame_idx])
    draw_players(ax, x_vals, y_vals, z_vals)

    plt.pause(0.01)
