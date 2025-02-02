import glob

import carball
import keras
import matplotlib.pyplot as plt
import numpy as np
from carball.analysis.analysis_manager import AnalysisManager
from carball.json_parser.game import Game
from matplotlib import patches

predict_dist = 15

boosts = np.array(
    [
        [-3072.0, -4096.0],
        [3072.0, -4096.0],
        [-3584.0, 0.0],
        [3072.0, 4096.0],
        [3584.0, 0.0],
        [-3072.0, 4096.0],
        [-1792.0, 4184.0],
        [1792.0, 4184.0],
        [-1792.0, -4184.0],
        [1792.0, -4184.0],
        [- 940.0, -3308.0],
        [940.0, -3308.0],
        [0.0, -2816.0],
        [-3584.0, -2484.0],
        [3584.0, -2484.0],
        [-1788.0, -2300.0],
        [1788.0, -2300.0],
        [-2048.0, -1036.0],
        [0.0, -1024.0],
        [2048.0, -1036.0],
        [0.0, -4240.0],
        [0.0, 4240.0],
        [-1024.0, 0.0],
        [1024.0, 0.0],
        [-2048.0, 1036.0],
        [0.0, 1024.0],
        [2048.0, 1036.0],
        [-1788.0, 2300.0],
        [1788.0, 2300.0],
        [-3584.0, 2484.0],
        [3584.0, 2484.0],
        [0.0, 2816.0],
        [- 940.0, 3310.0],
        [940.0, 3308.0]
    ]
)

boosts[:, 0] = (boosts[:, 0] + 4096) / 8192.0
boosts[:, 1] = (boosts[:, 1] + 6000) / 12000.0

boost_mapping = {
    40.0: 0,
    50.0: 1,
    160.0: 2,
    190.0: 3,
    300.0: 4,
    310.0: 5
}


def normalize(dataframe):
    dataframe.loc[:, (slice(None), 'pos_x')] = dataframe.loc[:, (slice(None), 'pos_x')].apply(lambda x: x + 4096)
    dataframe.loc[:, (slice(None), 'pos_x')] = dataframe.loc[:, (slice(None), 'pos_x')].apply(lambda x: x / 8192.0)
    dataframe.loc[:, (slice(None), 'pos_y')] = dataframe.loc[:, (slice(None), 'pos_y')].apply(lambda x: x + 6000)
    dataframe.loc[:, (slice(None), 'pos_y')] = dataframe.loc[:, (slice(None), 'pos_y')].apply(lambda x: x / 12000.0)
    dataframe.loc[:, (slice(None), 'pos_z')] = dataframe.loc[:, (slice(None), 'pos_z')].apply(lambda x: x / 2044.0)

    dataframe.loc[:, (slice(None), 'vel_x')] = dataframe.loc[:, (slice(None), 'vel_x')].apply(lambda x: x + 35000)
    dataframe.loc[:, (slice(None), 'vel_x')] = dataframe.loc[:, (slice(None), 'vel_x')].apply(lambda x: x / 70000.0)
    dataframe.loc[:, (slice(None), 'vel_y')] = dataframe.loc[:, (slice(None), 'vel_y')].apply(lambda x: x + 35000)
    dataframe.loc[:, (slice(None), 'vel_y')] = dataframe.loc[:, (slice(None), 'vel_y')].apply(lambda x: x / 70000.0)
    dataframe.loc[:, (slice(None), 'vel_z')] = dataframe.loc[:, (slice(None), 'vel_z')].apply(lambda x: x + 35000)
    dataframe.loc[:, (slice(None), 'vel_z')] = dataframe.loc[:, (slice(None), 'vel_z')].apply(lambda x: x / 70000.0)

    dataframe.loc[:, (slice(None), 'boost')] = dataframe.loc[:, (slice(None), 'boost')].apply(lambda x: x / 255.0)

    return dataframe


def prepareData(df):
    cols_to_keep = df.columns[df.columns.get_level_values(1).isin([
        "pos_x", "pos_y", "pos_z",
        "vel_x", "vel_y", "vel_z",
        "boost"
    ])]
    df = df.loc[:, cols_to_keep]
    return normalize(df)


def getMappedBoostId(pad_id):
    try:
        return boost_mapping.get(pad_id, -1)
    except KeyError:
        raise ValueError(f"Invalid replay pad_id: {pad_id}")


model = keras.models.load_model('model')

x8, x9, x10, y8, y9, y10, z8, z9, z10 = 0, 0, 0, 0, 0, 0, 0, 0, 0
old_x1, old_y1, old_z1, old_x8, old_x9, old_x10, old_y8, old_y9, old_y10 = 0, 0, 0, 0, 0, 0, 0, 0, 0

pad_list = []

# Precompute xlim, ylim (only set once)
# Create one figure and axes
fig, ax = plt.subplots(figsize=(6, 9), dpi=150)

# Set consistent limits (only once)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

first = True
orange = False
for filepath in glob.iglob('test_replays/*.replay'):
    print(f"Decompiling replay: {filepath}")
    _json = carball.decompile_replay(filepath)
    print("Replay decompiled successfully")

    game = Game()
    print("Initializing game...")
    game.initialize(loaded_json=_json)
    print("Game initialized successfully")

    analysis_manager = AnalysisManager(game)
    print("Creating analysis...")
    analysis_manager.create_analysis()
    print("Analysis created successfully")

    dataframe = analysis_manager.get_data_frame()

    boost_collect = dataframe.loc[:, (slice(None), 'boost_collect')]

    print("Converting boost pickups...")
    boostPads = np.ones((len(dataframe), 6))
    for rowNr in range(len(dataframe)):
        for padId in boost_collect.iloc[rowNr]:
            if np.isnan(padId):
                continue
            pickedBigPadId = getMappedBoostId(padId)
            for i in range(30 * 10):  # frame_tick is 0,03333 seconds -> 30 ticks/second
                if i + rowNr >= len(dataframe):
                    break
                boostPads[i + rowNr, pickedBigPadId] = 0

    orig_data = prepareData(dataframe)

    ball = np.nan_to_num(orig_data['ball'], nan=0.0)
    blue_team = orig_data.copy()
    orange_team = orig_data.copy()

    blue_team = blue_team.drop('ball', level=0, axis=1)
    orange_team = orange_team.drop('ball', level=0, axis=1)

    for player in game.players:
        if player.is_orange:
            blue_team = blue_team.drop([player.name], level=0, axis=1)
        else:
            orange_team = orange_team.drop([player.name], level=0, axis=1)

    blue_team = np.nan_to_num(blue_team, nan=0.0)
    orange_team = np.nan_to_num(orange_team, nan=0.0)

    player_index = 0
    orange = game.players[0].is_orange

    for i in range(len(game.players)):
        if 'miyjo' in game.players[i].name.lower():
            player_index = i
            orange = game.players[i].is_orange
            break

    print("Name: ", game.players[player_index].name)

    if orange:
        data = np.concatenate((np.concatenate((orange_team[:, (player_index * 7):(player_index * 7 + 7)],
                                               np.concatenate((orange_team[:, :(player_index * 7)],
                                                               orange_team[:, (player_index * 7 + 7):]), axis=1)),
                                              axis=1), blue_team), axis=1)
    else:
        data = np.concatenate((np.concatenate((blue_team[:, (player_index * 7):(player_index * 7 + 7)], np.concatenate(
            (blue_team[:, :(player_index * 7)], blue_team[:, (player_index * 7 + 7):]), axis=1)), axis=1), orange_team),
                              axis=1)

    data = np.concatenate((data, ball), axis=1)
    data = np.concatenate((data, boostPads), axis=1).astype(np.float32)

    # Loop through the data
    for i in range(len(data) - 1):
        x_vals = data[i, [0, 7, 14, 21, 28, 35, 42]]
        y_vals = data[i, [1, 8, 15, 22, 29, 36, 43]]
        z_vals = data[i, [2, 9, 16, 23, 30, 37, 44]]

        if i % 13 == 0:
            pred1 = model.predict(np.array([data[i]]))
            pred2 = model.predict(np.array([np.concatenate([pred1[0], data[i + predict_dist, 6:]])]))
            pred3 = model.predict(np.array([np.concatenate([pred2[0], data[i + 2 * predict_dist, 6:]])]))

            pred_x = [pred1[0][0], pred2[0][0], pred3[0][0]]
            pred_y = [pred1[0][1], pred2[0][1], pred3[0][1]]
            pred_z = [pred1[0][2], pred2[0][2], pred3[0][2]]

        if i % 26 == 0:
            old_pred_x, old_pred_y, old_pred_z = pred_x[:], pred_y[:], pred_z[:]
            old_x1, old_y1, old_z1 = x_vals[0], y_vals[0], z_vals[0]

        # Clear previous plot instead of recreating it
        ax.clear()

        ax.add_patch(
            patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
        )

        # Draw predicted points
        ax.scatter(pred_x, pred_y, s=np.array(pred_z) * 10 + 15, color='magenta', label="Predictions")
        ax.plot([x_vals[0]] + pred_x, [y_vals[0]] + pred_y, color='magenta')

        # Draw old predictions
        ax.scatter(old_pred_x, old_pred_y, s=np.array(old_pred_z) * 10 + 15, color='pink')
        ax.plot([old_x1] + old_pred_x, [old_y1] + old_pred_y, color='pink')

        # Draw Boost Pads
        ax.scatter(boosts[:, 0], boosts[:, 1], color='gold', s=10)

        # Highlight active boost pads
        active_boost_indices = np.where(boostPads[i] == 1)
        for idx in active_boost_indices:
            ax.scatter(boosts[idx, 0], boosts[idx, 1], s=30, color='gold')

        # Draw Players
        colors = ["green", "blue", "blue", "orangered", "orangered", "orangered", "gray"]
        ax.scatter(x_vals, y_vals, s=np.array(z_vals) * 10 + 40, c=colors)

        # Pause and refresh the plot
        plt.pause(0.01)
