import carball
from carball.json_parser.game import Game
from carball.analysis.analysis_manager import AnalysisManager
import glob
import numpy as np
import keras
import matplotlib.pyplot as plt

predict_dist = 15

boosts = [[0.0, -4240.0],
          [-1792.0, -4184.0],
          [1792.0, -4184.0],
          [-3072.0, -4096.0],
          [3072.0, -4096.0],
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
          [-3584.0, 0.0],
          [-1024.0, 0.0],
          [1024.0, 0.0],
          [3584.0, 0.0],
          [-2048.0, 1036.0],
          [0.0, 1024.0],
          [2048.0, 1036.0],
          [-1788.0, 2300.0],
          [1788.0, 2300.0],
          [-3584.0, 2484.0],
          [3584.0, 2484.0],
          [0.0, 2816.0],
          [- 940.0, 3310.0],
          [940.0, 3308.0],
          [-3072.0, 4096.0],
          [3072.0, 4096.0],
          [-1792.0, 4184.0],
          [1792.0, 4184.0],
          [0.0, 4240.0]]


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

    for x in boosts:
        x[0] = (x[0] + 4069) / 8192
        x[1] = (x[1] + 6000) / 12000

    return dataframe


def getTraindata(dataframe):
    dataframe = dataframe.drop(['ping',
                                'ball_cam',
                                'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
                                'throttle',
                                'steer',
                                'rot_x', 'rot_y', 'rot_z',
                                'handbrake',
                                'jump_active',
                                'double_jump_active',
                                'boost_active',
                                'dodge_active',
                                'hit_team_no',
                                'boost_collect',
                                'game',
                                'delta',
                                'game_time',
                                'seconds_remaining',
                                'replicated_seconds_remaining',
                                'ball_has_been_hit',
                                'goal_number',
                                'time'], level=1, axis=1)
    dataframe = normalize(dataframe)
    print(dataframe.head().to_string())
    return dataframe


model = keras.models.load_model('model')

x8, x9, x10, y8, y9, y10, z8, z9, z10 = 0, 0, 0, 0, 0, 0, 0, 0, 0
old_x1, old_y1, old_z1, old_x8, old_x9, old_x10, old_y8, old_y9, old_y10 = 0, 0, 0, 0, 0, 0, 0, 0, 0

boosts = np.nan_to_num(boosts)

pad_list = []

first = True
orange = False
for filepath in glob.iglob('test_replays/*.replay'):
    _json = carball.decompile_replay(filepath)
    game = Game()
    game.initialize(loaded_json=_json)

    analysis_manager = AnalysisManager(game)
    analysis_manager.create_analysis()

    dataframe = analysis_manager.get_data_frame()

    boost_data = dataframe.loc[:, (slice(None), 'boost_collect')]
    boost_pads = np.zeros((len(dataframe), 6))
    for i in range(len(dataframe)):
        game_time = dataframe.iloc[i]['game', 'time']
        if all(np.isnan(x) for x in boost_data.iloc[i]):
            continue

        for y in boost_data.iloc[i]:
            if np.isnan(y):
                continue
            switch_boost_num = -1
            if y == 40.0:
                switch_boost_num = 0
            elif y == 50.0:
                switch_boost_num = 1
            elif y == 160.0:
                switch_boost_num = 2
            elif y == 190.0:
                switch_boost_num = 3
            elif y == 300.0:
                switch_boost_num = 4
            elif y == 310.0:
                switch_boost_num = 5
            j = i
            while j < len(dataframe) and game_time + 10.0 >= dataframe.iloc[j]['game', 'time']:
                boost_pads[j, switch_boost_num] = 1
                j += 1

    orig_data = getTraindata(dataframe)

    blue_team = orig_data.copy()
    orange_team = orig_data.copy()

    for player in game.players:
        if player.is_orange:
            blue_team = blue_team.drop([player.name], level=0, axis=1)
        else:
            orange_team = orange_team.drop([player.name], level=0, axis=1)

    blue_team = np.nan_to_num(blue_team.drop('ball', level=0, axis=1), nan=0.0)
    orange_team = np.nan_to_num(orange_team.drop('ball', level=0, axis=1), nan=0.0)
    ball = np.nan_to_num(orig_data['ball'], nan=0.0)

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
        data[:, [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 21, 22, 24, 25, 28, 29, 31, 32, 35, 36, 38, 39]] *= -1
        data[:, [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 21, 22, 24, 25, 28, 29, 31, 32, 35, 36, 38, 39]] += 1
        ball[:, [0, 1, 3, 4]] *= -1
        ball[:, [0, 1, 3, 4]] += 1
    else:
        data = np.concatenate((np.concatenate((blue_team[:, (player_index * 7):(player_index * 7 + 7)], np.concatenate(
            (blue_team[:, :(player_index * 7)], blue_team[:, (player_index * 7 + 7):]), axis=1)), axis=1), orange_team),
                              axis=1)

    data = np.concatenate((data, ball), axis=1)
    data = np.concatenate((data, boost_pads), axis=1).astype(np.float32)

    for i in range(len(data) - 1):
        x1, y1, z1 = data[i][0], data[i][1], data[i][2]
        x2, y2, z2 = data[i][7], data[i][8], data[i][9]
        x3, y3, z3 = data[i][14], data[i][15], data[i][16]
        x4, y4, z4 = data[i][21], data[i][22], data[i][23]
        x5, y5, z5 = data[i][28], data[i][29], data[i][30]
        x6, y6, z6 = data[i][35], data[i][36], data[i][37]
        x7, y7, z7 = data[i][42], data[i][43], data[i][44]

        if i % 13 == 0:
            pred1 = model.predict(np.array([data[i]]))
            pred2 = model.predict(np.array([np.concatenate([pred1[0], data[i + predict_dist, 6:]])]))
            pred3 = model.predict(np.array([np.concatenate([pred2[0], data[i + 2 * predict_dist, 6:]])]))

            x8, y8, z8 = pred1[0][0], pred1[0][1], pred1[0][2]
            x9, y9, z9 = pred2[0][0], pred2[0][1], pred2[0][2]
            x10, y10, z10 = pred3[0][0], pred3[0][1], pred3[0][2]

        if i % 26 == 0:
            old_x8 = x8
            old_x9 = x9
            old_x10 = x10
            old_y8 = y8
            old_y9 = y9
            old_y10 = y10
            old_z8 = z8
            old_z9 = z9
            old_z10 = z10
            old_x1 = x1
            old_y1 = y1
            old_z1 = z1

        plt.plot(x8, y8, marker="o", markersize=z8 * 10 + 5, markeredgecolor="magenta", markerfacecolor="magenta")
        plt.plot(x9, y9, marker="o", markersize=z9 * 10 + 5, markeredgecolor="magenta", markerfacecolor="magenta")
        plt.plot(x10, y10, marker="o", markersize=z10 * 10 + 5, markeredgecolor="magenta", markerfacecolor="magenta")
        plt.plot([x1, x8, x9, x10], [y1, y8, y9, y10], color='magenta')

        plt.plot(old_x8, old_y8, marker="o", markersize=old_z8 * 10 + 5, markeredgecolor="pink", markerfacecolor="pink")
        plt.plot(old_x9, old_y9, marker="o", markersize=old_z9 * 10 + 5, markeredgecolor="pink", markerfacecolor="pink")
        plt.plot(old_x10, old_y10, marker="o", markersize=old_z10 * 10 + 5, markeredgecolor="pink", markerfacecolor="pink")
        plt.plot([old_x1, old_x8, old_x9, old_x10], [old_y1, old_y8, old_y9, old_y10], color='pink')

        plt.scatter(boosts[:, 0], boosts[:, 1], color='gold', s=3)

        if boost_pads[i, 0] == 0:
            plt.plot(boosts[3, 0], boosts[3, 1], marker="o", markersize=10, markeredgecolor="gold", markerfacecolor="gold")
        if boost_pads[i, 1] == 0:
            plt.plot(boosts[4, 0], boosts[4, 1], marker="o", markersize=10, markeredgecolor="gold", markerfacecolor="gold")
        if boost_pads[i, 2] == 0:
            plt.plot(boosts[15, 0], boosts[15, 1], marker="o", markersize=10, markeredgecolor="gold", markerfacecolor="gold")
        if boost_pads[i, 3] == 0:
            plt.plot(boosts[18, 0], boosts[18, 1], marker="o", markersize=10, markeredgecolor="gold", markerfacecolor="gold")
        if boost_pads[i, 4] == 0:
            plt.plot(boosts[29, 0], boosts[29, 1], marker="o", markersize=10, markeredgecolor="gold", markerfacecolor="gold")
        if boost_pads[i, 5] == 0:
            plt.plot(boosts[30, 0], boosts[30, 1], marker="o", markersize=10, markeredgecolor="gold", markerfacecolor="gold")

        plt.plot(x1, y1, marker="o", markersize=z1 * 10 + 6, markeredgecolor="green", markerfacecolor="green")
        plt.plot(x2, y2, marker="o", markersize=z2 * 10 + 6, markeredgecolor="blue", markerfacecolor="blue")
        plt.plot(x3, y3, marker="o", markersize=z3 * 10 + 6, markeredgecolor="blue", markerfacecolor="blue")
        plt.plot(x4, y4, marker="o", markersize=z4 * 10 + 6, markeredgecolor="orangered", markerfacecolor="orangered")
        plt.plot(x5, y5, marker="o", markersize=z5 * 10 + 6, markeredgecolor="orangered", markerfacecolor="orangered")
        plt.plot(x6, y6, marker="o", markersize=z6 * 10 + 6, markeredgecolor="orangered", markerfacecolor="orangered")
        plt.plot(x7, y7, marker="o", markersize=z7 * 10 + 6, markeredgecolor="gray", markerfacecolor="gray")

        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.pause(0.01)
        plt.clf()

    plt.show()
