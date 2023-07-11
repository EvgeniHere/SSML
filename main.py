import carball
import numpy
from carball.json_parser.game import Game
from carball.analysis.analysis_manager import AnalysisManager
import glob
import numpy as np
import keras
from keras import Sequential
from keras.layers import Dense
import sys

numpy.set_printoptions(threshold=sys.maxsize)

predict_dist = 5


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
                                'time',
                                'is_overtime'], level=1, axis=1)
    dataframe = normalize(dataframe)
    # print(dataframe.head().to_string())
    # print(dataframe.shape)
    return dataframe


X_data = []
y_data = []
first = True
number = 0
for filepath in glob.iglob('train_replays/*.replay'):
    number += 1
    if number > 10:
        break
    _json = carball.decompile_replay(filepath)
    game = Game()
    game.initialize(loaded_json=_json)

    analysis_manager = AnalysisManager(game)
    analysis_manager.create_analysis()

    print(filepath)
    print(number)

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
            if y == 50.0:
                switch_boost_num = 1
            if y == 160.0:
                switch_boost_num = 2
            if y == 190.0:
                switch_boost_num = 3
            if y == 300.0:
                switch_boost_num = 4
            if y == 310.0:
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

    for i in range(len(blue_team) - 1):
        while i < len(blue_team) - 1 and (
                blue_team[i][0] == blue_team[i + 1][0] or blue_team[i][7] == blue_team[i + 1][7] or blue_team[i][14] ==
                blue_team[i + 1][14] or
                orange_team[i][0] == orange_team[i + 1][0] or orange_team[i][7] == orange_team[i + 1][7] or orange_team[i][
                    14] == orange_team[i + 1][14] or ball[i][0] == ball[i + 1][0]):

            if any(boost_pads[i, j] != boost_pads[i + 1, j] for j in range(len(boost_pads[i]))):
                i += 1
                continue

            boost_pads = np.delete(boost_pads, i + 1, 0)
            blue_team = np.delete(blue_team, i + 1, 0)
            orange_team = np.delete(orange_team, i + 1, 0)
            ball = np.delete(ball, i + 1, 0)

    for i in range(len(game.goals)):
        start = 0
        if i > 0:
            start = game.goals[i-1].frame_number
        for player_num in range(3):
            data = np.concatenate((np.concatenate((np.concatenate((blue_team[start:game.goals[i].frame_number, (player_num*7):(player_num*7 + 7)], np.concatenate((blue_team[start:game.goals[i].frame_number, :(player_num*7)], blue_team[start:game.goals[i].frame_number, (player_num*7 + 7):]), axis=1)), axis=1), orange_team[start:game.goals[i].frame_number]), axis=1), ball[start:game.goals[i].frame_number]), axis=1)
            data = np.concatenate((data, boost_pads[start:game.goals[i].frame_number]), axis=1)
            if first:
                X_data = data[:-predict_dist]
                y_data = data[predict_dist:, :6]
                first = False
            else:
                X_data = np.concatenate((X_data, data[:-predict_dist]), axis=0)
                y_data = np.concatenate((y_data, data[predict_dist:, :6]), axis=0)

        for player_num in range(3):
            data = np.concatenate((np.concatenate((np.concatenate((orange_team[start:game.goals[i].frame_number, (player_num*7):(player_num*7 + 7)], np.concatenate((orange_team[start:game.goals[i].frame_number, :(player_num*7)], orange_team[start:game.goals[i].frame_number, (player_num*7 + 7):]), axis=1)), axis=1), blue_team[start:game.goals[i].frame_number]), axis=1), ball[start:game.goals[i].frame_number]), axis=1)
            data[:, [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 21, 22, 24, 25, 28, 29, 31, 32, 35, 36, 38, 39, 42, 43, 45, 46]] *= -1
            data[:, [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 21, 22, 24, 25, 28, 29, 31, 32, 35, 36, 38, 39, 42, 43, 45, 46]] += 1
            data = np.concatenate((data, boost_pads[start:game.goals[i].frame_number]), axis=1)

            X_data = np.concatenate((X_data, data[:-predict_dist]), axis=0)
            y_data = np.concatenate((y_data, data[predict_dist:, :6]), axis=0)


X_data = X_data.astype(np.float32)
y_data = y_data.astype(np.float32)

model = Sequential()
model.add(Dense(100, input_shape=(54,), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_data, y_data, epochs=30, batch_size=100)

model.save('model')
